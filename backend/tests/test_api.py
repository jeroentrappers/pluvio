from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pluvio_backend import schedules
from pluvio_backend.api import create_app
from pluvio_backend.cache import ForecastCache, GridSpec
from pluvio_backend.config import Settings


def _seed_cache(root) -> None:
    """Populate a cache with a nowcast + short band so the API has data."""
    cache = ForecastCache(root)
    all_bands: dict[schedules.BandName, np.ndarray] = {}
    bands: tuple[schedules.BandName, ...] = ("nowcast", "short")
    for name in bands:
        band = schedules.band(name)
        # Put a recognisable rain blob near Brussels so a point query is non-trivial.
        arr = np.zeros((band.n_leads, 100, 100), dtype="float32")
        arr[:, 40:50, 45:55] = 3.0  # moderate rain patch
        all_bands[name] = arr

    snap = cache.new_snapshot_dir()
    for name, arr in all_bands.items():
        cache.write_band(snap, name, arr)
        cache.write_overlays(snap, name, arr)
    cache.write_point_shards(snap, all_bands)
    cache.write_grid_metadata(snap, model_version="test-api")
    cache.mark_complete(snap)
    cache.swap_latest(snap)


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(cache_root=tmp_path)
    _seed_cache(tmp_path)
    return TestClient(create_app(settings))


@pytest.fixture
def empty_client(tmp_path) -> TestClient:
    settings = Settings(cache_root=tmp_path)
    return TestClient(create_app(settings))


def test_healthz_ok_when_fresh(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["model_version"] == "test-api"
    assert body["snapshot"] is not None


def test_healthz_empty_cache(empty_client: TestClient) -> None:
    r = empty_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "empty"


def test_forecast_returns_frames(client: TestClient) -> None:
    r = client.get("/v1/forecast", params={"lat": 50.85, "lon": 4.35})
    assert r.status_code == 200
    body = r.json()
    assert body["location"] == {"lat": 50.85, "lon": 4.35}
    assert len(body["frames"]) > 0
    f0 = body["frames"][0]
    assert {"band", "lead_min", "valid_time", "rate_mm_per_h", "overlay_url"} <= f0.keys()


def test_forecast_horizon_filter(client: TestClient) -> None:
    r = client.get("/v1/forecast", params={"lat": 50.85, "lon": 4.35, "horizon_min": 60})
    assert r.status_code == 200
    leads = [f["lead_min"] for f in r.json()["frames"]]
    assert max(leads) <= 60


def test_forecast_out_of_bounds(client: TestClient) -> None:
    # Tokyo — outside the Benelux grid.
    r = client.get("/v1/forecast", params={"lat": 35.0, "lon": 139.0})
    assert r.status_code == 400


def test_forecast_empty_cache_503(empty_client: TestClient) -> None:
    r = empty_client.get("/v1/forecast", params={"lat": 50.85, "lon": 4.35})
    assert r.status_code == 503


def test_overlay_png(client: TestClient) -> None:
    r = client.get("/v1/overlay/nowcast/30.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "max-age" in r.headers.get("cache-control", "")


def test_overlay_unknown_band_404(client: TestClient) -> None:
    r = client.get("/v1/overlay/bogus/30.png")
    assert r.status_code in {404, 422}


def test_animation_manifest(client: TestClient) -> None:
    r = client.get("/v1/animation/manifest.json", params={"band": "nowcast"})
    assert r.status_code == 200
    body = r.json()
    assert body["band"] == "nowcast"
    assert body["bounds"] is not None
    assert len(body["frames"]) > 0
    assert body["frames"][0]["url"].endswith(".png?t=" + body["snapshot"])


def test_ws_sends_current_snapshot_on_connect(client: TestClient) -> None:
    """A client connecting to /v1/ws is told the current snapshot immediately,
    so it can refetch the new prediction without polling."""
    with client.websocket_connect("/v1/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "snapshot"
    assert msg["snapshot"] is not None
    assert msg["model_version"] == "test-api"


def test_ws_on_empty_cache_connects_without_snapshot(empty_client: TestClient) -> None:
    """With no published snapshot yet, the socket still opens cleanly (no crash);
    it just sends nothing until the first prediction lands. Open and close
    without receiving (there's nothing to receive, and a blocking receive would
    hang)."""
    with empty_client.websocket_connect("/v1/ws"):
        pass


# -- 1.13/1.9: the API reports the grid a band was actually served on -------

FULL_BENELUX = GridSpec(
    bounds={"west": 1.0, "east": 8.5, "south": 47.5, "north": 53.5},
    shape=(192, 192),
)


def test_manifest_reports_the_grid_the_band_was_served_on(tmp_path) -> None:
    """A band produced on a 192x192 full-Benelux footprint (what a v3 npz
    carries in its own `bounds`, see model._grid_from_npz) is written and
    rendered on that grid, and the API hands the client those bounds — not
    the cache's still-legacy DEFAULT_GRID. The client places the overlay by
    exactly this field, so it is the thing 1.9 turns on.

    Uses the "short" band purely to keep the test cheap (9 leads instead of
    the nowcast's 61); the manifest's `bounds` come from grid.json either way.
    """
    band: schedules.BandName = "short"
    n_leads = schedules.band(band).n_leads
    arr = np.zeros((n_leads, *FULL_BENELUX.shape), dtype="float32")
    arr[:, 90:100, 90:100] = 2.0

    cache = ForecastCache(tmp_path)
    snap = cache.new_snapshot_dir()
    cache.write_band(snap, band, arr, grid=FULL_BENELUX)
    assert cache.write_overlays(snap, band, arr, grid=FULL_BENELUX) == n_leads
    cache.write_grid_metadata(snap, model_version="test-api", grid=FULL_BENELUX)
    cache.mark_complete(snap)
    cache.swap_latest(snap)

    client = TestClient(create_app(Settings(cache_root=tmp_path)))
    r = client.get("/v1/animation/manifest.json", params={"band": band})
    assert r.status_code == 200
    body = r.json()
    assert body["bounds"] == FULL_BENELUX.bounds
    assert body["bounds"] != cache.grid.bounds  # the legacy default, still in place
    assert len(body["frames"]) == n_leads


def test_scoreboard_routes_serve_the_generated_files_or_404(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pluvio_backend import api as api_mod
    from pluvio_backend.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "scoreboard_dir", tmp_path)
    app = api_mod.create_app()
    client = TestClient(app)
    assert client.get("/v1/scoreboard/").status_code == 404
    (tmp_path / "index.html").write_text("<title>Pluvio scoreboard</title>")
    (tmp_path / "2026" / "09").mkdir(parents=True)
    (tmp_path / "2026" / "09" / "03.json").write_text('{"day": "2026-09-03"}')
    r = client.get("/v1/scoreboard/")
    assert (
        r.status_code == 200
        and "scoreboard" in r.text
        and r.headers["content-type"].startswith("text/html")
    )
    assert client.get("/v1/scoreboard/2026/09/03.json").json()["day"] == "2026-09-03"
    assert client.get("/v1/scoreboard/2026/09/04.json").status_code == 404
