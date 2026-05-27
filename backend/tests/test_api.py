from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pluvio_backend import schedules
from pluvio_backend.api import create_app
from pluvio_backend.cache import ForecastCache
from pluvio_backend.config import Settings


def _seed_cache(root) -> None:
    """Populate a cache with a nowcast + short band so the API has data."""
    cache = ForecastCache(root)
    all_bands = {}
    for name in ("nowcast", "short"):
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
