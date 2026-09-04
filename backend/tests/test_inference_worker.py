from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pluvio_backend import schedules
from pluvio_backend.cache import ForecastCache, GridSpec
from pluvio_backend.config import Settings
from pluvio_backend.inference_worker import run_tick


def _png_size(path) -> tuple[int, int]:
    """(width, height) straight out of a PNG's IHDR — no image library needed."""
    header = path.read_bytes()[16:24]
    return int.from_bytes(header[:4], "big"), int.from_bytes(header[4:], "big")


def _const_infer(value: float):
    """A BandInference stub returning a uniform field for the given band."""

    def infer(client, base_url, grid, band_name):
        band = schedules.band(band_name)
        arr = np.full((band.n_leads, *grid.shape), value, dtype="float32")
        return arr, datetime(2026, 6, 15, 12, 0, tzinfo=UTC), grid

    return infer


@pytest.fixture(autouse=True)
def _settings(tmp_path, monkeypatch):
    s = Settings(cache_root=tmp_path)
    monkeypatch.setattr("pluvio_backend.inference_worker.get_settings", lambda: s)
    return s


def test_longer_band_only_tick_does_not_publish(_settings) -> None:
    """A short-band tick with no nowcast anywhere must not become `latest`."""
    summary = run_tick("short", infer=_const_infer(0.2))
    assert summary["published"] is False
    cache = ForecastCache(_settings.cache_root)
    assert cache.latest_snapshot() is None  # nothing published


def test_nowcast_tick_publishes(_settings) -> None:
    summary = run_tick("nowcast", infer=_const_infer(1.0))
    assert summary["published"] is True
    cache = ForecastCache(_settings.cache_root)
    assert cache.latest_snapshot() is not None
    df = cache.read_point(50.85, 4.35)
    assert df is not None and set(df["band"]) == {"nowcast"}


def test_tick_writes_sprite_sheet_and_index(_settings) -> None:
    """Publishing composes one sprite-sheet PNG of every frame and records its
    layout (tile dims + band:lead → tile index) in grid.json, so the client can
    animate the whole horizon from a single download."""
    import json

    run_tick("nowcast", infer=_const_infer(1.0))
    run_tick("short", infer=_const_infer(0.2))
    cache = ForecastCache(_settings.cache_root)
    snap = cache.latest_snapshot()
    assert snap is not None
    assert (snap / "sprite.png").exists()

    meta = json.loads((snap / "grid.json").read_text())
    sprite = meta["sprite"]
    # Tiles carry the overlay render resolution, not the model grid: frames are
    # bicubic-upsampled to the radar composite's angular pixel density so both
    # sides of the t=0 seam share pixel scale (display-only). The client reads
    # tile_w/tile_h out of this index, so what it declares must be what the
    # sheet actually holds — and a single-frame overlay is one such tile.
    overlay_w, overlay_h = _png_size(snap / "overlays" / "nowcast" / "0.png")
    assert (sprite["tile_w"], sprite["tile_h"]) == (overlay_w, overlay_h)
    # the sheet is exactly cols x rows of those tiles
    sheet_w, sheet_h = _png_size(snap / "sprite.png")
    assert sheet_w == sprite["cols"] * sprite["tile_w"]
    assert sheet_h == sprite["rows"] * sprite["tile_h"]
    # upsampling only ever adds pixels — never renders below the model grid
    assert sprite["tile_w"] >= cache.grid.shape[1]
    assert sprite["tile_h"] >= cache.grid.shape[0]
    # every nowcast + short lead has a unique tile index, ordered by lead
    idx = sprite["index"]
    assert "nowcast:0" in idx and f"short:{schedules.band('short').lead_min_start}" in idx
    assert len(set(idx.values())) == len(idx) == sprite["count"]
    # tiles fit the declared grid
    assert sprite["count"] <= sprite["cols"] * sprite["rows"]


def test_later_short_tick_folds_in_nowcast_and_serves_full_horizon(_settings) -> None:
    """After a nowcast exists, a short tick publishes a snapshot carrying both
    bands — so the API can serve the extended horizon from one snapshot."""
    run_tick("nowcast", infer=_const_infer(1.0))
    summary = run_tick("short", infer=_const_infer(0.2))
    assert summary["published"] is True
    assert set(summary["bands"]) >= {"nowcast", "short"}

    cache = ForecastCache(_settings.cache_root)
    df = cache.read_point(50.85, 4.35)
    assert df is not None
    assert {"nowcast", "short"} <= set(df["band"])
    # Long-range leads are present and ordered past the nowcast horizon.
    assert df["lead_min"].max() >= schedules.band("short").lead_min_start


# -- 1.13/1.9: a band served on its own grid keeps its footprint -----------

FULL_BENELUX = GridSpec(
    bounds={"west": 1.0, "east": 8.5, "south": 47.5, "north": 53.5},
    shape=(192, 192),
)


def _infer_on(spec: GridSpec, value: float):
    """A BandInference stub that ignores the caller's grid and reports its
    own — exactly what model_band does with a v3 npz's `bounds`."""

    def infer(client, base_url, grid, band_name):
        band = schedules.band(band_name)
        arr = np.full((band.n_leads, *spec.shape), value, dtype="float32")
        return arr, datetime(2026, 6, 15, 12, 0, tzinfo=UTC), spec

    return infer


def test_off_grid_nowcast_band_is_served_and_records_its_own_footprint(_settings) -> None:
    """A nowcast band on a 192x192 full-Benelux grid, while the cache is still
    at the legacy DEFAULT_GRID: the band and its overlays are written on that
    grid and grid.json reports it, so the client places the overlay correctly.
    Point shards and the sprite still key off the cache grid, so they are
    skipped with a warning rather than crashing on the shape mismatch — 1.9
    removes the divergence by widening the cache grid."""
    import json

    summary = run_tick("nowcast", infer=_infer_on(FULL_BENELUX, 1.0))
    assert summary["published"] is True
    cache = ForecastCache(_settings.cache_root)
    snap = cache.latest_snapshot()
    assert snap is not None
    served = cache.read_band("nowcast")
    assert served is not None and served.shape[-2:] == FULL_BENELUX.shape

    meta = json.loads((snap / "grid.json").read_text())
    assert meta["grid"]["bounds"] == FULL_BENELUX.bounds
    assert meta["grid"]["shape"] == list(FULL_BENELUX.shape)
    assert meta["sprite"] is None  # no uniform-grid band to fold into a sheet
    assert not (snap / "points").exists()


def test_a_later_band_tick_keeps_the_recorded_footprint(_settings) -> None:
    """A short-band tick reusing the same snapshot must not stamp the cache's
    legacy default over the footprint the nowcast band recorded."""
    import json

    run_tick("nowcast", infer=_infer_on(FULL_BENELUX, 1.0))
    run_tick("short", infer=_const_infer(0.2))
    cache = ForecastCache(_settings.cache_root)
    snap = cache.latest_snapshot()
    assert snap is not None
    meta = json.loads((snap / "grid.json").read_text())
    assert meta["grid"]["bounds"] == FULL_BENELUX.bounds


def test_grid_json_records_each_bands_own_footprint(_settings) -> None:
    """1.9: a 192x192 nowcast beside a legacy-grid short band — grid.json keeps
    one entry per band, so the API labels each band's overlays by the grid
    they were rendered on instead of the snapshot-wide footprint."""
    import json

    run_tick("nowcast", infer=_infer_on(FULL_BENELUX, 1.0))
    run_tick("short", infer=_const_infer(0.2))
    cache = ForecastCache(_settings.cache_root)
    snap = cache.latest_snapshot()
    assert snap is not None
    meta = json.loads((snap / "grid.json").read_text())
    assert meta["bands"]["nowcast"]["bounds"] == FULL_BENELUX.bounds
    assert meta["bands"]["nowcast"]["shape"] == list(FULL_BENELUX.shape)
    assert meta["bands"]["short"]["shape"] == list(cache.grid.shape)
    assert meta["bands"]["short"]["bounds"] == cache.grid.bounds
    assert "edge_bounds" in meta["bands"]["short"]
