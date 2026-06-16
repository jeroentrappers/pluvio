from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pluvio_backend import schedules
from pluvio_backend.cache import ForecastCache
from pluvio_backend.config import Settings
from pluvio_backend.inference_worker import run_tick


def _const_infer(value: float):
    """A BandInference stub returning a uniform field for the given band."""

    def infer(client, base_url, grid, band_name):
        band = schedules.band(band_name)
        arr = np.full((band.n_leads, *grid.shape), value, dtype="float32")
        return arr, datetime(2026, 6, 15, 12, 0, tzinfo=UTC)

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
    assert sprite["tile_w"] == cache.grid.shape[1]
    assert sprite["tile_h"] == cache.grid.shape[0]
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
