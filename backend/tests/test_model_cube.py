"""Serving the seamless forecast cube (rec #4): model.py reads the
producer-agnostic `model_forecast.npz` and serves every band from it, with
source/confidence provenance, falling back gracefully when it's stale/absent."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest

from pluvio_backend import schedules
from pluvio_backend.cache import DEFAULT_GRID, ForecastCache


def _write_cube(path, issue_epoch, *, producer="classical"):
    leads = [0, 10, 60, 120, 360, 1440, 14400]
    H, W = DEFAULT_GRID.shape
    rates = np.stack([np.full((H, W), float(i), dtype="float32") for i in range(len(leads))])
    source = ["nowcast", "nowcast", "nowcast", "nowcast", "blend", "nwp", "nwp"]
    conf = np.linspace(0.95, 0.1, len(leads)).astype("float32")
    np.savez(
        path,
        leads=np.asarray(leads, dtype="int16"),
        rates=rates,
        source=np.asarray(source),
        confidence=conf,
        bounds=np.asarray([1.5, 48.9, 7.5, 52.5], dtype="float64"),
        issue_epoch=np.int64(issue_epoch),
        producer=np.asarray(producer),
        engine=np.asarray("pysteps"),
    )


@pytest.fixture
def model_mod(tmp_path, monkeypatch):
    npz = tmp_path / "model_forecast.npz"
    monkeypatch.setenv("PLUVIO_MODEL_FORECAST_NPZ", str(npz))
    monkeypatch.setenv("PLUVIO_MODEL_NOWCAST_NPZ", str(tmp_path / "missing.npz"))
    import pluvio_backend.model as m
    importlib.reload(m)  # pick up the patched env paths
    return m, npz


def test_cube_serves_every_band(model_mod):
    m, npz = model_mod
    _write_cube(npz, int(datetime.now(UTC).timestamp()))
    for band_name in schedules.BANDS:
        out, _issued, used_grid = m.model_band(None, "", DEFAULT_GRID, band_name)
        band = schedules.band(band_name)
        assert out.shape == (band.n_leads, *DEFAULT_GRID.shape)
        assert np.isfinite(out).all() and (out >= 0).all()
        # the cube's own recorded bounds equal DEFAULT_GRID here, so the
        # returned grid should match it exactly (1.13).
        assert used_grid.shape == DEFAULT_GRID.shape
        assert used_grid.bounds == DEFAULT_GRID.bounds


def test_provenance_tags_each_band(model_mod):
    m, npz = model_mod
    _write_cube(npz, int(datetime.now(UTC).timestamp()))
    now = m.band_provenance("nowcast")
    assert now["source"] == "nowcast" and now["producer"] == "classical"
    assert 0.0 <= now["confidence"] <= 1.0
    long = m.band_provenance("long")
    assert long["source"] == "nwp"
    # confidence must not increase with lead
    assert long["confidence"] <= now["confidence"]


def test_stale_cube_falls_back_to_stub(model_mod, monkeypatch):
    m, npz = model_mod
    _write_cube(npz, int(datetime.now(UTC).timestamp()) - 10 * 3600)  # 10 h old

    called = {}

    def fake_stub(client, base_url, grid, band_name):
        called["band"] = band_name
        band = schedules.band(band_name)
        return np.zeros((band.n_leads, *grid.shape), dtype="float32"), datetime.now(UTC)

    monkeypatch.setattr(m, "stub_band", fake_stub)
    _out, _issued, used_grid = m.model_band(None, "", DEFAULT_GRID, "nowcast")
    assert called.get("band") == "nowcast"  # degraded to stub
    assert used_grid is DEFAULT_GRID  # stub path always serves the caller's grid
    assert m.band_provenance("nowcast") is None


# -- 1.9 prerequisite: the nowcast (v2) npz's own bounds/shape, not DEFAULT_GRID ---


def _write_nowcast_npz(path, issue_epoch, *, shape, bounds=None):
    """Mirrors research/model/infer_latest.py's output. `bounds=None` writes
    an npz with NO `bounds` key at all — a legacy artifact that predates the
    Grid contract (1.1)."""
    leads = [0, 30, 60, 90, 120]
    H, W = shape
    rates = np.stack([np.full((H, W), float(i), dtype="float32") for i in range(len(leads))])
    if bounds is not None:
        np.savez(path, leads=np.asarray(leads, dtype="int16"), rates=rates,
                 issue_epoch=np.int64(issue_epoch), bounds=np.asarray(bounds, dtype="float64"))
    else:
        np.savez(path, leads=np.asarray(leads, dtype="int16"), rates=rates,
                 issue_epoch=np.int64(issue_epoch))


@pytest.fixture
def nowcast_mod(tmp_path, monkeypatch):
    npz = tmp_path / "model_nowcast.npz"
    monkeypatch.setenv("PLUVIO_MODEL_NOWCAST_NPZ", str(npz))
    monkeypatch.setenv("PLUVIO_MODEL_FORECAST_NPZ", str(tmp_path / "missing_cube.npz"))
    monkeypatch.setenv("PLUVIO_OBSERVED_NPZ", str(tmp_path / "missing_observed.npz"))
    import pluvio_backend.model as m
    importlib.reload(m)
    return m, npz


def test_full_benelux_npz_accepted_and_bounds_propagated_to_api_response(nowcast_mod, tmp_path):
    """A 192x192 npz (a v3/full-Benelux infer_latest run) carrying its own
    `bounds`, different from DEFAULT_GRID, must be served on ITS grid rather
    than crashing or being silently resampled onto the legacy default — and
    that grid must reach the cache's grid.json, which the API's `bounds`
    field reads directly (pluvio_backend.api.forecast)."""
    m, npz = nowcast_mod
    full_bounds = (1.0, 47.5, 8.5, 53.5)  # wider than DEFAULT_GRID
    full_shape = (192, 192)
    _write_nowcast_npz(npz, int(datetime.now(UTC).timestamp()), shape=full_shape,
                       bounds=full_bounds)

    # Called with the caller's (still legacy) grid, exactly as inference_worker
    # does today — model.py must not assume the npz matches it.
    out, _issued_at, used_grid = m.model_band(None, "", DEFAULT_GRID, "nowcast")

    n_leads = schedules.band("nowcast").n_leads
    assert out.shape == (n_leads, *full_shape)  # not DEFAULT_GRID.shape — no crash, no crop
    assert np.isfinite(out).all() and (out >= 0).all()
    assert used_grid.shape == full_shape
    assert used_grid.bounds == {"west": 1.0, "south": 47.5, "east": 8.5, "north": 53.5}

    # cache.write_band/write_grid_metadata accept the mismatched-shape band
    # via the `grid` override (1.13) — even though the cache itself is still
    # configured at the legacy DEFAULT_GRID (pre-1.9).
    cache = ForecastCache(tmp_path / "cache_root", grid=DEFAULT_GRID)
    snap = cache.new_snapshot_dir()
    cache.write_band(snap, "nowcast", out, grid=used_grid)
    cache.write_grid_metadata(snap, model_version="test", grid=used_grid)
    cache.mark_complete(snap)
    cache.swap_latest(snap)

    meta = cache.latest_metadata()
    assert meta is not None
    assert meta["grid"]["bounds"] == used_grid.bounds
    assert meta["grid"]["shape"] == list(full_shape)


def test_legacy_npz_without_bounds_still_served_on_caller_grid(nowcast_mod):
    """An npz written before the Grid contract (1.1) — no `bounds` key at
    all — still serves fine, on the legacy DEFAULT_BOUNDS/DEFAULT_GRID_SHAPE
    the caller passes in."""
    m, npz = nowcast_mod
    _write_nowcast_npz(npz, int(datetime.now(UTC).timestamp()), shape=DEFAULT_GRID.shape,
                       bounds=None)

    out, _issued_at, used_grid = m.model_band(None, "", DEFAULT_GRID, "nowcast")

    n_leads = schedules.band("nowcast").n_leads
    assert out.shape == (n_leads, *DEFAULT_GRID.shape)
    assert used_grid is DEFAULT_GRID
    assert used_grid.bounds == DEFAULT_GRID.bounds


def test_npz_bounds_but_shape_matches_default_grid_round_trips_via_run_tick(nowcast_mod,
                                                                            tmp_path):
    """The common near-term case: the npz already carries `bounds` (1.1) but
    they equal the cache's own grid (nothing has moved yet) — full run_tick
    plumbing (point shards + sprite + publish) must work unchanged."""
    from pluvio_backend.config import Settings
    from pluvio_backend.inference_worker import run_tick

    m, npz = nowcast_mod
    _write_nowcast_npz(npz, int(datetime.now(UTC).timestamp()), shape=DEFAULT_GRID.shape,
                       bounds=(DEFAULT_GRID.bounds["west"], DEFAULT_GRID.bounds["south"],
                               DEFAULT_GRID.bounds["east"], DEFAULT_GRID.bounds["north"]))

    settings = Settings(cache_root=tmp_path / "cache_root")
    import pluvio_backend.inference_worker as worker

    orig_get_settings = worker.get_settings
    worker.get_settings = lambda: settings  # type: ignore[assignment]
    try:
        summary = run_tick("nowcast", infer=m.model_band)
    finally:
        worker.get_settings = orig_get_settings

    assert summary["published"] is True
    cache = ForecastCache(settings.cache_root)
    meta = cache.latest_metadata()
    assert meta is not None
    assert meta["grid"]["bounds"] == DEFAULT_GRID.bounds
