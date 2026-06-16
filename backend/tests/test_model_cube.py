"""Serving the seamless forecast cube (rec #4): model.py reads the
producer-agnostic `model_forecast.npz` and serves every band from it, with
source/confidence provenance, falling back gracefully when it's stale/absent."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest

from pluvio_backend import schedules
from pluvio_backend.cache import DEFAULT_GRID


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
        out, issued = m.model_band(None, "", DEFAULT_GRID, band_name)
        band = schedules.band(band_name)
        assert out.shape == (band.n_leads, *DEFAULT_GRID.shape)
        assert np.isfinite(out).all() and (out >= 0).all()


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
    out, _ = m.model_band(None, "", DEFAULT_GRID, "nowcast")
    assert called.get("band") == "nowcast"  # degraded to stub
    assert m.band_provenance("nowcast") is None
