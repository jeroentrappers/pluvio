from __future__ import annotations

from datetime import UTC

import numpy as np
import pytest

from pluvio_backend import schedules
from pluvio_backend.cache import ForecastCache


@pytest.fixture
def cache(tmp_path) -> ForecastCache:
    return ForecastCache(tmp_path)


def _make_band_array(band_name: schedules.BandName) -> np.ndarray:
    band = schedules.band(band_name)
    return np.linspace(0, 5, band.n_leads * 100 * 100, dtype="float32").reshape(
        band.n_leads, 100, 100
    )


def test_write_then_read_band_roundtrip(cache: ForecastCache) -> None:
    arr = _make_band_array("nowcast")
    snap = cache.new_snapshot_dir()
    cache.write_band(snap, "nowcast", arr)
    cache.write_grid_metadata(snap, model_version="test")
    cache.mark_complete(snap)
    cache.swap_latest(snap)

    out = cache.read_band("nowcast")
    assert out is not None
    np.testing.assert_allclose(out, arr, rtol=1e-6)


def test_swap_refuses_without_status(cache: ForecastCache) -> None:
    snap = cache.new_snapshot_dir()
    cache.write_band(snap, "nowcast", _make_band_array("nowcast"))
    with pytest.raises(RuntimeError, match=r"missing status\.json"):
        cache.swap_latest(snap)


def test_latest_metadata_after_swap(cache: ForecastCache) -> None:
    snap = cache.new_snapshot_dir()
    cache.write_band(snap, "nowcast", _make_band_array("nowcast"))
    cache.write_grid_metadata(snap, model_version="v1.2.3", extras={"hello": "world"})
    cache.mark_complete(snap)
    cache.swap_latest(snap)

    meta = cache.latest_metadata()
    assert meta is not None
    assert meta["model_version"] == "v1.2.3"
    assert meta["hello"] == "world"
    assert meta["grid"]["shape"] == [100, 100]


def test_prune_keeps_only_n_complete(cache: ForecastCache) -> None:
    from datetime import datetime, timedelta

    base = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    snaps = []
    for i in range(5):
        snap = cache.new_snapshot_dir(base + timedelta(minutes=5 * i))
        cache.write_band(snap, "nowcast", _make_band_array("nowcast"))
        cache.mark_complete(snap)
        snaps.append(snap)
    cache.swap_latest(snaps[-1])

    removed = cache.prune(keep=2)
    assert removed == 3
    remaining = sorted(d.name for d in cache.root.iterdir() if d.is_dir() and d.name != "latest")
    assert len(remaining) == 2
    # `latest` survives because it's a symlink, not a directory.
    assert (cache.root / "latest").is_symlink()


def test_point_shard_returns_per_band(cache: ForecastCache) -> None:
    all_bands: dict[schedules.BandName, np.ndarray] = {
        "nowcast": _make_band_array("nowcast"),
        "short": _make_band_array("short"),
    }
    snap = cache.new_snapshot_dir()
    for name, arr in all_bands.items():
        cache.write_band(snap, name, arr)
    cache.write_point_shards(snap, all_bands)
    cache.write_grid_metadata(snap, model_version="t")
    cache.mark_complete(snap)
    cache.swap_latest(snap)

    df = cache.read_point(lat=50.85, lon=4.35)
    assert df is not None
    assert set(df["band"].unique()) >= {"nowcast", "short"}
    # Every band shows up exactly once per lead at the bucket cell.
    assert (df.groupby("band").size() == df.groupby("band")["lead_min"].nunique()).all()
