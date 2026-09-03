"""`history.point_frames` indexes the observed cube on the EDGE convention.

Unlike a forecast grid (`cache.GridSpec`, cell-centre bounds), the observed
cube's `bounds` are the raster's outer pixel EDGES: produce_observed.py
rasterises with rasterio `from_bounds`, and the composite binning in
radar_single_site._polar_geometry is edge-based. So a lookup is the floor of
the fractional edge index over the whole pixel count — NOT
`GridSpec.latlon_to_cell`, which reads its bounds as centres and would shift
a quarter of all lookups by one cell. This file pins that difference so the
two conventions don't get "unified" in the wrong direction.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest

BOUNDS = (1.5, 48.9, 7.5, 52.5)  # west, south, east, north — pixel EDGES
SHAPE = (100, 100)


@pytest.fixture
def history_mod(tmp_path, monkeypatch):
    """history.py on a fresh observed cube whose every cell holds its own
    column index, so the value a lookup returns identifies the column it hit
    (point_frames maxes over a 3x3 block, so column c reads back as c+1)."""
    npz = tmp_path / "observed.npz"
    now = int(datetime.now(UTC).timestamp())
    ramp = np.tile(np.arange(SHAPE[1], dtype="float32"), (SHAPE[0], 1))
    np.savez(
        npz,
        times=np.asarray([now], dtype="int64"),
        rates=ramp[None, ...],
        bounds=np.asarray(BOUNDS, dtype="float64"),
    )
    monkeypatch.setenv("PLUVIO_OBSERVED_NPZ", str(npz))
    monkeypatch.setenv("PLUVIO_OBSERVED_HI", str(tmp_path / "missing_hi.npy"))
    import pluvio_backend.history as h

    importlib.reload(h)
    h._CACHE.update(mtime=None, data=None)
    h._HI_CACHE.update(mtime=None, data=None)
    yield h
    # The module latches its paths at import, so put it back on the real
    # environment (monkeypatch has already restored the env vars themselves)
    # rather than leaving tmp_path baked in for the rest of the session.
    importlib.reload(h)
    h._CACHE.update(mtime=None, data=None)
    h._HI_CACHE.update(mtime=None, data=None)


def _rate_at(mod, lat: float, lon: float) -> float:
    frames = mod.point_frames(lat, lon, span_min=60)
    assert frames is not None and len(frames) == 1
    return frames[0][1]


def test_point_frames_uses_edge_indexing_not_centre_indexing(history_mod):
    """A point 0.6 of a pixel east of the west edge is still in column 0 on
    the edge convention (max over cols 0-1 = 1.0). Reading the same bounds as
    cell centres would put it in column 1 (max over 0-2 = 2.0) — the
    regression this pins."""
    d_lon = (BOUNDS[2] - BOUNDS[0]) / SHAPE[1]
    mid_lat = (BOUNDS[1] + BOUNDS[3]) / 2
    assert _rate_at(history_mod, mid_lat, BOUNDS[0] + 0.6 * d_lon) == pytest.approx(1.0)
    # Sanity anchors: the first and last pixel of the row.
    assert _rate_at(history_mod, mid_lat, BOUNDS[0] + 0.1 * d_lon) == pytest.approx(1.0)
    assert _rate_at(history_mod, mid_lat, BOUNDS[2] - 0.1 * d_lon) == pytest.approx(SHAPE[1] - 1)


def test_point_frames_rejects_points_outside_the_bounds(history_mod):
    with pytest.raises(ValueError, match="outside observed bounds"):
        history_mod.point_frames(40.0, 4.0, span_min=60)
