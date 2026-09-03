"""`history.point_frames` shares the one backend pixel convention (1.13).

The observed cube's `bounds` are cell-CENTRE bounds; point lookups therefore
go through `GridSpec.latlon_to_cell`, which accepts the half-cell margin
around the boundary cells' centres (the cube's real footprint) and rejects
anything past it. Before 1.13 this function mixed centre bounds with a
whole-pixel-count index and rejected everything outside the centre envelope.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime

import numpy as np
import pytest

BOUNDS = (1.5, 48.9, 7.5, 52.5)  # west, south, east, north — cell centres
SHAPE = (100, 100)


@pytest.fixture
def history_mod(tmp_path, monkeypatch):
    """history.py with a fresh single-frame observed cube, no hi-res cube."""
    npz = tmp_path / "observed.npz"
    now = int(datetime.now(UTC).timestamp())
    rates = np.zeros((1, *SHAPE), dtype="float32")
    rates[0, 50, 50] = 7.5
    np.savez(
        npz,
        times=np.asarray([now], dtype="int64"),
        rates=rates,
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
    # instead of leaving tmp_path baked into it for the rest of the session.
    importlib.reload(h)
    h._CACHE.update(mtime=None, data=None)
    h._HI_CACHE.update(mtime=None, data=None)


def test_point_frames_reads_the_cell_gridspec_reports(history_mod):
    """The rain cell is found from its own centre coordinates, computed by the
    shared GridSpec — not by a hand-rolled inverse."""
    from pluvio_backend.cache import GridSpec

    grid = GridSpec(
        bounds={"west": BOUNDS[0], "south": BOUNDS[1], "east": BOUNDS[2], "north": BOUNDS[3]},
        shape=SHAPE,
    )
    lat, lon = grid.cell_center_latlon(50, 50)
    frames = history_mod.point_frames(lat, lon, span_min=60)
    assert frames is not None and len(frames) == 1
    assert frames[0][1] == pytest.approx(7.5)


def test_point_frames_accepts_the_half_cell_edge_margin(history_mod):
    """A point just outside the centre envelope but inside the cube's actual
    footprint belongs to the boundary cell, so it is served (0 mm/h here), not
    rejected."""
    frames = history_mod.point_frames(BOUNDS[1] - 1e-3, BOUNDS[0] - 1e-3, span_min=60)
    assert frames is not None and len(frames) == 1
    assert frames[0][1] == pytest.approx(0.0)


def test_point_frames_rejects_points_outside_the_footprint(history_mod):
    with pytest.raises(ValueError, match="outside observed bounds"):
        history_mod.point_frames(40.0, 4.0, span_min=60)
