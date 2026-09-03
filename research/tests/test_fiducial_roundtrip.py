"""A single-cell fiducial round-trips from the store through model input
assembly and independently through the two lat/lon<->pixel conventions the
production code actually uses.

Convention note (the actual contract, not an assumption): the zarr store's
``bounds`` attr (and build_store_v3's BOX) are CELL-CENTRE bounds — cell 0's
centre sits exactly on the west/north edge, cell N-1's centre on the
east/south edge (``np.linspace(edge, edge, N)``), and that's also the
convention backend.cache.GridSpec's bounds carry — its latlon_to_cell resolves
a point to the cell whose footprint contains it, exactly like
Grid.cell_of (1.13). Painters
(colormap.draw_fiducials, and any PNG/tile renderer) instead take EDGE
bounds — the outer boundary of the rendered image, half a cell further out
on every side. Converting cell-centre bounds to edge bounds means inflating
by half a cell on each side:

    edge = (west - dlon/2, south - dlat/2, east + dlon/2, north + dlat/2)

Get this backwards and every rendered pixel is off by half a cell (a whole
cell at the south/east edges) — exactly the class of bug this test is for.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import zarr

from model.zarr_dataset import ZarrCorrectionDataset

_BACKEND_SRC = pathlib.Path(__file__).resolve().parents[2] / "backend" / "src"


def _load_grid_spec():
    """Import the real backend.cache.GridSpec.

    Unlike morph.py, cache.py does ``from . import schedules`` — a
    package-relative import — so it can't be loaded standalone via
    importlib.util.spec_from_file_location the way morph.py is; it needs to
    be imported as part of the real ``pluvio_backend`` package. Add
    backend/src to sys.path (this test only reads GridSpec from it, nothing
    under backend/ is modified) and import it properly so the relative
    import resolves.
    """
    if str(_BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(_BACKEND_SRC))
    cache = importlib.import_module("pluvio_backend.cache")
    return cache.GridSpec


GridSpec = _load_grid_spec()


def test_fiducial_survives_build_input_and_both_display_conventions(synthetic_store):
    root = zarr.open_group(str(synthetic_store), mode="a")

    # The store is the source of truth for its own georeference — read
    # attrs, don't assume they agree with the array shape (that's part of
    # what's under test).
    bounds_attr = tuple(float(x) for x in root.attrs["bounds"])  # west, south, east, north
    grid_n = int(root.attrs["grid_n"])
    west, south, east, north = bounds_attr
    assert tuple(root["radar"].shape[-2:]) == (grid_n, grid_n), (
        "store attrs['grid_n'] disagrees with the radar array's own shape"
    )

    # Pick a point strictly inside one cell, off the cell boundary — a point
    # sitting exactly on a boundary is at the mercy of float rounding either
    # way, in the painter's floor() as much as in GridSpec.
    dlon = (east - west) / (grid_n - 1)
    dlat = (north - south) / (grid_n - 1)
    # Near the south/east edge: there the half-cell edge-vs-centre error
    # exceeds one cell, so a painter using centre bounds (or h instead of
    # h-1) lands in a different cell instead of being hidden by truncation.
    row_pick, col_pick = 21, 20
    lats = np.linspace(north, south, grid_n)
    lons = np.linspace(west, east, grid_n)
    target_lat = float(lats[row_pick] - 0.25 * dlat)
    target_lon = float(lons[col_pick] + 0.25 * dlon)

    # Forward mapping via the real production code, not a hand-rolled inverse.
    grid = GridSpec(
        bounds={"west": west, "east": east, "south": south, "north": north},
        shape=(grid_n, grid_n),
    )
    row, col = grid.latlon_to_cell(target_lat, target_lon)
    assert (row, col) == (row_pick, col_pick)

    issue_idx = 10
    radar = np.asarray(root["radar"][issue_idx])
    radar[:] = 0.0
    radar[0, row, col] = 99.0  # lead-0 fiducial
    root["radar"][issue_idx] = radar

    ds = ZarrCorrectionDataset(synthetic_store)
    sample = next(s for s in ds.index if s.issue_idx == issue_idx)
    chans = ds.build_input(sample.issue_idx, sample.lead_min, sample.history_idx)

    # The newest history channel (index history_steps - 1) is the issue's
    # own lead-0 analysis — recover the fiducial cell from it.
    newest_history = chans[ds.history_steps - 1]
    found_row, found_col = np.unravel_index(np.argmax(newest_history), newest_history.shape)
    assert (int(found_row), int(found_col)) == (row, col)

    # Painter-side agreement: colormap.draw_fiducials
    # (backend/src/pluvio_backend/colormap.py:131-150) is not importable
    # here without refactoring backend code, which is out of scope for this
    # branch — replicate its two index lines verbatim (same variable names,
    # same formula) against the EDGE-inflated bounds, and check they land on
    # the same cell the store/GridSpec agreed on.
    wst, sth, est, nth = west - dlon / 2, south - dlat / 2, east + dlon / 2, north + dlat / 2
    h = w = grid_n
    # --- verbatim from colormap.draw_fiducials ---
    r = int((nth - target_lat) / (nth - sth) * h)
    c = int((target_lon - wst) / (est - wst) * w)
    # --- end verbatim block ---
    assert (r, c) == (row, col)
