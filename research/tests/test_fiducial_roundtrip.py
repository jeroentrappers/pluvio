"""A single-cell fiducial round-trips through build_input and the display
pixel<->lat/lon mapping.

This is independent of geo.grid_latlon()/scipy.griddata on purpose: it only
exercises the store's own ``bounds`` attr with the same bounds-linear
row0=north convention build_store_v3 writes (west, south, east, north; row 0
= north, col 0 = west), which is also what the frontend/tiler use to place a
pixel on the map. If ZarrCorrectionDataset ever remapped/cropped the radar
array before handing it to build_input, this would catch the misalignment.
"""

from __future__ import annotations

import numpy as np

from model.zarr_dataset import ZarrCorrectionDataset
from tests.conftest import BOUNDS, GRID_N


def _rowcol_for_latlon(lat: float, lon: float, bounds, n: int) -> tuple[int, int]:
    """Same linear mapping build_store_v3 uses to lay its grid: row 0 = north,
    col 0 = west, cell centres at linspace(edge, edge, n)."""
    west, south, east, north = bounds
    lats = np.linspace(north, south, n)
    lons = np.linspace(west, east, n)
    row = int(np.argmin(np.abs(lats - lat)))
    col = int(np.argmin(np.abs(lons - lon)))
    return row, col


def _latlon_for_rowcol(row: int, col: int, bounds, n: int) -> tuple[float, float]:
    west, south, east, north = bounds
    lats = np.linspace(north, south, n)
    lons = np.linspace(west, east, n)
    return float(lats[row]), float(lons[col])


def test_fiducial_survives_build_input_and_display_mapping(synthetic_store):
    import zarr

    # Inject a single bright cell at a known lat/lon into the store's radar
    # array (lead-0 slice), at an issue that build_index will keep.
    target_lat, target_lon = 51.2, 4.4  # inside BOUNDS
    row, col = _rowcol_for_latlon(target_lat, target_lon, BOUNDS, GRID_N)

    root = zarr.open_group(str(synthetic_store), mode="a")
    issue_idx = 10
    radar = np.asarray(root["radar"][issue_idx])
    radar[:] = 0.0
    radar[0, row, col] = 99.0  # lead-0 fiducial
    root["radar"][issue_idx] = radar

    ds = ZarrCorrectionDataset(synthetic_store)
    # Find the sample whose issue is our fiducial issue and whose newest
    # history frame (index H-1) is the issue itself (lead-0 == history[-1]).
    sample = next(s for s in ds.index if s.issue_idx == issue_idx)
    chans = ds.build_input(sample.issue_idx, sample.lead_min, sample.history_idx)

    # The newest history channel (index history_steps - 1) is the issue's
    # own lead-0 analysis — recover the fiducial cell from it.
    newest_history = chans[ds.history_steps - 1]
    found_row, found_col = np.unravel_index(np.argmax(newest_history), newest_history.shape)

    recovered_lat, recovered_lon = _latlon_for_rowcol(int(found_row), int(found_col),
                                                       BOUNDS, GRID_N)

    cell_h = abs(BOUNDS[3] - BOUNDS[1]) / (GRID_N - 1)
    cell_w = abs(BOUNDS[2] - BOUNDS[0]) / (GRID_N - 1)
    assert abs(recovered_lat - target_lat) <= cell_h + 1e-9
    assert abs(recovered_lon - target_lon) <= cell_w + 1e-9
    assert (found_row, found_col) == (row, col)
