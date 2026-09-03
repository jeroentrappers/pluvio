"""geo.grid_latlon() and notebooks/_lib._resample: the 765x700 trim.

The native KNMI radar product is 765x700. notebooks/_lib._resample block-means
it down to the 100x100 analysis grid via integer factors (yh = 765//100 = 7,
xw = 700//100 = 7) and, before reshaping, trims to field[:700, :700] — i.e.
the analysis grid only ever sees the NORTHERN 700 of the native 765 rows. That
trim is exactly what geo.grid_latlon() has to reproduce when it lays its
100x100 lat/lon grid across the projected corner extent (see geo.py's
"THE TRIM" comment) — otherwise the two 100x100 grids (radar pixels vs. their
claimed lat/lon) disagree about which row is which.
"""

from __future__ import annotations

import numpy as np
import pyproj

from model import geo
from notebooks._lib import ANALYSIS_GRID, _resample


def test_grid_latlon_row0_is_north():
    lat, lon = geo.grid_latlon()
    assert lat.shape == geo.GRID
    assert lat[0].mean() > lat[-1].mean()


def test_grid_latlon_south_edge_matches_700_of_765_trim():
    lat, lon = geo.grid_latlon()

    # Reproduce geo.py's own corner-projection + trim math independently.
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", geo._PROJ4, always_xy=True)
    xs, ys = [], []
    for lon_c, lat_c in geo._CORNERS_LONLAT:
        x, y = to_xy.transform(lon_c, lat_c)
        xs.append(x)
        ys.append(y)
    xmax, ymin, ymax = max(xs), min(ys), max(ys)
    y_south = ymax - (700.0 / 765.0) * (ymax - ymin)

    # The corner quadrilateral is a rectangle in projected space, but its
    # LR corner (east, south) has the lowest latitude of the four (the
    # domain isn't symmetric about the projection's central meridian), so
    # the grid's global lat-minimum sits on the south row's EAST edge, not
    # the west one — check that edge, at the row the 700/765 trim puts there.
    to_ll = pyproj.Transformer.from_crs(geo._PROJ4, "EPSG:4326", always_xy=True)
    _, lat_expected_south = to_ll.transform(xmax, y_south)

    # geo.grid_latlon() applies a small residual calibration bias on top of
    # the raw trim (PLUVIO_GRID_LATLON_BIAS, default dlat=0). Undo it before
    # comparing to the raw-trim expectation so this test isolates the trim.
    import os
    dlat_s, _dlon_s = os.environ.get("PLUVIO_GRID_LATLON_BIAS", "0,0.07").split(",")
    dlat = float(dlat_s)

    assert abs(float(lat.min()) - dlat - lat_expected_south) < 0.01


def test_resample_last_row_averages_source_rows_693_to_699():
    # 765x700 field of row indices: column j is constant, row i == i, so the
    # block-mean of any block of rows is just the mean of those row indices.
    field = np.tile(np.arange(765, dtype="float64")[:, None], (1, 700))
    out = _resample(field, ANALYSIS_GRID)
    assert out.shape == ANALYSIS_GRID

    expected_last_row = np.arange(693, 700).mean()  # trimmed at 700: last block is rows 693..699
    assert np.allclose(out[-1], expected_last_row)

    # Row 700 (and beyond) never contributes — confirm the trim, not just the
    # last block's arithmetic, by checking that a full 700-average (which
    # would include row 699 differently only if untrimmed) matches the
    # source's own trimmed reshape directly.
    yh = 765 // 100
    trimmed = field[: 100 * yh, :700]
    manual = trimmed.reshape(100, yh, 100, 7).mean(axis=(1, 3))
    assert np.allclose(out, manual)
