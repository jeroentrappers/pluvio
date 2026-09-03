"""geo.grid_latlon()/analysis_grid_dst() and notebooks/_lib._resample: the
765x700 trim, on BOTH sides of the store.

The native KNMI radar product is 765x700. notebooks/_lib._resample block-means
it down to the 100x100 analysis grid via integer factors (yh = 765//100 = 7,
xw = 700//100 = 7) and, before reshaping, trims to field[:700, :700] — i.e.
the analysis grid only ever sees the NORTHERN 700 of the native 765 rows.
geo.grid_latlon() (radar/truth side) and geo.analysis_grid_dst() (aux/reproject
side) each carry their OWN independent copy of that same 700/765 trim
constant — the 2026-09-02 input-validation incident was exactly these two
copies drifting apart. Both are exercised here.
"""

from __future__ import annotations

import numpy as np
import pyproj
from model import geo
from notebooks._lib import ANALYSIS_GRID, _resample

# geo.grid_latlon() takes the registration bias as an explicit argument
# (1.11) — pass it directly instead of monkeypatching PLUVIO_GRID_LATLON_BIAS
# and clearing a process-wide cache.
_ZERO_BIAS = (0.0, 0.0)


def _corner_xy() -> tuple[float, float, float, float]:
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", geo._PROJ4, always_xy=True)
    xs, ys = [], []
    for lon_c, lat_c in geo._CORNERS_LONLAT:
        x, y = to_xy.transform(lon_c, lat_c)
        xs.append(x)
        ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def test_grid_latlon_row0_is_north():
    lat, _lon = geo.grid_latlon(bias=_ZERO_BIAS)
    assert lat.shape == geo.GRID
    assert lat[0].mean() > lat[-1].mean()


def test_grid_latlon_south_row_matches_700_of_765_trim():
    lat, _lon = geo.grid_latlon(bias=_ZERO_BIAS)
    _h, w = geo.GRID

    xmin, xmax, ymin, ymax = _corner_xy()
    y_south = ymax - (700.0 / 765.0) * (ymax - ymin)

    # The whole south row, not just its extremum: the stereographic grid's
    # south row latitude varies ~0.475 deg west->east (the domain isn't
    # symmetric about the projection's central meridian, so its SE corner —
    # not the SW one — carries the grid's global-minimum latitude). Checking
    # only one column would miss a per-column bug in the trim.
    to_ll = pyproj.Transformer.from_crs(geo._PROJ4, "EPSG:4326", always_xy=True)
    cx = np.linspace(xmin, xmax, w)
    expected_row = np.array([to_ll.transform(x, y_south)[1] for x in cx])

    np.testing.assert_allclose(lat[-1], expected_row, atol=0.01)


def test_analysis_grid_dst_aux_trim_matches_radar_trim():
    """analysis_grid_dst() (used to reproject aux sources onto the analysis
    grid) carries its own copy of the 765->700 trim, independent of
    grid_latlon()'s. Pin the two to agree: the aux grid's south pixel EDGE
    (from its affine transform) must sit exactly half a cell south of the
    radar grid's south row CENTRE (from grid_latlon()) — both trims apply the
    same 700/765 ratio to the same corner extent. If analysis_grid_dst's copy
    were ever dropped (falling back to the untrimmed native ymin), this
    would be off by ~50 km, not half a cell.
    """
    _xmin, _xmax, ymin, ymax = _corner_xy()
    y_south_radar_centre = ymax - (700.0 / 765.0) * (ymax - ymin)

    _, transform, (h, _w) = geo.analysis_grid_dst()
    py = abs(transform.e)
    y_south_aux_edge = transform.f + h * transform.e

    assert abs((y_south_radar_centre - y_south_aux_edge) - py / 2) < 0.05 * py


def test_grid_latlon_delegates_bit_identical_to_grid_module():
    """geo.grid_latlon() is a thin wrapper around
    Grid.legacy_knmi_analysis(GRID).latlon() — the two must never drift."""
    from model.grid import Grid

    geo_lat, geo_lon = geo.grid_latlon(bias=_ZERO_BIAS)
    grid_lat, grid_lon = Grid.legacy_knmi_analysis(geo.GRID, bias=_ZERO_BIAS).latlon()
    np.testing.assert_array_equal(geo_lat, grid_lat)
    np.testing.assert_array_equal(geo_lon, grid_lon)


def test_grid_latlon_env_bias_change_takes_effect_without_cache_clear(monkeypatch):
    """1.11: grid_latlon()'s memoisation is keyed on the *resolved* bias, not
    read from the environment inside the cached call — so a later
    PLUVIO_GRID_LATLON_BIAS change must be picked up on the very next call
    with no explicit cache_clear() needed."""
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "1.0,2.0")
    lat_a, lon_a = geo.grid_latlon()
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "3.0,4.0")
    lat_b, lon_b = geo.grid_latlon()
    np.testing.assert_allclose(lat_b - lat_a, 2.0, atol=1e-4)
    np.testing.assert_allclose(lon_b - lon_a, 2.0, atol=1e-4)


def test_resample_last_row_averages_source_rows_693_to_699():
    th, tw = ANALYSIS_GRID
    native_h, native_w = 765, 700
    yh = native_h // th
    xw = native_w // tw
    assert th * yh == 700  # the trim: 100 * 7 == 700, not the native 765
    assert tw * xw == 700  # columns aren't trimmed (700 // 100 == 7 exactly)

    field = np.tile(np.arange(native_h, dtype="float64")[:, None], (1, native_w))
    out = _resample(field, ANALYSIS_GRID)
    assert out.shape == ANALYSIS_GRID

    last_block_rows = np.arange((th - 1) * yh, th * yh)
    assert np.allclose(out[-1], last_block_rows.mean())
