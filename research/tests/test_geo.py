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

import pathlib

import numpy as np
import pyproj
import pytest

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


def test_grid_latlon_cache_clear_surface_kept_for_notebooks():
    """geo.grid_latlon used to be an lru_cache-decorated function itself;
    notebooks call geo.grid_latlon.cache_clear() directly, so that surface
    must survive the 1.11 refactor even though the memoisation moved to an
    internal helper."""
    geo.grid_latlon(bias=(1.0, 1.0))
    geo.grid_latlon.cache_clear()  # must not raise


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


# ---------------------------------------------------------------------------
# 1.10: geo.bbox()/envelope()/inner_rectangle() — see
# research/docs/geometry_audit.md and test_grid.py's Grid-level equivalents.
# ---------------------------------------------------------------------------

def test_bbox_is_envelope():
    assert geo.bbox() == geo.envelope()


def test_envelope_contains_every_grid_point():
    lat, lon = geo.grid_latlon()
    w, s, e, n = geo.envelope()
    assert lon.min() >= w and lon.max() <= e
    assert lat.min() >= s and lat.max() <= n


def test_inner_rectangle_contained_by_every_row_and_column():
    lat, lon = geo.grid_latlon()
    w, s, e, n = geo.inner_rectangle()
    assert (lon.min(axis=1) <= w + 1e-6).all()
    assert (lon.max(axis=1) >= e - 1e-6).all()
    assert (lat.min(axis=0) <= s + 1e-6).all()
    assert (lat.max(axis=0) >= n - 1e-6).all()


def test_inner_rectangle_strictly_smaller_than_envelope():
    ew, es, ee, en = geo.envelope()
    iw, is_, ie, in_ = geo.inner_rectangle()
    assert (ie - iw) < (ee - ew)
    assert (in_ - is_) < (en - es)


def test_envelope_and_inner_rectangle_delegate_to_grid_module():
    """geo.envelope()/inner_rectangle() are thin wrappers around
    geo.grid_latlon() — call the wrappers themselves and pin them to the
    Grid-level equivalents built from the same zero-bias geometry, so the two
    implementations can never drift apart."""
    from model.grid import Grid

    g = Grid.legacy_knmi_analysis(geo.GRID, bias=_ZERO_BIAS)
    assert geo.envelope(bias=_ZERO_BIAS) == pytest.approx(g.envelope())
    assert geo.inner_rectangle(bias=_ZERO_BIAS) == pytest.approx(g.inner_rectangle())


def test_envelope_bias_argument_bypasses_the_environment(monkeypatch):
    """1.11: envelope()/inner_rectangle()/bbox() take `bias` like
    grid_latlon() does, so a caller can pin the geometry without touching (or
    being affected by) PLUVIO_GRID_LATLON_BIAS."""
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "5.0,6.0")
    assert geo.envelope(bias=_ZERO_BIAS) == pytest.approx(geo.bbox(bias=_ZERO_BIAS))
    # ...and the env value really would have moved it, so the pin is doing work.
    shift = np.array([6.0, 5.0, 6.0, 5.0])  # (lon, lat, lon, lat) of (W,S,E,N)
    np.testing.assert_allclose(
        np.array(geo.envelope()) - np.array(geo.envelope(bias=_ZERO_BIAS)),
        shift, atol=1e-4)
    np.testing.assert_allclose(
        np.array(geo.inner_rectangle()) - np.array(geo.inner_rectangle(bias=_ZERO_BIAS)),
        shift, atol=1e-4)


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


# ---------------------------------------------------------------------------
# 1.10 / #24: geo.GRID is resolved ONCE, at import time, from PLUVIO_GRID_N —
# so any CLI with a --grid-n flag must set the env BEFORE its first
# `import model.geo`, or the flag is a silent no-op. radar_single_site.py had
# the two lines the wrong way round.
# ---------------------------------------------------------------------------

def _first_line(src: str, needle: str) -> int:
    for i, line in enumerate(src.splitlines(), start=1):
        if needle in line and not line.lstrip().startswith("#"):
            return i
    raise AssertionError(f"{needle!r} not found")


@pytest.mark.parametrize("tool", ["radar_single_site.py", "verify_radar.py"])
def test_grid_n_env_is_set_before_model_geo_is_imported(tool):
    src = (pathlib.Path(__file__).resolve().parents[1] / "tools" / tool).read_text()
    setenv = _first_line(src, 'os.environ.setdefault("PLUVIO_GRID_N"')
    import_geo = _first_line(src, "from model.geo import")
    assert setenv < import_geo, (
        f"{tool}: PLUVIO_GRID_N is set at line {setenv}, after model.geo is "
        f"imported at line {import_geo} — geo.GRID is already resolved by "
        "then, so --grid-n would be silently ignored"
    )
