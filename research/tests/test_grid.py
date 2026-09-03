from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from model import geo
from model.grid import Grid, GridContractError, centre_to_edge_bounds


def test_regular_grid_to_from_attrs_roundtrip():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    attrs = g.to_attrs()
    assert attrs["grid_crs"] == "EPSG:4326"
    assert attrs["grid_bounds"] == [1.5, 48.9, 7.5, 54.2]
    assert attrs["grid_shape"] == [192, 192]
    assert attrs["grid_row_order"] == "north_first"
    assert "grid_version" in attrs

    g2 = Grid.from_attrs(attrs)
    assert g2 == g


def test_legacy_grid_to_from_attrs_roundtrip():
    g = Grid.legacy_knmi_analysis((100, 100))
    attrs = g.to_attrs()
    assert "grid_proj_extent" in attrs
    assert "grid_trim_note" in attrs
    g2 = Grid.from_attrs(attrs)
    assert g2 == g


def test_from_attrs_missing_keys_raises():
    with pytest.raises(GridContractError):
        Grid.from_attrs({"grid_crs": "EPSG:4326"})


def test_from_attrs_empty_raises_helpful_message():
    with pytest.raises(GridContractError, match="missing Grid keys"):
        Grid.from_attrs({})


def test_bounds_must_be_ordered():
    with pytest.raises(GridContractError):
        Grid.regular(bounds=(7.5, 48.9, 1.5, 54.2), shape=(10, 10))


def test_row_order_only_north_first():
    with pytest.raises(GridContractError):
        Grid(crs="EPSG:4326", bounds=(0, 0, 1, 1), shape=(10, 10), row_order="south_first")


def test_from_zarr_rejects_shape_mismatch(tmp_path):
    store_path = tmp_path / "mismatch.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(50, 50))
    root.attrs.update(g.to_attrs())
    # radar array shape disagrees with the declared grid shape (192 != 50)
    root.create_array("radar", shape=(4, 1, 192, 192), dtype="float32")

    with pytest.raises(GridContractError, match="does not match"):
        Grid.from_zarr(root)


def test_from_zarr_missing_attrs_raises(tmp_path):
    store_path = tmp_path / "legacy_v2.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    root.create_array("radar", shape=(4, 1, 100, 100), dtype="float32")

    with pytest.raises(GridContractError):
        Grid.from_zarr(root)


def test_from_zarr_reads_regular_grid_identically(tmp_path):
    store_path = tmp_path / "regular_v3.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=2)
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    root.attrs.update(g.to_attrs())
    root.create_array("radar", shape=(4, 1, 192, 192), dtype="float32")

    g2 = Grid.from_zarr(root)
    assert g2 == g


def test_legacy_latlon_matches_geo_module():
    g = Grid.legacy_knmi_analysis(shape=geo.GRID)
    lat, lon = g.latlon()
    geo_lat, geo_lon = geo.grid_latlon()
    assert lat.shape == geo_lat.shape
    np.testing.assert_allclose(lat, geo_lat, atol=1e-4)
    np.testing.assert_allclose(lon, geo_lon, atol=1e-4)


def test_regular_grid_latlon_orientation():
    g = Grid.regular(bounds=(0.0, 0.0, 10.0, 20.0), shape=(3, 2))
    lat, lon = g.latlon()
    # row 0 = north
    assert lat[0, 0] == pytest.approx(20.0)
    assert lat[-1, 0] == pytest.approx(0.0)
    assert lon[0, 0] == pytest.approx(0.0)
    assert lon[0, -1] == pytest.approx(10.0)


@pytest.mark.parametrize("row,col", [(0, 0), (5, 7), (99, 99), (42, 13)])
def test_cell_of_and_bounds_of_cell_invert(row, col):
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(100, 100))
    lat, lon = g.latlon()
    center_lat, center_lon = float(lat[row, col]), float(lon[row, col])

    got = g.cell_of(center_lat, center_lon)
    assert got == (row, col)

    w, s, e, n = g.bounds_of_cell(row, col)
    assert w <= center_lon <= e
    assert s <= center_lat <= n


def test_cell_of_outside_bounds_returns_none():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(100, 100))
    assert g.cell_of(lat=0.0, lon=0.0) is None
    assert g.cell_of(lat=90.0, lon=100.0) is None


def test_edge_bounds_spans_exact_cell_footprint():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    h, wid = g.shape
    w, s, e, n = g.bounds
    dlon = (e - w) / (wid - 1)
    dlat = (n - s) / (h - 1)

    ew, es, ee, en = g.edge_bounds()
    assert (ee - ew) == pytest.approx(wid * dlon)
    assert (en - es) == pytest.approx(h * dlat)


def test_transform_matches_edge_bounds_origin():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    ew, _es, _ee, en = g.edge_bounds()
    x0, y0, dx, dy = g.transform()
    assert x0 == pytest.approx(ew)
    assert y0 == pytest.approx(en)
    assert dx > 0 and dy > 0


def test_cell_of_accepts_point_inside_boundary_cells_own_footprint():
    # bounds_of_cell(191, 0)'s south edge sits half a cell below the centre
    # envelope's south bound (48.9) — a point there is still inside cell
    # (191, 0)'s own footprint and must resolve to it, not None.
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    _w, s, _e, _n = g.bounds_of_cell(191, 0)
    assert s < 48.9  # south edge is below the centre-envelope bound
    assert g.cell_of(lat=48.895, lon=1.5) == (191, 0)


def test_bounds_of_cell_tiles_exactly():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192))
    _h, wid = g.shape
    w, _s, e, _n = g.bounds
    dlon = (e - w) / (wid - 1)
    for r, c in [(0, 0), (10, 10), (191, 190)]:
        _, _, east0, _ = g.bounds_of_cell(r, c)
        west1, _, _, _ = g.bounds_of_cell(r, c + 1)
        assert east0 == pytest.approx(west1)
        assert (east0 - g.bounds_of_cell(r, c)[0]) == pytest.approx(dlon)


def test_bias_override_recomputes_agreement(monkeypatch):
    # geo.grid_latlon()'s memoisation is keyed on the resolved bias (1.11),
    # so a fresh env value is picked up on the very next call — no
    # cache_clear() needed.
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "0.05,-0.03")
    geo_lat, geo_lon = geo.grid_latlon()
    g = Grid.legacy_knmi_analysis(geo.GRID)
    assert g.latlon_bias == (0.05, -0.03)
    lat, lon = g.latlon()
    np.testing.assert_allclose(lat, geo_lat, atol=1e-4)
    np.testing.assert_allclose(lon, geo_lon, atol=1e-4)


def test_legacy_grid_records_active_bias_for_reproducibility(monkeypatch):
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "0.11,0.22")
    g = Grid.legacy_knmi_analysis((10, 10))
    attrs = g.to_attrs()
    assert attrs["grid_latlon_bias"] == [0.11, 0.22]

    # a serialised Grid reproduces its own latlon() regardless of a later,
    # different environment bias.
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "-9,-9")
    g2 = Grid.from_attrs(attrs)
    lat1, lon1 = g.latlon()
    lat2, lon2 = g2.latlon()
    np.testing.assert_array_equal(lat1, lat2)
    np.testing.assert_array_equal(lon1, lon2)


def test_to_attrs_emits_no_numpy_scalars():
    for g in (Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(192, 192)),
              Grid.legacy_knmi_analysis((100, 100))):
        attrs = g.to_attrs()
        for key, value in attrs.items():
            values = value if isinstance(value, list) else [value]
            for v in values:
                assert type(v) in (int, float, str), (key, v, type(v))
        json.dumps(attrs)  # must be plain-JSON-serialisable


def test_from_attrs_rejects_newer_grid_version():
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(10, 10))
    attrs = g.to_attrs()
    attrs["grid_version"] = 999
    with pytest.raises(GridContractError, match="newer than"):
        Grid.from_attrs(attrs)


# ---------------------------------------------------------------------------
# 1.10: envelope() vs inner_rectangle() — geo.bbox() over-claims the curved
# stereographic domain. See research/docs/geometry_audit.md.
# ---------------------------------------------------------------------------

def test_envelope_contains_every_grid_point_legacy():
    g = Grid.legacy_knmi_analysis((100, 100), bias=(0.0, 0.0))
    lat, lon = g.latlon()
    w, s, e, n = g.envelope()
    assert lon.min() >= w and lon.max() <= e
    assert lat.min() >= s and lat.max() <= n
    # tight: the envelope is exactly the corner box of the actual points.
    assert w == pytest.approx(float(lon.min()))
    assert e == pytest.approx(float(lon.max()))
    assert s == pytest.approx(float(lat.min()))
    assert n == pytest.approx(float(lat.max()))


def test_envelope_matches_regular_grid_bounds():
    """For a regular (EPSG:4326) grid, envelope() == inner_rectangle() ==
    bounds — the curvature gap only exists for a projected grid."""
    g = Grid.regular(bounds=(1.5, 48.9, 7.5, 54.2), shape=(50, 40))
    assert g.envelope() == pytest.approx(g.bounds)
    assert g.inner_rectangle() == pytest.approx(g.bounds)


def test_inner_rectangle_contained_by_every_row_and_column_legacy():
    g = Grid.legacy_knmi_analysis((100, 100), bias=(0.0, 0.0))
    lat, lon = g.latlon()
    w, s, e, n = g.inner_rectangle()
    # every row's own lon range must contain [w, e]
    assert (lon.min(axis=1) <= w + 1e-6).all()
    assert (lon.max(axis=1) >= e - 1e-6).all()
    # every column's own lat range must contain [s, n]
    assert (lat.min(axis=0) <= s + 1e-6).all()
    assert (lat.max(axis=0) >= n - 1e-6).all()


def test_inner_rectangle_strictly_smaller_than_envelope_for_legacy_grid():
    """The whole point of 1.10: the legacy grid curves, so the guaranteed
    subset is strictly smaller than the corner envelope on three of its four
    edges. The west edge is the exception BY CONSTRUCTION, not by luck:
    `_LEGACY_PROJ4` has lon_0=0 and both west corners of
    `_LEGACY_CORNERS_LONLAT` sit at lon 0.0, so the whole west column lies on
    projected x=0 and shares one exact longitude (see
    test_documented_km_displacement_at_domain_edges)."""
    g = Grid.legacy_knmi_analysis((100, 100), bias=(0.0, 0.0))
    ew, es, ee, en = g.envelope()
    iw, is_, ie, in_ = g.inner_rectangle()
    assert iw >= ew
    assert ie <= ee
    assert is_ >= es
    assert in_ <= en
    assert (ie - iw) < (ee - ew)
    assert (in_ - is_) < (en - es)


def test_documented_km_displacement_at_domain_edges():
    """The ~53/65/115 km numbers quoted in research/docs/geometry_audit.md,
    asserted within tolerance so the doc can't silently drift from the
    geometry it describes."""
    g = Grid.legacy_knmi_analysis((100, 100), bias=(0.0, 0.0))
    lat, lon = g.latlon()
    row_min, row_max = lon.min(axis=1), lon.max(axis=1)
    col_min, col_max = lat.min(axis=0), lat.max(axis=0)
    lat0 = float(lat.mean())
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat0))

    north_gap_km = (float(lat.max()) - float(col_max.min())) * km_per_deg_lat
    south_gap_km = (float(col_min.max()) - float(lat.min())) * km_per_deg_lat
    east_gap_km = (float(lon.max()) - float(row_max.min())) * km_per_deg_lon
    west_gap_km = (float(row_min.max()) - float(lon.min())) * km_per_deg_lon

    assert north_gap_km == pytest.approx(64.9, abs=3.0)
    assert south_gap_km == pytest.approx(52.8, abs=3.0)
    assert east_gap_km == pytest.approx(115.2, abs=5.0)
    # Exactly zero, not "approximately": lon_0=0 and both west corners at lon
    # 0.0 put the entire west column on projected x=0, so every row's own
    # west-most longitude IS lon.min(). No tolerance needed or wanted here —
    # any nonzero value means the projection or the corners changed.
    assert west_gap_km == 0.0
    assert float(row_min.max()) == float(lon.min())
    assert (lon[:, 0] == lon[0, 0]).all()


def test_documented_naive_regular_grid_displacement():
    """The domain-wide naive-vs-true numbers quoted in
    research/docs/geometry_audit.md ("The bug"): the great-circle distance
    between each cell centre of an independent REGULAR lat/lon raster over the
    envelope (what `radar_single_site.polar_to_grid` bins onto:
    col = (lon - w)/(e - w) * wd, i.e. cell centres at half-pixel offsets) and
    the TRUE curved cell centre at the same (row, col). Pinned so the doc's
    "4-9 cells everywhere, not an edge effect" claim can't silently drift."""
    from pyproj import Geod

    g = Grid.legacy_knmi_analysis((100, 100), bias=(0.0, 0.0))
    lat, lon = g.latlon()
    w, s, e, n = g.envelope()
    h, wd = lat.shape
    naive_lon = w + (np.arange(wd) + 0.5) * (e - w) / wd
    naive_lat = n - (np.arange(h) + 0.5) * (n - s) / h
    nlon = np.broadcast_to(naive_lon[None, :], lat.shape)
    nlat = np.broadcast_to(naive_lat[:, None], lat.shape)
    _, _, d_m = Geod(ellps="WGS84").inv(nlon, nlat, lon, lat)
    km = d_m / 1000.0

    assert float(km.min()) == pytest.approx(0.1, abs=0.5)
    assert float(np.median(km)) == pytest.approx(37.1, abs=1.5)
    assert float(km.mean()) == pytest.approx(39.4, abs=1.5)
    assert float(km.max()) == pytest.approx(120.4, abs=3.0)
    assert km[0, 0] == pytest.approx(5.0, abs=1.0)     # NW
    assert km[0, -1] == pytest.approx(61.5, abs=2.0)   # NE
    assert km[-1, 0] == pytest.approx(49.4, abs=2.0)   # SW
    assert km[-1, -1] == pytest.approx(120.4, abs=3.0)  # SE (= domain max)
    assert km[50, 50] == pytest.approx(30.7, abs=1.5)  # domain centre

    # The point of the doc paragraph: even the CENTRE is several cells out, so
    # this is not a boundary effect that a margin could be trimmed away.
    dy_km, dx_km = 7.7, 7.1  # geo.grid_resolution_km() at 100x100
    cell_km = (dy_km + dx_km) / 2
    assert km[50, 50] / cell_km > 3.5
    assert float(np.median(km)) / cell_km > 4.0


def test_centre_to_edge_bounds_matches_grid_edge_bounds():
    """The module-level helper is the conversion `Grid.edge_bounds()` applies,
    for callers holding a loose (bounds, shape) pair — an npz's `bounds` on
    its way into a regrid or a lat/lon -> cell lookup."""
    for bounds, shape in (((1.5, 48.9, 7.5, 54.2), (192, 192)),
                          ((1.5, 48.9, 7.5, 52.5), (100, 100)),
                          ((0.0, 45.0, 10.0, 55.0), (16, 32))):
        g = Grid.regular(bounds=bounds, shape=shape)
        assert centre_to_edge_bounds(bounds, shape) == pytest.approx(g.edge_bounds())
        ew, es, ee, en = centre_to_edge_bounds(bounds, shape)
        w, s, e, n = bounds
        h, wid = shape
        assert (ee - ew) == pytest.approx(wid * (e - w) / (wid - 1))
        assert (en - es) == pytest.approx(h * (n - s) / (h - 1))


def test_centre_to_edge_bounds_degenerate_axis_is_the_identity():
    """A shape of 1 along an axis has no derivable cell size, so that axis's
    edges equal its centres — no division by zero (documented convention)."""
    bounds = (1.5, 48.9, 7.5, 52.5)
    assert centre_to_edge_bounds(bounds, (1, 1)) == pytest.approx(bounds)
    w, s, e, n = bounds
    dlat = (n - s) / 9
    assert centre_to_edge_bounds(bounds, (10, 1)) == pytest.approx(
        (w, s - dlat / 2, e, n + dlat / 2))
    dlon = (e - w) / 9
    assert centre_to_edge_bounds(bounds, (1, 10)) == pytest.approx(
        (w - dlon / 2, s, e + dlon / 2, n))


def test_centre_to_edge_bounds_rejects_malformed_input():
    with pytest.raises(GridContractError):
        centre_to_edge_bounds((1.5, 48.9, 7.5), (10, 10))
    with pytest.raises(GridContractError):
        centre_to_edge_bounds((1.5, 48.9, 7.5, 52.5), (10,))
    with pytest.raises(GridContractError):
        centre_to_edge_bounds((1.5, 48.9, 7.5, 52.5), (0, 10))
