from __future__ import annotations

import json

import numpy as np
import pytest
import zarr
from model import geo
from model.grid import Grid, GridContractError


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
    monkeypatch.setenv("PLUVIO_GRID_LATLON_BIAS", "0.05,-0.03")
    geo.grid_latlon.cache_clear()
    try:
        geo_lat, geo_lon = geo.grid_latlon()
        g = Grid.legacy_knmi_analysis(geo.GRID)
        assert g.latlon_bias == (0.05, -0.03)
        lat, lon = g.latlon()
        np.testing.assert_allclose(lat, geo_lat, atol=1e-4)
        np.testing.assert_allclose(lon, geo_lon, atol=1e-4)
    finally:
        geo.grid_latlon.cache_clear()


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
