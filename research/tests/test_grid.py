from __future__ import annotations

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
