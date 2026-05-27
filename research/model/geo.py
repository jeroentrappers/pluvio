"""Geometry of the analysis grid.

The radar product is a polar-stereographic grid (KNMI proj4 below). To place
auxiliary sources (AWS point stations, Meteosat / ALARO rasters) onto the
same 100×100 analysis grid the model uses, we need each grid cell's lat/lon.

The native grid is regular in *projected* (km) space, so we:
  1. project the 4 corner lat/lons to stereographic x/y,
  2. lay a 100×100 regular grid across that x/y bounding box (cell centres),
  3. inverse-project back to lat/lon.

This is exact for a regular projected grid (our 100×100 is the block-mean of
the native 765×700, which shares the same projected extent).
"""

from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np
import pyproj

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import _lib as kpi  # noqa: E402

GRID = kpi.ANALYSIS_GRID  # (100, 100)

# KNMI radar stereographic projection (from the HDF5 map_projection group).
# The HDF5 gives the ellipsoid radii in km (a=6378.14, b=6356.75); pyproj
# rejects those as a non-Earth body, so we express the same ellipsoid in
# metres. Forward/inverse stay self-consistent, which is all we need to map
# the regular projected grid back to lat/lon.
_PROJ4 = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +x_0=0 +y_0=0"

# Corner lon/lat pairs from `geographic/geo_product_corners`
# order: LL, UL, UR, LR (each lon, lat).
_CORNERS_LONLAT = [
    (0.0, 49.362064),
    (0.0, 55.973602),
    (10.856453, 55.388973),
    (9.0093, 48.8953),
]


@functools.lru_cache(maxsize=1)
def grid_latlon() -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays of shape GRID for the analysis grid.

    Row 0 is the north edge, row H-1 the south edge (matching how the radar
    field is stored, DISPLAY_ORIGIN=UL).
    """
    h, w = GRID
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", _PROJ4, always_xy=True)
    to_ll = pyproj.Transformer.from_crs(_PROJ4, "EPSG:4326", always_xy=True)

    xs, ys = [], []
    for lon, lat in _CORNERS_LONLAT:
        x, y = to_xy.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Cell centres: linspace across the projected extent. Row 0 = north (ymax).
    cx = np.linspace(xmin, xmax, w)
    cy = np.linspace(ymax, ymin, h)
    gx, gy = np.meshgrid(cx, cy)  # (h, w)
    lon, lat = to_ll.transform(gx, gy)
    return lat.astype("float32"), lon.astype("float32")


def bbox() -> tuple[float, float, float, float]:
    """(west, south, east, north) lon/lat envelope of the grid, for WMS GetMap."""
    lat, lon = grid_latlon()
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())
