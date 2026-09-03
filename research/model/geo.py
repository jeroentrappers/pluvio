"""Geometry of the analysis grid.

The radar product is a polar-stereographic grid. To place auxiliary sources
(AWS point stations, Meteosat / ALARO rasters) onto the same 100x100 analysis
grid the model uses, we need each grid cell's lat/lon.

The native grid is regular in *projected* (km) space, so we:
  1. project the 4 corner lat/lons to stereographic x/y,
  2. lay a 100x100 regular grid across that x/y bounding box (cell centres),
  3. inverse-project back to lat/lon.

This is exact for a regular projected grid (our 100x100 is the block-mean of
the native 765x700, which shares the same projected extent).

The geometry itself (proj4, corners, the 700/765 trim, the empirical
registration bias) lives in `model.grid` as the single source — this module
is now a thin, cached wrapper around `Grid.legacy_knmi_analysis()` so there
is exactly one place that can get the trim or the bias wrong.
"""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import sys

import numpy as np

from .grid import (
    Grid,
    _LEGACY_CORNERS_LONLAT,
    _LEGACY_PROJ4,
    _legacy_bias,
    _legacy_trimmed_extent,
    log_resolved_geometry as _grid_log_resolved_geometry,
)

LOG = logging.getLogger("pluvio.geo")

# The analysis grid default is 100x100 over the ~707x773 km KNMI radar domain —
# i.e. an effective resolution of ~7-8 km/cell (see grid_resolution_km()), NOT
# the "2 km" some design notes claimed. That is **coarser than a convective
# cell** (1-5 km), which directly bounds the heavy-rain/convective claim
# (docs/seamless_model_plan.md §1). To train/serve at a finer resolution, set
#     PLUVIO_GRID_N=256     # → ~2.8 km/cell
# and rebuild the zarr; everything reprojects from the same projected extent, so
# the only cost is compute + storage (and checkpoints are resolution-specific).
#
# Resolution order of preference: env override → notebooks canonical → literal.
def _default_grid() -> tuple[int, int]:
    n = os.environ.get("PLUVIO_GRID_N")
    if n:
        return (int(n), int(n))
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
        import _lib as kpi  # noqa: E402

        return tuple(kpi.ANALYSIS_GRID)  # (100, 100)
    except Exception:
        return (100, 100)


GRID = _default_grid()

# Kept for any external code that still imports geo._PROJ4 / geo._CORNERS_LONLAT
# directly; model.grid is the source of truth for both.
_PROJ4 = _LEGACY_PROJ4
_CORNERS_LONLAT = _LEGACY_CORNERS_LONLAT


def log_resolved_geometry() -> None:
    """Log the resolved analysis GRID and (via model.grid) the registration
    bias at INFO. A module-import-time log call would run before a CLI's own
    logging.basicConfig() and be swallowed (no handlers configured yet) —
    call this explicitly from a CLI's main(), after logging is configured,
    instead."""
    LOG.info("model.geo: resolved analysis GRID=%s (PLUVIO_GRID_N=%s)",
             GRID, os.environ.get("PLUVIO_GRID_N"))
    _grid_log_resolved_geometry()


@functools.lru_cache(maxsize=8)
def _grid_latlon_cached(shape: tuple[int, int],
                         bias: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    return Grid.legacy_knmi_analysis(shape, bias=bias).latlon()


def grid_latlon(bias: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays of shape GRID for the analysis grid.

    Row 0 is the north edge, row H-1 the south edge (matching how the radar
    field is stored, DISPLAY_ORIGIN=UL). Delegates to
    `Grid.legacy_knmi_analysis(GRID, bias=...).latlon()` — see model.grid for
    the trim (northern 700/765 of the corner-derived projected extent).

    `bias` is the empirical registration bias; pass it explicitly to bypass
    the environment entirely. Left as None (the default), it falls back to
    the current `PLUVIO_GRID_LATLON_BIAS` env value (default (0, 0.07)) —
    read fresh on *every* call, not frozen at import or inside the memoised
    computation below, so a later env change is picked up on the next call
    instead of being silently served from a stale cache entry (1.11: that
    staleness was the mechanism of the 192² training crash).
    """
    resolved_bias = bias if bias is not None else _legacy_bias()
    return _grid_latlon_cached(GRID, resolved_bias)


# Keep the old grid_latlon.cache_clear() surface (notebooks call it) — it now
# clears the underlying memoised-by-resolved-inputs cache rather than a cache
# on grid_latlon() itself, which no longer carries one.
grid_latlon.cache_clear = _grid_latlon_cached.cache_clear


def bbox(bias: tuple[float, float] | None = None) -> tuple[float, float, float, float]:
    """(west, south, east, north) lon/lat CORNER ENVELOPE of the grid.

    This is `envelope()` — see its docstring. The legacy analysis grid is NOT
    a lat/lon rectangle: its rows/columns curve in the stereographic
    projection (the south row's latitude varies ~0.475 deg / ~53 km, the
    east column's longitude ~1.71 deg / ~115 km — see
    `research/docs/geometry_audit.md`), so this box over-claims the domain
    near every edge except the single row/column touching its extremum.

    `bbox()` is kept (rather than renamed) because most existing callers use
    it correctly: as a SUPERSET region for fetching/reprojecting external
    rasters (fetch_eumetsat_msg, fetch_alaro_24h, build_dem, build_aux_msg),
    where over-claiming is harmless because every true grid point is still
    guaranteed to be inside it. Callers that instead treat this box as if it
    WERE the grid — building an independent regular lat/lon raster over it
    and then comparing/binning that cell-for-cell against the true curved
    grid (e.g. `tools.radar_single_site.polar_to_grid` and everything that
    calls it with `bounds=bbox()`) are misaligned by the same curvature
    error; see the audit doc rather than assuming this box is a rectangle.
    Use `inner_rectangle()` where a box must be a guaranteed SUBSET of the
    true domain instead.

    `bias` is forwarded to `grid_latlon()` — see there; None (the default)
    means "read `PLUVIO_GRID_LATLON_BIAS` fresh on this call"."""
    return envelope(bias)


def envelope(bias: tuple[float, float] | None = None) -> tuple[float, float, float, float]:
    """(west, south, east, north) lon/lat corner envelope — see `bbox()`.

    `bias` is forwarded to `grid_latlon()`; pass it explicitly to bypass the
    environment entirely (1.11), exactly as `grid_latlon(bias=...)` does."""
    lat, lon = grid_latlon(bias)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def inner_rectangle(bias: tuple[float, float] | None = None) -> tuple[float, float, float, float]:
    """(west, south, east, north) largest lon/lat rectangle guaranteed to be
    covered by every row and every column of the grid — see
    `Grid.inner_rectangle()`. Use this, not `bbox()`/`envelope()`, wherever a
    lat/lon box must actually be a SUBSET of the true (curved) domain.

    `bias` is forwarded to `grid_latlon()` — see `envelope()`."""
    lat, lon = grid_latlon(bias)
    west = float(lon.min(axis=1).max())
    east = float(lon.max(axis=1).min())
    south = float(lat.min(axis=0).max())
    north = float(lat.max(axis=0).min())
    return west, south, east, north


@functools.lru_cache(maxsize=1)
def analysis_grid_dst():
    """Destination for reprojecting any source onto the analysis grid: returns
    (proj4, affine_transform, (h, w)) in the KNMI stereographic CRS. Use with
    rasterio.warp.reproject — handles arbitrary source CRS (OPERA LAEA, AIFS
    lat/lon, MTG EPSG:4326) onto our regular projected grid.

    The trimmed extent comes from `model.grid._legacy_trimmed_extent()` — the
    same trim `grid_latlon()` uses, so aux stays aligned with radar/truth.
    """
    from rasterio.transform import from_origin

    h, w = GRID
    xmin, ymin, xmax, ymax = _legacy_trimmed_extent()
    px = (xmax - xmin) / (w - 1)
    py = (ymax - ymin) / (h - 1)
    transform = from_origin(xmin - px / 2, ymax + py / 2, px, py)  # north-up
    return _PROJ4, transform, (h, w)


@functools.lru_cache(maxsize=1)
def grid_resolution_km() -> tuple[float, float]:
    """Effective cell size (dy_km, dx_km) of the current analysis grid.

    The stereographic extent is fixed by the radar domain corners, so the
    resolution is purely a function of ``GRID``. Use this to gate convective
    claims honestly: at the 100x100 default this returns ~(7.7, 7.1) km — much
    coarser than a convective cell, so heavy-rain structure below the cell scale
    is averaged away and should not be claimed at face value.
    """
    _, transform, (h, w) = analysis_grid_dst()
    return (abs(transform.e) / 1000.0, abs(transform.a) / 1000.0)
