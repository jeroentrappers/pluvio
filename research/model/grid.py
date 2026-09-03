"""The `Grid` data contract: every array on disk carries its own georeference,
and every consumer reads it — never assumes it.

This week's ~50 km projection error and ~0.5 deg aux misregistration both came
from geometry being reconstructed by hand in three different files (the
KNMI-stereographic analysis grid, whose rows only cover the northern 700/765
of its corner extent — see geo.py; a regular lat/lon Benelux box; and a wide
1.5-km composite grid), with no single source of truth and no check that a
store's data actually matched the geometry a consumer assumed for it.

`Grid` fixes that: a small, frozen, serialisable description of a raster's
georeference (CRS, bounds, shape, row order), written into zarr attrs by
every store builder and read — with a cross-check against on-disk array
shape — by every consumer. `geo.py` now delegates its legacy-grid geometry
to this module instead of keeping a second copy of the corners/trim/bias.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyproj

LOG = logging.getLogger("pluvio.grid")

GRID_VERSION = 1

# The only row order this contract supports: row 0 is the north edge, the
# last row is the south edge. This matches every producer/consumer today
# (radar DISPLAY_ORIGIN=UL, build_store_v3, infer_latest's backend grid).
_SUPPORTED_ROW_ORDERS = ("north_first",)

# KNMI radar stereographic projection (from the HDF5 map_projection group).
# The HDF5 gives the ellipsoid radii in km (a=6378.14, b=6356.75); pyproj
# rejects those as a non-Earth body, so we express the same ellipsoid in
# metres. Forward/inverse stay self-consistent, which is all we need to map
# the regular projected grid back to lat/lon. This is the single source for
# the legacy grid's geometry — geo.py imports it rather than redeclaring it.
_LEGACY_PROJ4 = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +x_0=0 +y_0=0"

# Corner lon/lat pairs from `geographic/geo_product_corners`, order LL, UL, UR, LR.
_LEGACY_CORNERS_LONLAT = [
    (0.0, 49.362064),
    (0.0, 55.973602),
    (10.856453, 55.388973),
    (9.0093, 48.8953),
]

# The analysis fields only ever cover the northern 700 of the native 765-row
# stereographic domain (see geo.py / commit dfc8e5e). Encode that trim here,
# explicitly, so it is never rediscovered by hand again.
_LEGACY_TRIM_ROWS = 700
_LEGACY_NATIVE_ROWS = 765
_LEGACY_TRIM_NOTE = (
    f"analysis rows cover the northern {_LEGACY_TRIM_ROWS}/{_LEGACY_NATIVE_ROWS} "
    "of the corner-derived projected extent (native RAC domain), not the full "
    "corner box — see geo.py grid_latlon()."
)


# Residual empirical registration calibration applied on top of the pure
# corner/trim geometry (commit df83fbf). `Grid.legacy_knmi_analysis()` reads
# this once and records the value it used on the returned Grid (`latlon_bias`)
# so a serialised legacy Grid reproduces its own lat/lon regardless of what
# the environment says later — the env var only controls freshly-built grids.
def _legacy_bias() -> tuple[float, float]:
    try:
        dlat_s, dlon_s = os.environ.get("PLUVIO_GRID_LATLON_BIAS", "0,0.07").split(",")
        return float(dlat_s), float(dlon_s)
    except ValueError:
        return 0.0, 0.07


# Resolved once at import purely for visibility (what did this process start
# with) — `Grid.legacy_knmi_analysis()` and `geo.grid_latlon()` still call
# `_legacy_bias()` fresh on every call rather than reading this constant, so a
# later `PLUVIO_GRID_LATLON_BIAS` change (e.g. via monkeypatch in a test, or a
# long-running process) takes effect on the next call instead of being served
# from a stale value (1.11: the mechanism of the 192² incident was exactly a
# cache that *did* freeze an env read like this).
_LATLON_BIAS_AT_IMPORT = _legacy_bias()


def log_resolved_geometry() -> None:
    """Log the resolved PLUVIO_GRID_LATLON_BIAS at INFO. A module-import-time
    log call would run before a CLI's own logging.basicConfig() and be
    swallowed (no handlers configured yet) — call this explicitly from a
    CLI's main(), after logging is configured, instead."""
    LOG.info("model.grid: PLUVIO_GRID_LATLON_BIAS resolved to %s at import", _LATLON_BIAS_AT_IMPORT)


class GridContractError(ValueError):
    """Raised when zarr/attrs geometry is missing or internally inconsistent."""


def _to_float_tuple(x, n: int, name: str) -> tuple[float, ...]:
    try:
        t = tuple(float(v) for v in x)
    except (TypeError, ValueError) as exc:
        raise GridContractError(f"{name} must be {n} numbers, got {x!r}") from exc
    if len(t) != n:
        raise GridContractError(f"{name} must have {n} elements, got {x!r}")
    return t


def _to_int_tuple(x, n: int, name: str) -> tuple[int, ...]:
    try:
        t = tuple(int(v) for v in x)
    except (TypeError, ValueError) as exc:
        raise GridContractError(f"{name} must be {n} ints, got {x!r}") from exc
    if len(t) != n:
        raise GridContractError(f"{name} must have {n} elements, got {x!r}")
    return t


def centre_to_edge_bounds(bounds, shape) -> tuple[float, float, float, float]:
    """(west, south, east, north) footprint EDGES for a CELL-CENTRE bounds
    envelope + shape — `bounds` inflated by half a cell in each direction.

    The repo's `bounds` are almost always the envelope of the CELL CENTRES of
    the first/last row/col (`Grid.bounds`, the forecast/nowcast npz `bounds`
    written by tools/infer_latest.py and tools/produce_forecast.py, the v3
    zarr store attrs). Anything that interprets bounds as pixel EDGES — a
    painter, a regrid that bins by fractional pixel index, a raster
    transform — must inflate them first, or the footprint is half a cell too
    small on every side (0 error at the centre, up to half a cell at the
    edges: ~3.5 km on the 100x100 Belgium serving grid).

    Degenerate axes (a shape of 1 along an axis) have no derivable cell size,
    so the spacing is taken as 0 and that axis's edges equal its centres —
    the caller must supply a real cell size if it needs one.

    Raises GridContractError on a malformed bounds/shape.
    """
    w, s, e, n = _to_float_tuple(bounds, 4, "bounds")
    h, wid = _to_int_tuple(shape, 2, "shape")
    if h <= 0 or wid <= 0:
        raise GridContractError(f"shape must be positive, got {(h, wid)}")
    dlon = (e - w) / (wid - 1) if wid > 1 else 0.0
    dlat = (n - s) / (h - 1) if h > 1 else 0.0
    return (w - dlon / 2, s - dlat / 2, e + dlon / 2, n + dlat / 2)


@dataclass(frozen=True)
class Grid:
    """A raster's georeference: CRS, lon/lat envelope, shape, row order.

    Bounds convention: `bounds` is the envelope of the CELL CENTRES of the
    first/last row/col (build_store_v3 and the legacy grid both linspace
    centres across the box) — the raster's actual footprint extends half a
    cell further in each direction; painters that treat bounds as pixel
    EDGES (backend cache/colormap/model/verify, web RadarMap, flutter
    OverlayImage) must use `edge_bounds()` or `transform()`, not `bounds`,
    or content shifts by half a cell (~1-2 km at today's resolutions).

    Attributes:
        crs: "EPSG:4326" for a regular lat/lon grid, or a proj4 string for a
            projected grid (e.g. the legacy KNMI stereographic analysis grid).
        bounds: (west, south, east, north) lon/lat envelope of the CELL
            CENTRES — see the class docstring above.
            For a projected grid this is the envelope of the reprojected cell
            centres, not the CRS-native box — use `proj_extent` for that.
        shape: (rows, cols).
        row_order: only "north_first" is supported (row 0 = north edge).
        proj_extent: (xmin, ymin, xmax, ymax) in CRS units, for projected
            grids only (None for regular lat/lon grids).
        trim_note: free-text note documenting any non-obvious crop of a wider
            native domain (e.g. the legacy 700/765 trim). None if the grid is
            not trimmed from anything.
        latlon_bias: (dlat, dlon) empirical registration bias applied on top
            of the pure projected geometry, for the legacy grid only. Recorded
            at construction time so a serialised legacy Grid reproduces its
            own `latlon()` regardless of the current environment.
    """

    crs: str
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    row_order: str = "north_first"
    proj_extent: tuple[float, float, float, float] | None = None
    trim_note: str | None = None
    latlon_bias: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.row_order not in _SUPPORTED_ROW_ORDERS:
            raise GridContractError(
                f"unsupported row_order {self.row_order!r}; only "
                f"{_SUPPORTED_ROW_ORDERS!r} is supported"
            )
        bounds = _to_float_tuple(self.bounds, 4, "bounds")
        object.__setattr__(self, "bounds", bounds)
        shape = _to_int_tuple(self.shape, 2, "shape")
        if shape[0] <= 0 or shape[1] <= 0:
            raise GridContractError(f"shape must be positive, got {shape}")
        object.__setattr__(self, "shape", shape)
        if self.proj_extent is not None:
            object.__setattr__(
                self, "proj_extent", _to_float_tuple(self.proj_extent, 4, "proj_extent")
            )
        if self.latlon_bias is not None:
            object.__setattr__(
                self, "latlon_bias", _to_float_tuple(self.latlon_bias, 2, "latlon_bias")
            )
        w, s, e, n = bounds
        if not (w < e and s < n):
            raise GridContractError(f"bounds must be (west<east, south<north), got {bounds}")

    # -- constructors ---------------------------------------------------

    @staticmethod
    def regular(bounds: tuple[float, float, float, float], shape: tuple[int, int]) -> "Grid":
        """A regular lat/lon grid: bounds=(west, south, east, north) envelope
        of the CELL CENTRES, shape=(rows, cols), row 0 = north."""
        return Grid(crs="EPSG:4326", bounds=tuple(bounds), shape=tuple(shape),
                    row_order="north_first")

    @staticmethod
    def legacy_knmi_analysis(
        shape: tuple[int, int] = (100, 100),
        bias: tuple[float, float] | None = None,
    ) -> "Grid":
        """The legacy KNMI-stereographic analysis grid (geo.py), including its
        700/765 north-only trim of the corner-derived projected extent and
        the empirical registration bias (recorded on the Grid so it
        reproduces itself later regardless of the environment). `bias`
        defaults to the current `PLUVIO_GRID_LATLON_BIAS` env value — pass it
        explicitly to bypass the env entirely."""
        proj_extent = _legacy_trimmed_extent()
        if bias is None:
            bias = _legacy_bias()
        lat, lon = _legacy_latlon(shape, proj_extent, bias=bias)
        bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
        return Grid(crs=_LEGACY_PROJ4, bounds=bounds, shape=tuple(shape),
                    row_order="north_first", proj_extent=proj_extent,
                    trim_note=_LEGACY_TRIM_NOTE, latlon_bias=bias)

    # -- (de)serialisation ------------------------------------------------

    def to_attrs(self) -> dict[str, Any]:
        """JSON-serialisable dict, keys prefixed `grid_*`, for zarr .attrs.
        Values are always plain Python int/float/str/list — never numpy
        scalars — so this round-trips through JSON-backed zarr attrs stores."""
        attrs: dict[str, Any] = {
            "grid_version": GRID_VERSION,
            "grid_crs": self.crs,
            "grid_bounds": [float(x) for x in self.bounds],
            "grid_shape": [int(x) for x in self.shape],
            "grid_row_order": self.row_order,
        }
        if self.proj_extent is not None:
            attrs["grid_proj_extent"] = [float(x) for x in self.proj_extent]
        if self.trim_note is not None:
            attrs["grid_trim_note"] = str(self.trim_note)
        if self.latlon_bias is not None:
            attrs["grid_latlon_bias"] = [float(x) for x in self.latlon_bias]
        return attrs

    @staticmethod
    def from_attrs(attrs: dict[str, Any]) -> "Grid":
        """Reconstruct a Grid from a zarr-attrs-style dict. Raises
        GridContractError with a helpful message if required keys are missing,
        the values are internally inconsistent, or `grid_version` is newer
        than this reader supports."""
        required = ("grid_crs", "grid_bounds", "grid_shape", "grid_row_order")
        missing = [k for k in required if k not in attrs]
        if missing:
            raise GridContractError(
                f"store attrs are missing Grid keys {missing!r} — this store "
                "predates the Grid contract (1.1); either rebuild it or fall "
                "back to legacy geometry explicitly, never assume a grid."
            )
        version = attrs.get("grid_version")
        if version is not None:
            try:
                version = int(version)
            except (TypeError, ValueError) as exc:
                raise GridContractError(f"grid_version must be an int, got {version!r}") from exc
            if version > GRID_VERSION:
                raise GridContractError(
                    f"store grid_version {version} is newer than this reader "
                    f"supports ({GRID_VERSION}) — upgrade model.grid before "
                    "reading this store"
                )
        proj_extent = attrs.get("grid_proj_extent")
        trim_note = attrs.get("grid_trim_note")
        latlon_bias = attrs.get("grid_latlon_bias")
        try:
            return Grid(
                crs=str(attrs["grid_crs"]),
                bounds=tuple(attrs["grid_bounds"]),
                shape=tuple(attrs["grid_shape"]),
                row_order=str(attrs["grid_row_order"]),
                proj_extent=tuple(proj_extent) if proj_extent is not None else None,
                trim_note=str(trim_note) if trim_note is not None else None,
                latlon_bias=tuple(latlon_bias) if latlon_bias is not None else None,
            )
        except GridContractError:
            raise
        except Exception as exc:
            raise GridContractError(f"inconsistent Grid attrs {attrs!r}: {exc}") from exc

    @staticmethod
    def from_zarr(group, array_name: str = "radar") -> "Grid":
        """Read the Grid from a zarr group's attrs, and cross-check it against
        the trailing (rows, cols) shape of `array_name` (default "radar").
        Raises GridContractError on missing attrs OR a shape mismatch."""
        grid = Grid.from_attrs(dict(group.attrs))
        if array_name not in group:
            raise GridContractError(
                f"cannot cross-check Grid: no array {array_name!r} in this store"
            )
        arr_shape = tuple(int(x) for x in group[array_name].shape[-2:])
        if arr_shape != grid.shape:
            raise GridContractError(
                f"Grid attrs shape {grid.shape} does not match {array_name!r} "
                f"array shape {arr_shape} — store attrs are stale or wrong"
            )
        return grid

    # -- geometry ---------------------------------------------------------

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) 2-D float32 arrays of shape `self.shape`, cell
        centres, row 0 = north. Regular grids: linspace across bounds. The
        legacy grid: reproduce geo.grid_latlon() exactly (same trim, same
        linspace-in-projected-space-then-inverse-project, same bias), so the
        two agree to 1e-4 deg everywhere."""
        if self.crs == "EPSG:4326":
            w, s, e, n = self.bounds
            h, wid = self.shape
            lons = np.linspace(w, e, wid)
            lats = np.linspace(n, s, h)  # row 0 = north
            lon, lat = np.meshgrid(lons, lats)
            return lat.astype("float32"), lon.astype("float32")
        if self.proj_extent is None:
            raise GridContractError(
                f"projected grid (crs={self.crs!r}) has no proj_extent; cannot "
                "compute latlon()"
            )
        return _legacy_latlon(self.shape, self.proj_extent, crs=self.crs, bias=self.latlon_bias)

    def envelope(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) CORNER ENVELOPE of `latlon()` — the
        smallest axis-aligned lon/lat box that contains every cell centre.

        For a regular (EPSG:4326) grid this is identical to `bounds`. For a
        PROJECTED grid whose rows/columns are not lon/lat-aligned (the legacy
        KNMI stereographic analysis grid: the south row's latitude varies
        ~0.475 deg / ~53 km west->east, the east column's longitude varies
        ~1.71 deg / ~115 km north->south — see `inner_rectangle()`), this
        box is NOT the grid's true footprint: it over-claims area the grid
        does not actually cover near every edge except the one row/column
        that touches the envelope's extremum. Safe wherever a SUPERSET of
        the domain is wanted (a WMS/DEM fetch region, a reproject
        destination target that itself carries the true CRS) — unsafe
        wherever the box itself is later treated as a lat/lon rectangle to
        paint, crop, or bin data onto cell-for-cell (use `inner_rectangle()`
        for a conservative subset, or reproject properly)."""
        lat, lon = self.latlon()
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

    def inner_rectangle(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) largest lon/lat rectangle GUARANTEED to
        be covered by every row and every column of `latlon()`.

        west/east are chosen so every row's own lon range contains
        [west, east]; south/north so every column's own lat range contains
        [south, north]. For a regular grid this equals `envelope()` /
        `bounds`. For the legacy stereographic grid it is strictly smaller
        than `envelope()` — the gap is the curvature `envelope()` over-claims
        (see its docstring). Use this wherever a lat/lon box must actually be
        a SUBSET of the true domain (e.g. a crop or a sanity bound), never
        `envelope()`/`bounds` for that purpose."""
        lat, lon = self.latlon()
        west = float(lon.min(axis=1).max())
        east = float(lon.max(axis=1).min())
        south = float(lat.min(axis=0).max())
        north = float(lat.max(axis=0).min())
        return west, south, east, north

    def edge_bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) footprint EDGES — `bounds` inflated by
        half a cell in each direction. Use this (or `transform()`) wherever
        `bounds` would otherwise be fed to something that paints pixel edges
        (backend cache/colormap/model/verify, web RadarMap, flutter
        OverlayImage) — feeding it the centre-envelope `bounds` instead shifts
        the painted content by half a cell.

        Same conversion as the module-level `centre_to_edge_bounds()`, which
        callers holding loose (bounds, shape) pairs — e.g. an npz's `bounds`
        on its way into a regrid — should use."""
        return centre_to_edge_bounds(self.bounds, self.shape)

    def transform(self) -> tuple[float, float, float, float]:
        """(x0, y0, dx, dy) north-up geotransform: (x0, y0) is the upper-left
        CORNER (not the row-0/col-0 cell centre) and (dx, dy) is the cell size
        — both dx, dy positive. Feed to e.g.
        ``rasterio.transform.from_origin(x0, y0, dx, dy)``. Regular grids: x/y
        are lon/lat degrees. Projected grids: x/y are in the CRS's projected
        units (from `proj_extent`, not `bounds`)."""
        h, wid = self.shape
        if self.crs == "EPSG:4326":
            w, s, e, n = self.bounds
            dx = (e - w) / (wid - 1) if wid > 1 else 0.0
            dy = (n - s) / (h - 1) if h > 1 else 0.0
            return (w - dx / 2, n + dy / 2, dx, dy)
        if self.proj_extent is None:
            raise GridContractError(
                f"projected grid (crs={self.crs!r}) has no proj_extent; cannot "
                "compute transform()"
            )
        xmin, ymin, xmax, ymax = self.proj_extent
        dx = (xmax - xmin) / (wid - 1) if wid > 1 else 0.0
        dy = (ymax - ymin) / (h - 1) if h > 1 else 0.0
        return (xmin - dx / 2, ymax + dy / 2, dx, dy)

    def cell_of(self, lat: float, lon: float) -> tuple[int, int] | None:
        """(row, col) of the cell containing (lat, lon) for a *regular*
        lat/lon grid, or None if outside the cell footprint (`edge_bounds()`,
        i.e. `bounds` inflated by half a cell — a point inside a boundary
        cell's own footprint but just past its centre must still resolve to
        that cell). Nearest-cell-centre index, not a georeferenced sample —
        do not use for projected grids."""
        if self.crs != "EPSG:4326":
            raise GridContractError("cell_of() only supports regular (EPSG:4326) grids")
        ew, es, ee, en = self.edge_bounds()
        eps = 1e-6 * max(abs(ee - ew), abs(en - es), 1.0)
        if not (ew - eps <= lon <= ee + eps and es - eps <= lat <= en + eps):
            return None
        w, s, e, n = self.bounds
        h, wid = self.shape
        col = 0 if wid == 1 else round((lon - w) / (e - w) * (wid - 1))
        row = 0 if h == 1 else round((n - lat) / (n - s) * (h - 1))  # row 0 = north
        row = min(max(int(row), 0), h - 1)
        col = min(max(int(col), 0), wid - 1)
        return row, col

    def bounds_of_cell(self, row: int, col: int) -> tuple[float, float, float, float]:
        """(west, south, east, north) envelope of one cell for a *regular*
        lat/lon grid — the cell centred on `latlon()[row, col]`, extending
        half a cell spacing in each direction. Adjacent cells tile exactly
        (cell (r, c)'s east edge equals cell (r, c+1)'s west edge)."""
        if self.crs != "EPSG:4326":
            raise GridContractError("bounds_of_cell() only supports regular (EPSG:4326) grids")
        h, wid = self.shape
        if not (0 <= row < h and 0 <= col < wid):
            raise GridContractError(f"cell ({row}, {col}) out of range for shape {self.shape}")
        w, s, e, n = self.bounds
        dlon = (e - w) / (wid - 1) if wid > 1 else 0.0
        dlat = (n - s) / (h - 1) if h > 1 else 0.0
        lon_c = w + col * dlon
        lat_c = n - row * dlat  # row 0 = north
        return (lon_c - dlon / 2, lat_c - dlat / 2, lon_c + dlon / 2, lat_c + dlat / 2)


def _legacy_trimmed_extent() -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) projected extent of the legacy grid, trimmed
    to the northern 700/765 rows — the single source for this trim, used by
    both `Grid.legacy_knmi_analysis()` and `geo.analysis_grid_dst()`."""
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", _LEGACY_PROJ4, always_xy=True)
    xs, ys = [], []
    for lon, lat in _LEGACY_CORNERS_LONLAT:
        x, y = to_xy.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    y_trim = ymax - (_LEGACY_TRIM_ROWS / _LEGACY_NATIVE_ROWS) * (ymax - ymin)
    return (xmin, y_trim, xmax, ymax)


def _legacy_latlon(
    shape: tuple[int, int],
    proj_extent: tuple[float, float, float, float],
    crs: str = _LEGACY_PROJ4,
    bias: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared implementation for the legacy grid's lat/lon, used by both
    `Grid.legacy_knmi_analysis()` (to compute `bounds`) and `Grid.latlon()`
    (to reproduce geo.grid_latlon() exactly). `bias` defaults to the current
    `PLUVIO_GRID_LATLON_BIAS`-derived value when not given explicitly."""
    h, w = shape
    xmin, ymin, xmax, ymax = proj_extent
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cx = np.linspace(xmin, xmax, w)
    cy = np.linspace(ymax, ymin, h)  # row 0 = north (ymax)
    gx, gy = np.meshgrid(cx, cy)
    lon, lat = to_ll.transform(gx, gy)
    dlat, dlon = bias if bias is not None else _legacy_bias()
    return (lat + dlat).astype("float32"), (lon + dlon).astype("float32")
