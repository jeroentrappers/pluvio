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
shape — by every consumer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyproj

GRID_VERSION = 1

# The only row order this contract supports: row 0 is the north edge, the
# last row is the south edge. This matches every producer/consumer today
# (radar DISPLAY_ORIGIN=UL, build_store_v3, infer_latest's backend grid).
_SUPPORTED_ROW_ORDERS = ("north_first",)

# KNMI radar stereographic projection (see geo.py for provenance).
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

# Residual empirical registration calibration applied by geo.grid_latlon()
# (commit df83fbf) on top of the pure corner/trim geometry. Reproduced here,
# with the same env override, so Grid.legacy_knmi_analysis().latlon() agrees
# with geo.grid_latlon() bit-for-bit (not just up to this offset).
def _legacy_bias() -> tuple[float, float]:
    try:
        dlat_s, dlon_s = os.environ.get("PLUVIO_GRID_LATLON_BIAS", "0,0.07").split(",")
        return float(dlat_s), float(dlon_s)
    except ValueError:
        return 0.0, 0.07


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


@dataclass(frozen=True)
class Grid:
    """A raster's georeference: CRS, lon/lat envelope, shape, row order.

    Attributes:
        crs: "EPSG:4326" for a regular lat/lon grid, or a proj4 string for a
            projected grid (e.g. the legacy KNMI stereographic analysis grid).
        bounds: (west, south, east, north) lon/lat *envelope* of the grid.
            For a projected grid this is the envelope of the reprojected cell
            centres, not the CRS-native box — use `proj_extent` for that.
        shape: (rows, cols).
        row_order: only "north_first" is supported (row 0 = north edge).
        proj_extent: (xmin, ymin, xmax, ymax) in CRS units, for projected
            grids only (None for regular lat/lon grids).
        trim_note: free-text note documenting any non-obvious crop of a wider
            native domain (e.g. the legacy 700/765 trim). None if the grid is
            not trimmed from anything.
    """

    crs: str
    bounds: tuple[float, float, float, float]
    shape: tuple[int, int]
    row_order: str = "north_first"
    proj_extent: tuple[float, float, float, float] | None = None
    trim_note: str | None = None

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
        w, s, e, n = bounds
        if not (w < e and s < n):
            raise GridContractError(f"bounds must be (west<east, south<north), got {bounds}")

    # -- constructors ---------------------------------------------------

    @staticmethod
    def regular(bounds: tuple[float, float, float, float], shape: tuple[int, int]) -> "Grid":
        """A regular lat/lon grid: bounds=(west, south, east, north),
        shape=(rows, cols), row 0 = north, cell centres linspace across bounds."""
        return Grid(crs="EPSG:4326", bounds=tuple(bounds), shape=tuple(shape),
                    row_order="north_first")

    @staticmethod
    def legacy_knmi_analysis(shape: tuple[int, int] = (100, 100)) -> "Grid":
        """The legacy KNMI-stereographic analysis grid (geo.py), including its
        700/765 north-only trim of the corner-derived projected extent."""
        to_xy = pyproj.Transformer.from_crs("EPSG:4326", _LEGACY_PROJ4, always_xy=True)
        xs, ys = [], []
        for lon, lat in _LEGACY_CORNERS_LONLAT:
            x, y = to_xy.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        y_trim = ymax - (_LEGACY_TRIM_ROWS / _LEGACY_NATIVE_ROWS) * (ymax - ymin)
        proj_extent = (xmin, y_trim, xmax, ymax)

        lat, lon = _legacy_latlon(shape, proj_extent)
        bounds = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
        return Grid(crs=_LEGACY_PROJ4, bounds=bounds, shape=tuple(shape),
                    row_order="north_first", proj_extent=proj_extent,
                    trim_note=_LEGACY_TRIM_NOTE)

    # -- (de)serialisation ------------------------------------------------

    def to_attrs(self) -> dict[str, Any]:
        """JSON-serialisable dict, keys prefixed `grid_*`, for zarr .attrs."""
        attrs: dict[str, Any] = {
            "grid_version": GRID_VERSION,
            "grid_crs": self.crs,
            "grid_bounds": list(self.bounds),
            "grid_shape": list(self.shape),
            "grid_row_order": self.row_order,
        }
        if self.proj_extent is not None:
            attrs["grid_proj_extent"] = list(self.proj_extent)
        if self.trim_note is not None:
            attrs["grid_trim_note"] = self.trim_note
        return attrs

    @staticmethod
    def from_attrs(attrs: dict[str, Any]) -> "Grid":
        """Reconstruct a Grid from a zarr-attrs-style dict. Raises
        GridContractError with a helpful message if required keys are missing
        or the values are internally inconsistent."""
        required = ("grid_crs", "grid_bounds", "grid_shape", "grid_row_order")
        missing = [k for k in required if k not in attrs]
        if missing:
            raise GridContractError(
                f"store attrs are missing Grid keys {missing!r} — this store "
                "predates the Grid contract (1.1); either rebuild it or fall "
                "back to legacy geometry explicitly, never assume a grid."
            )
        proj_extent = attrs.get("grid_proj_extent")
        trim_note = attrs.get("grid_trim_note")
        try:
            return Grid(
                crs=str(attrs["grid_crs"]),
                bounds=tuple(attrs["grid_bounds"]),
                shape=tuple(attrs["grid_shape"]),
                row_order=str(attrs["grid_row_order"]),
                proj_extent=tuple(proj_extent) if proj_extent is not None else None,
                trim_note=str(trim_note) if trim_note is not None else None,
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
        linspace-in-projected-space-then-inverse-project), so the two agree
        to 1e-4 deg everywhere."""
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
        return _legacy_latlon(self.shape, self.proj_extent, crs=self.crs)

    def cell_of(self, lat: float, lon: float) -> tuple[int, int] | None:
        """(row, col) of the cell containing (lat, lon) for a *regular*
        lat/lon grid, or None if outside `bounds`. Nearest-cell-centre index,
        not a georeferenced sample — do not use for projected grids."""
        if self.crs != "EPSG:4326":
            raise GridContractError("cell_of() only supports regular (EPSG:4326) grids")
        w, s, e, n = self.bounds
        eps = 1e-6 * max(abs(e - w), abs(n - s), 1.0)
        if not (w - eps <= lon <= e + eps and s - eps <= lat <= n + eps):
            return None
        h, wid = self.shape
        col = 0 if wid == 1 else round((lon - w) / (e - w) * (wid - 1))
        row = 0 if h == 1 else round((n - lat) / (n - s) * (h - 1))  # row 0 = north
        row = min(max(int(row), 0), h - 1)
        col = min(max(int(col), 0), wid - 1)
        return row, col

    def bounds_of_cell(self, row: int, col: int) -> tuple[float, float, float, float]:
        """(west, south, east, north) envelope of one cell for a *regular*
        lat/lon grid — the cell centred on `latlon()[row, col]`, extending
        half a cell spacing in each direction."""
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


def _legacy_latlon(
    shape: tuple[int, int],
    proj_extent: tuple[float, float, float, float],
    crs: str = _LEGACY_PROJ4,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared implementation for the legacy grid's lat/lon, used by both
    `Grid.legacy_knmi_analysis()` (to compute `bounds`) and `Grid.latlon()`
    (to reproduce geo.grid_latlon() exactly)."""
    h, w = shape
    xmin, ymin, xmax, ymax = proj_extent
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cx = np.linspace(xmin, xmax, w)
    cy = np.linspace(ymax, ymin, h)  # row 0 = north (ymax)
    gx, gy = np.meshgrid(cx, cy)
    lon, lat = to_ll.transform(gx, gy)
    dlat, dlon = _legacy_bias()
    return (lat + dlat).astype("float32"), (lon + dlon).astype("float32")
