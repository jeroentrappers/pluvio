"""Turn ECMWF open-data GRIB precipitation into model-grid rate fields.

Two steps the seamless model's NWP context path needs:
  1. read `tp` (accumulated mm from forecast start) off a 0.25° lat/lon GRIB,
  2. bilinearly regrid it onto a target lat/lon grid (the analysis grid, or a
     wider context grid), and difference consecutive steps → mm/h.

Kept dependency-light: xarray+cfgrib to read, scipy to interpolate. No domain
assumptions baked in — pass whatever target (dst_lat, dst_lon) you want, so the
same code serves the narrow radar grid and a wide European context grid.
"""

from __future__ import annotations

import pathlib

import numpy as np


def open_tp(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read `tp` from a GRIB2 file. Returns (lat1d, lon1d, tp2d) with lon in
    [-180, 180]. `tp` is accumulated precipitation (kg m⁻² = mm) from step 0."""
    import xarray as xr

    ds = xr.open_dataset(str(path), engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds["tp"].values.astype("float32")
    lat = ds["latitude"].values.astype("float64")
    lon = ds["longitude"].values.astype("float64")
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lat, lon, tp


def regrid_to(
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    src_field: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a regular (src_lat × src_lon) field at the dst lat/lon
    points. dst_lat/dst_lon are same-shaped 2-D arrays (the target grid)."""
    from scipy.interpolate import RegularGridInterpolator

    la_order = np.argsort(src_lat)
    lo_order = np.argsort(src_lon)
    la = src_lat[la_order]
    lo = src_lon[lo_order]
    f = src_field[np.ix_(la_order, lo_order)]
    interp = RegularGridInterpolator(
        (la, lo), f, method="linear", bounds_error=False, fill_value=np.nan
    )
    pts = np.stack([np.asarray(dst_lat).ravel(), np.asarray(dst_lon).ravel()], axis=-1)
    return interp(pts).reshape(np.asarray(dst_lat).shape).astype("float32")


def rate_mm_per_h(tp_lo: np.ndarray, tp_hi: np.ndarray, hours: float) -> np.ndarray:
    """Mean rate over an interval from two cumulative-tp snapshots `hours` apart."""
    return np.clip((tp_hi - tp_lo) / float(hours), 0.0, None).astype("float32")


def reproject_to_analysis_grid(src_path, band: int = 1, resampling: str = "bilinear",
                               src_crs=None, nodata_as_zero: bool = False) -> np.ndarray:
    """Reproject any georeferenced raster (OPERA LAEA COG, MTG EPSG:4326 GeoTIFF,
    AIFS lat/lon …) onto the 100×100 analysis grid. CRS-agnostic — reads the
    source CRS from the file and warps to the KNMI-stereographic destination
    (model.geo.analysis_grid_dst). Out-of-coverage → NaN.

    `src_crs` overrides the file's CRS — needed for ERA5 NetCDF, whose GDAL
    subdatasets carry a valid geotransform but no CRS tag (it's plain WGS84
    lat/lon, so pass "EPSG:4326").

    `nodata_as_zero` treats source nodata as a real 0 before warping. Set it for
    OPERA RATE/ACRR, where nodata means "no rain" rather than "not measured" —
    without it an interpolating resampler erodes ~88% of the rain (see below).
    Leave it False for MTG/AIFS/ERA5, where nodata genuinely means missing and
    filling zeros would invent data."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    from model.geo import analysis_grid_dst

    dst_crs, dst_transform, (h, w) = analysis_grid_dst()
    dst = np.full((h, w), np.nan, dtype="float32")
    rs = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        if nodata_as_zero:
            # ⚠️ OPERA RATE encodes NO RAIN as nodata (-9999000), so ~92% of the tiff
            # is "nodata" that actually means zero. Warping that with an interpolating
            # resampler makes every target cell touching a nodata source pixel nodata
            # too, which ERODES rain areas from their edges and deletes small ones
            # outright. Measured on 20260830T0730: 7031 raw wet pixels inside the
            # analysis bbox produced just 357 wet cells (12%); filling nodata with 0
            # first gives 2860, matching the ~3100 expected from 2 km -> 3 km.
            arr = src.read(band).astype("float32")
            nod = src.nodata
            if nod is not None:
                arr = np.where(arr == nod, 0.0, arr)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src_crs or src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rs,
                src_nodata=None,
                dst_nodata=np.nan,
            )
        else:
            reproject(
                source=rasterio.band(src, band),
                destination=dst,
                src_transform=src.transform,
                src_crs=src_crs or src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    return dst


def reproject_era5_var(nc_path, var: str, band: int, scale: float = 1.0) -> np.ndarray:
    """Regrid one ERA5 variable at one time-step (1-indexed band = hour-of-month)
    onto the analysis grid. ERA5 NetCDF is WGS84 lat/lon with no CRS tag and its
    GDAL subdataset bands don't warp via rasterio.band(), so we read the band into
    an array and warp the ndarray with explicit EPSG:4326 georef. `scale` converts
    units (e.g. tp m→mm/h = 1000 for hourly accumulation)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject

    from model.geo import analysis_grid_dst

    dst_crs, dst_transform, (h, w) = analysis_grid_dst()
    dst = np.full((h, w), np.nan, dtype="float32")
    with rasterio.open(f"netcdf:{nc_path}:{var}") as src:
        arr = src.read(band)
        src_transform = src.transform
    reproject(source=arr, destination=dst, src_transform=src_transform,
              src_crs=CRS.from_epsg(4326), dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return (dst * scale).astype("float32") if scale != 1.0 else dst
