"""Diagnostic: why does ERA5 netcdf -> analysis grid warp produce all-NaN?"""
import sys

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

sys.path.insert(0, "/app")
from model.geo import analysis_grid_dst

WGS84 = CRS.from_epsg(4326)

nc = sys.argv[1]
dst_crs, dst_transform, (h, w) = analysis_grid_dst()
print("dst_crs:", dst_crs)
print("dst_transform:", dst_transform)

sds = f"netcdf:{nc}:total_precipitation"
with rasterio.open(sds) as src:
    print("src.crs:", src.crs)
    print("src.transform:", src.transform)
    print("src.bounds:", src.bounds)
    print("src.shape:", src.shape, "count:", src.count, "nodata:", src.nodata)
    arr = src.read(1)
    src_transform = src.transform
    print("native band1: dtype", arr.dtype, "finite", np.isfinite(arr).mean(),
          "min", np.nanmin(arr), "max", np.nanmax(arr))

# warp the ndarray directly with explicit georef (avoids netcdf-band warp quirks)
dst = np.full((h, w), np.nan, dtype="float32")
reproject(source=arr, destination=dst,
          src_transform=src_transform, src_crs=WGS84,
          dst_transform=dst_transform, dst_crs=dst_crs,
          resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
print("warped(ndarray): finite", np.isfinite(dst).mean(),
      "min", np.nanmin(dst) if np.isfinite(dst).any() else "n/a",
      "max", np.nanmax(dst) if np.isfinite(dst).any() else "n/a")
