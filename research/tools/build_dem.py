"""Mosaic a domain DEM from the open Copernicus 30 m COGs.

Why this exists: beam blockage is the quality term ODYSSEY weights by and the reason we
still lose to OPERA in the Ardennes. Computed against the analysis grid's own
`elevation_m` (256x256, ~3 km) it finds essentially NO blockage — mean quality 0.89-1.00
and 0% of cells partly blocked — because 3 km terrain is too smooth to clip a beam. That
is a resolution artefact, not a physical result, so the DEM has to come from elsewhere.

Copernicus DSM 30 m is public on S3 with no authentication, as Cloud-Optimised GeoTIFF.
Reading each 1-degree tile through its overviews gives a domain mosaic at a chosen
resolution without downloading ~88 full tiles.

    python -m tools.build_dem --res-m 500 --out /opt/pluvio/radarproc/dem_500m.npz
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

LOG = logging.getLogger("pluvio.build_dem")

URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
       "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def tile_url(lat: int, lon: int) -> str:
    return URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                      ew="E" if lon >= 0 else "W", lon=abs(lon))


def build(bounds, res_m: float = 500.0):
    """Return (dem, (w, s, e, n), (ny, nx)) covering `bounds` at ~res_m."""
    import rasterio
    from rasterio.errors import RasterioIOError

    w, s, e, n = bounds
    deg_lat = res_m / 111320.0
    ny = int(np.ceil((n - s) / deg_lat))
    nx = int(np.ceil((e - w) / deg_lat))
    dem = np.full((ny, nx), np.nan, "float32")
    lats = n - (np.arange(ny) + 0.5) * (n - s) / ny
    lons = w + (np.arange(nx) + 0.5) * (e - w) / nx

    got = 0
    for tlat in range(int(np.floor(s)), int(np.ceil(n))):
        for tlon in range(int(np.floor(w)), int(np.ceil(e))):
            rows = np.where((lats >= tlat) & (lats < tlat + 1))[0]
            cols = np.where((lons >= tlon) & (lons < tlon + 1))[0]
            if not len(rows) or not len(cols):
                continue
            url = "/vsicurl/" + tile_url(tlat, tlon)
            try:
                with rasterio.open(url) as src:
                    # Read the tile downsampled to just what this block needs; the COG
                    # overviews make this cheap and avoid pulling the full 3600x3600.
                    arr = src.read(1, out_shape=(len(rows), len(cols)),
                                   resampling=rasterio.enums.Resampling.average)
                    nod = src.nodata
                if nod is not None:
                    arr = np.where(arr == nod, np.nan, arr)
                dem[np.ix_(rows, cols)] = arr.astype("float32")
                got += 1
            except (RasterioIOError, Exception) as exc:   # ocean tiles simply do not exist
                LOG.debug("tile %s%s missing (%s)", tlat, tlon, exc)
    LOG.info("mosaicked %d tiles -> %dx%d at ~%.0f m", got, ny, nx, res_m)
    # Tiles are absent over sea, where the true height is 0.
    return np.nan_to_num(dem, nan=0.0), (w, s, e, n), (ny, nx)


def main(argv=None) -> int:
    import os
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res-m", type=float, default=500.0)
    p.add_argument("--out", default="/opt/pluvio/radarproc/dem_500m.npz")
    p.add_argument("--grid-n", type=int, default=256)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))
    from model.geo import bbox

    dem, bnds, shape = build(bbox(), args.res_m)
    np.savez_compressed(args.out, dem=dem, bounds=np.array(bnds), res_m=args.res_m)
    LOG.info("wrote %s  min %.0f max %.0f m", args.out, float(dem.min()), float(dem.max()))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
