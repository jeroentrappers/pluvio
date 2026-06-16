"""EUMETSAT MTG Lightning Imager — gridded flash density for the model.

Collects the LI Level-2 **Accumulated Flashes (AF, EO:EUM:DAT:0686)** and,
optionally, **Accumulated Flash Radiance (AFR, 0688)** from the EUMETSAT Data
Store via EUMDAC. These are sparse points in the MTG geostationary projection
(x, y scan-angles + flash counts over 10-min windows); we project them to
lat/lon and bin onto a regular grid over our BeNeLux+context domain, written as
a small GeoTIFF per window — same downstream path as the OPERA/MSG rasters.

Auth: EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET (env). Free EUMETSAT
account → EUMDAC consumer key/secret.

    python collectors/fetch_mtg_lightning.py --mode forward --out data/mtg_li
    python collectors/fetch_mtg_lightning.py --mode backfill \
        --start 2024-09-01 --end 2024-10-01 --out data/mtg_li
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile

LOG = logging.getLogger("pluvio.fetch_mtg_li")

COLLECTIONS = {"AF": "EO:EUM:DAT:0686", "AFR": "EO:EUM:DAT:0688"}
VAR = {"AF": "flash_accumulation", "AFR": "flash_radiance"}
DEFAULT_BBOX = (-2.0, 47.0, 12.0, 56.0)  # W, S, E, N (EPSG:4326)
DEFAULT_RES = 0.02  # ~2 km, matches the LI/FCI grid


def _grid_axes(bbox, res):
    import numpy as np

    w, s, e, n = bbox
    lons = np.arange(w, e, res)
    lats = np.arange(n, s, -res)  # north-up
    return lats, lons


def _token():
    import eumdac

    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit("set EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET")
    return eumdac.AccessToken((key, secret))


def _geos_to_lonlat(x, y, proj_var):
    """Project MTG geostationary scan-angles (x, y in radians) to lon/lat."""
    import numpy as np
    import pyproj

    a = proj_var.attrs
    h = float(a["perspective_point_height"])
    lon0 = float(a.get("longitude_of_projection_origin", 0.0))
    sweep = a.get("sweep_angle_axis", "y")
    semi_major = float(a.get("semi_major_axis", 6378137.0))
    semi_minor = float(a.get("semi_minor_axis", 6356752.31424518))
    geos = pyproj.CRS.from_proj4(
        f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} "
        f"+a={semi_major} +b={semi_minor} +units=m +no_defs"
    )
    tf = pyproj.Transformer.from_crs(geos, "EPSG:4326", always_xy=True)
    # CF scan-angle coords → projection metres = angle × height.
    lon, lat = tf.transform(np.asarray(x) * h, np.asarray(y) * h)
    return lon, lat


def grid_product(nc_path: str, product: str, bbox, res) -> "object":
    """Bin one AF/AFR netCDF onto a regular lat/lon grid (sum per cell)."""
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    val = ds[VAR[product]].values.astype("float64")
    lon, lat = _geos_to_lonlat(ds["x"].values, ds["y"].values, ds["mtg_geos_projection"])
    lats, lons = _grid_axes(bbox, res)
    w, s, e, n = bbox
    m = np.isfinite(lon) & np.isfinite(lat) & (lon >= w) & (lon < e) & (lat >= s) & (lat < n) & (val > 0)
    grid = np.zeros((len(lats), len(lons)), dtype="float32")
    if m.any():
        col = ((lon[m] - w) / res).astype(int)
        row = ((n - lat[m]) / res).astype(int)
        np.add.at(grid, (row, col), val[m])
    return grid


def write_geotiff(grid, bbox, res, out_path: pathlib.Path):
    import rasterio
    from rasterio.transform import from_origin

    w, _, _, n = bbox
    transform = from_origin(w, n, res, res)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tiff")
    with rasterio.open(
        tmp, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, compress="deflate",
        predictor=2, tiled=True, nodata=0.0,
    ) as dst:
        dst.write(grid, 1)
    os.replace(tmp, out_path)


def _disk_free_gb(p: pathlib.Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1e9


def collect(products, bbox, res, out_dir, start, end, min_free_gb):
    import eumdac

    ds = eumdac.DataStore(_token())
    n_ok = n_skip = n_fail = 0
    for product in products:
        col = ds.get_collection(COLLECTIONS[product])
        prods = list(col.search(dtstart=start, dtend=end))
        LOG.info("%s: %d products in %s..%s", product, len(prods), start, end)
        for p in prods:
            if _disk_free_gb(out_dir) < min_free_gb:
                LOG.error("disk below %.0f GB free — stopping", min_free_gb)
                return {"written": n_ok, "skipped": n_skip, "failed": n_fail}
            # Accumulation-end time = the last 14-digit timestamp in the product
            # id (after the processing time and accumulation-start time).
            ts14 = re.findall(r"\d{14}", str(p))
            if not ts14:
                n_fail += 1
                continue
            stamp = ts14[-1]  # YYYYMMDDhhmmss
            out_path = out_dir / product / stamp[:8] / f"{stamp}_{product}.tiff"
            if out_path.exists():
                n_skip += 1
                continue
            body = [e for e in p.entries if e.endswith(".nc") and "BODY" in e]
            if not body:
                n_fail += 1
                continue
            tmp = tempfile.mkdtemp()
            try:
                local = os.path.join(tmp, "li.nc")
                with p.open(entry=body[0]) as src, open(local, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                grid = grid_product(local, product, bbox, res)
                write_geotiff(grid, bbox, res, out_path)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("failed %s (%s)", stamp, exc)
                n_fail += 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        LOG.info("%s done: %d written, %d skipped, %d failed", product, n_ok, n_skip, n_fail)
    return {"written": n_ok, "skipped": n_skip, "failed": n_fail}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["forward", "backfill"], default="forward")
    p.add_argument("--products", default="AF", help="comma list: AF,AFR")
    p.add_argument("--start", help="backfill start YYYY-MM-DD")
    p.add_argument("--end", help="backfill end YYYY-MM-DD (exclusive)")
    p.add_argument("--forward-hours", type=int, default=3, help="forward: how far back to scan")
    p.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)), help="W,S,E,N EPSG:4326")
    p.add_argument("--res", type=float, default=DEFAULT_RES, help="grid resolution, degrees")
    p.add_argument("--out", default="data/mtg_li")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    bbox = tuple(float(x) for x in args.bbox.split(","))
    products = [x.strip().upper() for x in args.products.split(",") if x.strip()]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "forward":
        end = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        start = end - dt.timedelta(hours=args.forward_hours)
    else:
        if not args.start:
            LOG.error("--start required for backfill")
            return 2
        start = dt.datetime.fromisoformat(args.start)
        end = dt.datetime.fromisoformat(args.end) if args.end else dt.datetime.now(dt.UTC).replace(tzinfo=None)

    summary = collect(products, bbox, args.res, out_dir, start, end, args.min_free_gb)
    LOG.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
