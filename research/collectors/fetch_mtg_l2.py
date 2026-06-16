"""EUMETSAT MTG-FCI Level-2 gridded products — convective-precursor channels.

One config-driven collector for the useful gridded L2 products. Each is a dense
full-disk netCDF; we read it with satpy (the EUMETSAT-supported reader, which
handles the GEOS geolocation), resample onto our BeNeLux+context grid, and write
one small GeoTIFF per (timestamp, variable) — same downstream path as the other
raster channels.

Products (all EO:EUM:DAT, open via the same EUMDAC creds as MTG-LI):
  GII   0683  instability + moisture (k_index, lifted_index, precipitable water)
  CTTH  0681  cloud top temperature / height / pressure
  OCA   0684  optimal cloud analysis (optical depth, top pressure, phase)
  CT    0680  cloud type / phase (categorical)
  OLR   0685  outgoing longwave radiation

Auth: EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET (env).

    python collectors/fetch_mtg_l2.py --mode forward --products GII,CTTH --out data/mtg_l2
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
import warnings

warnings.filterwarnings("ignore")
LOG = logging.getLogger("pluvio.fetch_mtg_l2")

DEFAULT_BBOX = (-2.0, 47.0, 12.0, 56.0)
DEFAULT_RES = 0.02

# product -> (collection id, satpy reader, [desired dataset names])
PRODUCTS: dict[str, tuple[str, str, list[str]]] = {
    "GII":  ("EO:EUM:DAT:0683", "fci_l2_nc", ["k_index", "lifted_index", "prec_water_total"]),
    "CTTH": ("EO:EUM:DAT:0681", "fci_l2_nc", ["cloud_top_temperature", "cloud_top_height", "cloud_top_pressure"]),
    "OCA":  ("EO:EUM:DAT:0684", "fci_l2_nc", ["retrieved_cloud_optical_thickness", "retrieved_cloud_top_height", "retrieved_cloud_phase"]),
    "CT":   ("EO:EUM:DAT:0680", "fci_l2_nc", ["cloud_type", "cloud_phase"]),
    "OLR":  ("EO:EUM:DAT:0685", "fci_l2_nc", ["olr"]),
}


def _token():
    import eumdac

    key, secret = os.environ.get("EUMETSAT_CONSUMER_KEY"), os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not key or not secret:
        raise SystemExit("set EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET")
    return eumdac.AccessToken((key, secret))


def _area(bbox, res):
    from pyresample import create_area_def

    return create_area_def("pluvio", "EPSG:4326", area_extent=list(bbox), resolution=res)


def _disk_free_gb(p: pathlib.Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1e9


def process_product(files, reader, desired, area, out_dir, prod_key, stamp):
    """satpy-read + resample requested datasets; write a GeoTIFF each. Returns count."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from satpy import Scene

    scn = Scene(filenames=files, reader=reader)
    avail = set(scn.available_dataset_names())
    load = [d for d in desired if d in avail]
    if not load:
        LOG.warning("%s: none of %s available (have %s)", prod_key, desired, sorted(avail)[:8])
        return 0
    scn.load(load)
    out = scn.resample(area, resampler="nearest", radius_of_influence=4000)
    w, s, e, n = area.area_extent[0], area.area_extent[1], area.area_extent[2], area.area_extent[3]
    res = (e - w) / area.width
    transform = from_origin(w, n, res, res)
    written = 0
    for d in load:
        arr = np.asarray(out[d].values, dtype="float32")
        out_path = out_dir / prod_key / d / stamp[:8] / f"{stamp}_{d}.tiff"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.tiff")
        with rasterio.open(
            tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
            dtype="float32", crs="EPSG:4326", transform=transform, compress="deflate", tiled=True,
        ) as dst:
            dst.write(arr, 1)
        os.replace(tmp, out_path)
        written += 1
    return written


def collect(products, bbox, res, out_dir, start, end, min_free_gb):
    import eumdac

    ds = eumdac.DataStore(_token())
    area = _area(bbox, res)
    n_ok = n_skip = n_fail = 0
    for prod_key in products:
        cid, reader, desired = PRODUCTS[prod_key]
        col = ds.get_collection(cid)
        prods = list(col.search(dtstart=start, dtend=end))
        LOG.info("%s (%s): %d products in window", prod_key, cid, len(prods))
        for p in prods:
            if _disk_free_gb(out_dir) < min_free_gb:
                LOG.error("disk below %.0f GB free — stopping", min_free_gb)
                return {"written": n_ok, "skipped": n_skip, "failed": n_fail}
            ts14 = re.findall(r"\d{14}", str(p))
            if not ts14:
                n_fail += 1
                continue
            stamp = ts14[-1]
            marker = out_dir / prod_key / ".done" / f"{stamp}"
            if marker.exists():
                n_skip += 1
                continue
            tmp = tempfile.mkdtemp()
            try:
                files = []
                for e in p.entries:
                    if e.endswith(".nc"):
                        local = os.path.join(tmp, os.path.basename(e))
                        with p.open(entry=e) as src, open(local, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        files.append(local)
                if process_product(files, reader, desired, area, out_dir, prod_key, stamp):
                    n_ok += 1
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.touch()
                else:
                    n_fail += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s %s failed (%s)", prod_key, stamp, exc)
                n_fail += 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        LOG.info("%s done: %d written, %d skipped, %d failed", prod_key, n_ok, n_skip, n_fail)
    return {"written": n_ok, "skipped": n_skip, "failed": n_fail}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["forward", "backfill"], default="forward")
    p.add_argument("--products", default="GII,CTTH,OCA,CT,OLR", help="comma list of " + ",".join(PRODUCTS))
    p.add_argument("--start", help="backfill start YYYY-MM-DD")
    p.add_argument("--end", help="backfill end YYYY-MM-DD")
    p.add_argument("--forward-hours", type=int, default=2)
    p.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)))
    p.add_argument("--res", type=float, default=DEFAULT_RES)
    p.add_argument("--out", default="data/mtg_l2")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    bbox = tuple(float(x) for x in args.bbox.split(","))
    products = [x.strip().upper() for x in args.products.split(",") if x.strip() in PRODUCTS]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "forward":
        end = dt.datetime.now(dt.UTC).replace(tzinfo=None)
        start = end - dt.timedelta(hours=args.forward_hours)
    else:
        if not args.start:
            LOG.error("--start required")
            return 2
        start = dt.datetime.fromisoformat(args.start)
        end = dt.datetime.fromisoformat(args.end) if args.end else dt.datetime.now(dt.UTC).replace(tzinfo=None)

    LOG.info("MTG-L2 collect: %s products=%s bbox=%s", args.mode, products, bbox)
    LOG.info("done: %s", collect(products, bbox, args.res, out_dir, start, end, args.min_free_gb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
