"""Verification gate for adding a radar source to the composite.

Nothing enters the composite unverified — the a1gate incident (every scan of a radar
rotated by a random angle, read as "93-98% of cells flip at every intensity") went
undetected for days because fields LOOK plausible frame by frame. Two cheap tests
catch that whole bug class:

  self-correlation   the radar's own field across two consecutive scans. A scrambling
                     or rotation bug drives this to ~0 or negative at ALL intensities;
                     sparse drizzle also scores low, so judge alongside wet%.
  geo-correlation    the field against a reference that already got the geography
                     right: the max-composite of verified radars where they overlap,
                     or (--opera-ref) the pan-European OPERA COMP composite for
                     candidates far from our own coverage. Geometry errors (wrong
                     azimuth zero, wrong site, wrong range scale) show as ~0 or
                     NEGATIVE correlation.

Verdicts on a near-dry footprint (< ~1% wet) are unreliable in both directions:
re-verify when the radar's area actually has weather.

Usage:
    PLUVIO_RADAR_CAL=... python -m tools.verify_radar --candidates frabb,frtra \\
        --reference nlhrw,behel,deess --t1 20260831T1300
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
LOG = logging.getLogger("pluvio.verify_radar")


def opera_ref(stamp, shape, bounds):
    """Reference field from the pan-EU OPERA COMP DBZH tiff (fetched if needed)."""
    import urllib.request

    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    fn = f"OPERA@{stamp[:8]}T{stamp[9:13]}@0@DBZH.tiff"
    day = f"{stamp[0:4]}/{stamp[4:6]}/{stamp[6:8]}"
    cache = pathlib.Path("/opt/pluvio/cache/opera_comp")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / fn
    if not (path.exists() and path.stat().st_size > 0):
        url = f"https://s3.waw3-1.cloudferro.com/openradar-24h/{day}/OPERA/COMP/{fn}"
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
            fh.write(r.read())
    with rasterio.open(path) as src:
        dbz = src.read(1, masked=True)
        tr, crs = src.transform, src.crs
    rate = np.zeros(dbz.shape, "float32")
    wet = ~dbz.mask & (dbz.data > 7.0)
    rate[wet] = (10.0 ** (dbz.data[wet].astype("float32") / 10.0) / 200.0) ** (1.0 / 1.6)
    rate[dbz.mask] = np.nan
    w, sth, e, n = bounds
    dst = np.full(shape, np.nan, "float32")
    reproject(rate, dst, src_transform=tr, src_crs=crs,
              dst_transform=from_bounds(w, sth, e, n, shape[1], shape[0]),
              dst_crs=CRS.from_epsg(4326), resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def field(radar, stamp, shape, bounds):
    from tools.radar_single_site import polar_to_grid
    from tools import rtcor_chain as rc

    try:
        sw = rc.read_sweeps_any(radar, stamp)
        if not sw:
            return None
        return rc.single_radar_h(sw, shape, bounds, polar_to_grid)[0]
    except Exception as exc:
        LOG.warning("%s @ %s unreadable: %s", radar, stamp, exc)
        return None


def logcorr(a, b, joint_wet_mask):
    if joint_wet_mask.sum() < 300:
        return float("nan")
    return float(np.corrcoef(np.log10(np.nan_to_num(a[joint_wet_mask]) + 0.05),
                             np.log10(np.nan_to_num(b[joint_wet_mask]) + 0.05))[0, 1])


def main(argv=None) -> int:
    import datetime as dt

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--reference", default="behel,bejab,bewid,deess,denhb,deasb")
    p.add_argument("--t1", required=True, help="YYYYmmddTHHMM")
    p.add_argument("--grid-n", type=int, default=768)
    p.add_argument("--bounds", default=None,
                   help="W,S,E,N box for the comparison grid (default: model.geo bbox). "
                        "Give a box around the candidate for radars far from Belgium.")
    p.add_argument("--opera-ref", action="store_true",
                   help="reference = pan-EU OPERA composite instead of our own radars "
                        "(for candidates with no overlap with the verified set)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))
    os.environ.setdefault("PLUVIO_SWEEP_MERGE", "local")
    os.environ.setdefault("PLUVIO_LOCAL_DH", "1200")
    os.environ.setdefault("PLUVIO_VPR_STATE", "")

    from model.geo import GRID, bbox
    if args.bounds:
        bounds = tuple(float(x) for x in args.bounds.split(","))
    else:
        bounds = bbox()
    t1 = args.t1
    t2 = (dt.datetime.strptime(t1, "%Y%m%dT%H%M")
          + dt.timedelta(minutes=5)).strftime("%Y%m%dT%H%M")

    if args.opera_ref:
        refmax = opera_ref(t1, GRID, bounds)
        print("reference: OPERA COMP composite")
    else:
        ref = [f for f in (field(r, t1, GRID, bounds)
                           for r in args.reference.split(",")) if f is not None]
        if not ref:
            LOG.error("no reference radars available at %s", t1)
            return 2
        refmax = np.nanmax(np.stack(ref), axis=0)
        print(f"reference radars: {len(ref)}")
    print(f"{'radar':7s} {'selfcorr':>9s} {'geocorr':>9s} {'wet%':>7s}  verdict")
    for r in args.candidates.split(","):
        a = field(r, t1, GRID, bounds)
        b = field(r, t2, GRID, bounds)
        if a is None or b is None:
            print(f"{r:7s} no data")
            continue
        m = (np.isfinite(a) & np.isfinite(b)
             & ((np.nan_to_num(a) > 0.1) | (np.nan_to_num(b) > 0.1)))
        mo = (np.isfinite(a) & np.isfinite(refmax)
              & ((np.nan_to_num(a) > 0.1) | (np.nan_to_num(refmax) > 0.1)))
        sc, gc = logcorr(a, b, m), logcorr(a, refmax, mo)
        wet = 100 * float(np.nanmean(np.nan_to_num(a) > 0.1))
        ok = (np.isnan(gc) or gc > 0.05) and (np.isnan(sc) or sc > 0.0)
        note = " (near-dry: unreliable, re-verify on a wet day)" if wet < 1.0 else ""
        print(f"{r:7s} {sc:9.3f} {gc:9.3f} {wet:6.2f}%  "
              f"{'PASS' if ok else 'FAIL'}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
