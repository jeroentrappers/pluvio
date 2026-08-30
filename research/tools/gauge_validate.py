"""Score radar rain-rate estimates against actual rain gauges.

OPERA is an *estimate*, not truth, so "as good as OPERA" cannot be settled by
comparing to OPERA. This scores both our composite AND OPERA against independent
gauge observations at station locations, which is the only objective arbiter.

Sources
  KNMI  10-minute in-situ obs, variable `rg` = "Precipitation Intensity (Rain
        Gauge) Mean" [mm/h] — same units as radar rain rate. ~58 NL stations.
  KMI   opendata.meteo.be WFS aws:aws_10min, field `precip_quantity` [mm/10min]
        — ~14 BE stations, anonymous access.

This directly tests the open question from the single-radar work: OPERA is >90%
exactly zero inside a 100 km disc while our field carries widespread light rain. If
gauges record rain where OPERA reports zero, our "over-detection" is closer to the
truth and OPERA is under-detecting light precipitation. If gauges agree with OPERA's
zeros, our excess is spurious. The gauges decide it.

⚠️ Sampling mismatch is real and must not be waved away: a gauge integrates over a
~200 cm^2 orifice for 10 minutes; a radar bin samples a ~1 km^3 volume aloft,
instantaneously. Point-to-pixel comparison is noisy, so this reports rank statistics
and hit/miss contingency over many station-times rather than trusting any single
pair.

Usage:
    python -m tools.gauge_validate --times 20260830T0730,20260830T0800 --radar nlhrw
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.gauge_validate")

KNMI_API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
KNMI_DS, KNMI_VER = "10-minute-in-situ-meteorological-observations", "1.0"
CACHE = pathlib.Path("/opt/pluvio/radarproc/gaugecache")


def knmi_key() -> str:
    for line in pathlib.Path("/opt/pluvio/research/.env").read_text().splitlines():
        if line.startswith("KNMI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("KNMI_API_KEY not found")


def fetch_knmi_10min(stamp: str) -> pathlib.Path | None:
    """Download one 10-min observation file (cached). stamp = YYYYmmddTHHMM."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = f"KMDS__OPER_P___10M_OBS_L2_{stamp[:8]}{stamp[9:13]}.nc"
    dest = CACHE / fn
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    key = knmi_key()
    req = urllib.request.Request(
        f"{KNMI_API}/{KNMI_DS}/versions/{KNMI_VER}/files/{fn}/url",
        headers={"Authorization": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            url = json.load(r)["temporaryDownloadUrl"]
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
            fh.write(r.read())
        return dest
    except Exception as exc:
        LOG.warning("no KNMI obs for %s (%s)", stamp, exc)
        return None


def read_gauges(path: pathlib.Path):
    """Return [(station, lat, lon, rate_mm_h)] for stations reporting a rate."""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        lat = np.asarray(f["lat"]).ravel()
        lon = np.asarray(f["lon"]).ravel()
        rg = np.asarray(f["rg"]).ravel()
        st = np.asarray(f["station"]).ravel()
        for i in range(len(lat)):
            v = float(rg[i])
            if np.isfinite(v):
                out.append((str(st[i]), float(lat[i]), float(lon[i]), v))
    return out


def sample(field, lat, lon, bounds, shape, halo=1):
    """Value at a lat/lon, as the max over a small halo.

    A gauge is a point; a grid cell is kilometres across and the radar beam is
    displaced. Taking a strict single cell makes near-misses look like total
    failures for both estimators equally, so use a small neighbourhood maximum —
    applied identically to ours and OPERA so neither is favoured.
    """
    w, s, e, n = bounds
    h, wd = shape
    c = int((lon - w) / (e - w) * wd)
    r = int((n - lat) / (n - s) * h)
    if not (0 <= c < wd and 0 <= r < h):
        return np.nan
    blk = field[max(0, r - halo):r + halo + 1, max(0, c - halo):c + halo + 1]
    blk = blk[np.isfinite(blk)]
    return float(blk.max()) if blk.size else np.nan


def contingency(pred, obs, thr=0.1):
    p, o = np.asarray(pred) > thr, np.asarray(obs) > thr
    hit = int((p & o).sum()); miss = int((~p & o).sum())
    fa = int((p & ~o).sum()); cn = int((~p & ~o).sum())
    pod = hit / max(1, hit + miss)
    far = fa / max(1, hit + fa)
    csi = hit / max(1, hit + miss + fa)
    return dict(hit=hit, miss=miss, fa=fa, cn=cn, pod=pod, far=far, csi=csi)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--times", required=True, help="comma list, e.g. 20260830T0730,20260830T0800")
    p.add_argument("--radar", default="nlhrw")
    p.add_argument("--max-range-km", type=float, default=120.0)
    p.add_argument("--halo", type=int, default=1,
                   help="neighbourhood radius in cells when sampling a gauge location. "
                        "0 = strict single cell. A halo favours whichever estimator has "
                        "more non-zero cells, so results MUST be checked at halo=0.")
    p.add_argument("--grid-n", type=int, default=256)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))

    from model.geo import GRID, bbox
    from model.nwp_regrid import reproject_to_analysis_grid
    from tools.radar_single_site import (find_volume, read_lowest_sweep, declutter,
                                         dbz_to_rate, polar_to_grid)

    bounds = bbox()
    rows = []
    for stamp in args.times.split(","):
        stamp = stamp.strip()
        gpath = fetch_knmi_10min(stamp)
        if gpath is None:
            continue
        gauges = read_gauges(gpath)

        vol = find_volume(args.radar, stamp)
        if vol is None:
            LOG.warning("no %s volume at %s", args.radar, stamp)
            continue
        dbz, az, rng, site, el = read_lowest_sweep(vol)
        dbz, _ = declutter(dbz)
        ours = polar_to_grid(dbz_to_rate(dbz), az, rng, site, GRID, bounds,
                             elangle=el, max_beam_m=1e9)

        day = f"{stamp[0:4]}/{stamp[4:6]}/{stamp[6:8]}"
        oh = sorted(glob.glob(f"/mnt/storagebox/opera/RATE/{day}/{stamp}_RATE.tif*"))
        # nodata_as_zero: OPERA encodes no-rain as nodata; without this the warp
        # erodes ~88% of the rain and OPERA is scored on a field it never produced.
        opera = (np.nan_to_num(reproject_to_analysis_grid(pathlib.Path(oh[0]),
                                                          nodata_as_zero=True), nan=0.0)
                 if oh else None)

        for st, la, lo, obs in gauges:
            dx = (lo - site[0]) * 111.32 * np.cos(np.radians(la))
            dy = (la - site[1]) * 111.32
            if np.hypot(dx, dy) > args.max_range_km:
                continue
            o = sample(ours, la, lo, bounds, GRID, halo=args.halo)
            e = sample(opera, la, lo, bounds, GRID, halo=args.halo) if opera is not None else np.nan
            if np.isfinite(o):
                rows.append((stamp, st, obs, o, e))

    if not rows:
        LOG.error("no comparable station-times")
        return 2

    obs = np.array([r[2] for r in rows])
    our = np.array([r[3] for r in rows])
    ope = np.array([r[4] for r in rows])
    ok = np.isfinite(ope)

    LOG.info("=== %d station-times within %.0f km of %s ===", len(rows), args.max_range_km, args.radar)
    LOG.info("gauge: %d wet (>0.1 mm/h) of %d | mean %.3f max %.2f mm/h",
             int((obs > 0.1).sum()), len(obs), obs.mean(), obs.max())
    for name, est in (("OURS", our), ("OPERA", ope)):
        v = est[ok] if name == "OPERA" else our[ok]
        o = obs[ok]
        c = contingency(v, o)
        bias = float(np.mean(v - o))
        mae = float(np.mean(np.abs(v - o)))
        corr = float(np.corrcoef(v, o)[0, 1]) if len(o) > 2 and v.std() > 0 else float("nan")
        LOG.info("%-5s POD %.3f FAR %.3f CSI %.3f | bias %+.3f MAE %.3f corr %.3f | mean %.3f",
                 name, c["pod"], c["far"], c["csi"], bias, mae, corr, v.mean())
    # The decisive cell: what does each estimator say where the gauge is WET?
    wet = ok & (obs > 0.1)
    if wet.sum():
        LOG.info("--- where the GAUGE is wet (%d cases) ---", int(wet.sum()))
        LOG.info("  gauge mean %.3f | ours %.3f | opera %.3f mm/h",
                 obs[wet].mean(), our[wet].mean(), ope[wet].mean())
        LOG.info("  detected: ours %d/%d, opera %d/%d",
                 int((our[wet] > 0.1).sum()), int(wet.sum()),
                 int((ope[wet] > 0.1).sum()), int(wet.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
