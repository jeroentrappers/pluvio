"""Gauge-calibrate the Z–R relation, with an honest train/test split.

Marshall-Palmer (Z = 200 R^1.6) is a generic stratiform pair, not a calibration for
these radars. Measured against KNMI gauges over 599 station-times, our estimate
under-reads badly: gauge mean 1.744 mm/h where the radar says 0.431. Operational QPE
closes that with gauge adjustment; this does the same, fitting a power correction

    R_cal = c * R_raw ** d

in log space on gauge/radar pairs. A pure multiplicative factor (d = 1) is the usual
first move, but rain-rate errors are typically multiplicative AND intensity-dependent
— light drizzle and convective cores are biased differently — so the exponent is
fitted too and reported alongside, letting us see whether d ≈ 1 (pure scale) or not.

⚠️ Fitting and evaluating on the same data would manufacture a win. The split is by
TIME, not at random: consecutive station-times are strongly correlated (same weather
system over the same stations), so a random split would leak the test set into the
fit and flatter the result. Early slots train, later slots test.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.zr_calibrate")


def collect(times, radar, halo, max_range_km):
    """Gather (gauge, ours_raw, opera) triples across the given times."""
    from model.geo import GRID, bbox
    from model.nwp_regrid import reproject_to_analysis_grid
    from tools.gauge_validate import fetch_knmi_10min, read_gauges, sample
    from tools.radar_single_site import (find_volume, read_lowest_sweep, declutter,
                                         dbz_to_rate, polar_to_grid)

    bounds = bbox()
    rows = []
    for stamp in times:
        gp = fetch_knmi_10min(stamp)
        vol = find_volume(radar, stamp)
        if gp is None or vol is None:
            continue
        dbz, az, rng, site, el = read_lowest_sweep(vol)
        dbz, _ = declutter(dbz)
        ours = polar_to_grid(dbz_to_rate(dbz), az, rng, site, GRID, bounds,
                             elangle=el, max_beam_m=1e9)
        day = f"{stamp[0:4]}/{stamp[4:6]}/{stamp[6:8]}"
        oh = sorted(glob.glob(f"/mnt/storagebox/opera/RATE/{day}/{stamp}_RATE.tif*"))
        opera = (np.nan_to_num(reproject_to_analysis_grid(pathlib.Path(oh[0]),
                                                          nodata_as_zero=True), nan=0.0)
                 if oh else None)
        for st, la, lo, obs in read_gauges(gp):
            dx = (lo - site[0]) * 111.32 * np.cos(np.radians(la))
            dy = (la - site[1]) * 111.32
            if np.hypot(dx, dy) > max_range_km:
                continue
            o = sample(ours, la, lo, bounds, GRID, halo=halo)
            e = sample(opera, la, lo, bounds, GRID, halo=halo) if opera is not None else np.nan
            if np.isfinite(o):
                rows.append((stamp, obs, o, e))
    return rows


def fit_power(gauge, radar_raw, min_rate=0.1):
    """Least-squares fit of log R_gauge = log c + d log R_radar on mutually wet pairs.

    Restricted to pairs where BOTH exceed min_rate: zeros cannot be logged, and
    including near-zero radar values would let noise dominate the slope.
    """
    m = (gauge > min_rate) & (radar_raw > min_rate)
    if m.sum() < 10:
        return None
    x = np.log(radar_raw[m]); y = np.log(gauge[m])
    d, logc = np.polyfit(x, y, 1)
    return float(np.exp(logc)), float(d), int(m.sum())


def scores(pred, obs, thr=0.1):
    p, o = pred > thr, obs > thr
    hit = int((p & o).sum()); miss = int((~p & o).sum()); fa = int((p & ~o).sum())
    return dict(
        pod=hit / max(1, hit + miss), far=fa / max(1, hit + fa),
        csi=hit / max(1, hit + miss + fa),
        bias=float(np.mean(pred - obs)), mae=float(np.mean(np.abs(pred - obs))),
        corr=float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 0 else float("nan"),
        wet_mean=float(pred[o].mean()) if o.any() else 0.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--times", required=True)
    p.add_argument("--radar", default="nlhrw")
    p.add_argument("--halo", type=int, default=1)
    p.add_argument("--max-range-km", type=float, default=120.0)
    p.add_argument("--grid-n", type=int, default=256)
    p.add_argument("--train-frac", type=float, default=0.5)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("PLUVIO_GRID_N", str(args.grid_n))

    times = [t.strip() for t in args.times.split(",") if t.strip()]
    rows = collect(times, args.radar, args.halo, args.max_range_km)
    if len(rows) < 50:
        LOG.error("only %d pairs — too few", len(rows))
        return 2

    stamps = sorted({r[0] for r in rows})
    cut = stamps[int(len(stamps) * args.train_frac)]
    tr = [r for r in rows if r[0] < cut]
    te = [r for r in rows if r[0] >= cut]
    LOG.info("%d pairs | train %d (< %s) | test %d", len(rows), len(tr), cut, len(te))

    g_tr = np.array([r[1] for r in tr]); o_tr = np.array([r[2] for r in tr])
    fit = fit_power(g_tr, o_tr)
    if fit is None:
        LOG.error("not enough mutually-wet training pairs to fit")
        return 3
    c, d, n = fit
    LOG.info("fitted on %d mutually-wet train pairs:  R_cal = %.3f * R_raw ** %.3f", n, c, d)
    LOG.info("  (d≈1 would mean a pure scale factor; d=%.2f means intensity-dependent)", d)

    g = np.array([r[1] for r in te]); o = np.array([r[2] for r in te]); e = np.array([r[3] for r in te])
    ok = np.isfinite(e)
    g, o, e = g[ok], o[ok], e[ok]
    cal = c * np.power(np.clip(o, 1e-6, None), d)
    cal[o <= 0] = 0.0

    # Mean-field bias: the classic operational correction, a single multiplicative
    # factor with the exponent pinned at 1. With few mutually-wet pairs the free
    # exponent above collapses toward a constant (measured d=0.28 on 36 pairs, which
    # raised POD to 1.0 but FAR to 0.75 and HALVED held-out CSI). Fixing d=1 keeps the
    # field's dynamic range and only removes the overall scale error.
    mwet = (g_tr > 0.1) & (o_tr > 0.1)
    mfb = float(g_tr[mwet].sum() / o_tr[mwet].sum()) if mwet.sum() >= 5 else 1.0
    LOG.info("mean-field bias factor (d fixed at 1): %.3f from %d pairs", mfb, int(mwet.sum()))
    cal_mfb = o * mfb

    LOG.info("=== HELD-OUT TEST: %d pairs, %d wet ===", len(g), int((g > 0.1).sum()))
    LOG.info("gauge mean %.3f (wet mean %.3f)", g.mean(), g[g > 0.1].mean() if (g > 0.1).any() else 0)
    for name, v in (("ours raw", o), ("ours cal(power)", cal),
                    ("ours cal(scale)", cal_mfb), ("OPERA", e)):
        s = scores(v, g)
        LOG.info("%-16s POD %.3f FAR %.3f CSI %.3f | bias %+.3f MAE %.3f corr %.3f | mean@wet %.3f",
                 name, s["pod"], s["far"], s["csi"], s["bias"], s["mae"], s["corr"], s["wet_mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
