"""Hourly QC watchdog over the served composite — every defect class we have met,
measured continuously per region instead of chased per incident.

Every check below is the exact signature that caught a real bug this week:

  churn        median wet-cell flip % between consecutive SCANS, and interpolants
               relative to scans (flow mis-scaling, morph defects read as
               interp >> scan; the a1gate rotation read as churn at ALL intensities)
  parity       alternation index of the wet-area series at scan cadence: |lag-1
               autocorrelation| when negative — alternating scan programs and
               cadence mismatches (DK 5-min programs, UKMO 15-min slots) show as
               strong negative lag-1 (area ping-pong)
  freeze       fraction of consecutive scan pairs with near-identical fields —
               a stuck source (UKMO slot reuse, wedged feed) shows as freezing
  staleness    age of the newest frame vs wall clock
  gauge_bias   served rate vs the hourly gauge JSONs the adjustment already
               collects (KNMI/KMI/DWD/EA) — level drift per region

Output: one JSON (region -> metrics + verdicts) written atomically for anything to
scrape, WARN lines in the journal for thresholds crossed, exit code 1 if any region
is red (so the systemd unit shows failed and can be alerted on).

Usage: PYTHONPATH=... python -m tools.qc_watchdog [--npz PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.qc")

REGIONS = {
    "BE":      (2.5, 49.5, 6.5, 51.5),
    "NL":      (3.3, 50.7, 7.3, 53.7),
    "DE-west": (6.0, 47.5, 12.0, 55.0),
    "DE-east": (12.0, 49.0, 15.0, 54.5),
    "DK":      (8.0, 54.5, 13.0, 57.8),
    "Oresund": (12.3, 54.9, 14.8, 56.6),
    "PL":      (14.0, 49.0, 24.0, 55.0),
    "IE":      (-10.5, 51.3, -5.5, 55.5),
    "UK":      (-8.0, 50.0, 2.0, 59.0),
    "FR-nord": (-2.0, 46.5, 6.0, 49.5),
}

# Thresholds: set from this week's measured healthy/broken values.
CHURN_SCAN_WARN = 55.0        # a1gate-class scrambling measured 93-98%
INTERP_RATIO_WARN = 1.3       # healthy morph measured 0.7-0.9x of scan churn
PARITY_WARN = -0.45           # Oresund ping-pong measured ~-0.8; healthy ~0
FREEZE_WARN = 0.34            # UKMO slot reuse froze 2 of every 3 pairs (0.67)
STALE_WARN_S = 1500
GAUGE_BIAS_WARN = 5.0         # mm/h at wet gauges; DE convective bug measured +7-8


def region_metrics(rates, times, bounds, box):
    W, S, E, N = bounds
    h, w = rates.shape[1:]
    c0, c1 = int((box[0] - W) / (E - W) * w), int((box[2] - W) / (E - W) * w)
    r0, r1 = int((N - box[3]) / (N - S) * h), int((N - box[1]) / (N - S) * h)
    sub = np.nan_to_num(rates[:, max(0, r0):r1, max(0, c0):c1])
    if sub.size == 0:
        return None
    wet = sub > 0.3
    sc = [i for i in range(len(times)) if times[i] % 300 == 0]

    flips_s, flips_i, frozen = [], [], 0
    for i in range(1, len(times)):
        u = (wet[i - 1] | wet[i]).sum()
        fl = 100.0 * np.logical_xor(wet[i - 1], wet[i]).sum() / max(u, 1)
        (flips_s if times[i] % 300 == 0 else flips_i).append(fl)
    for a, b in zip(sc[:-1], sc[1:]):
        u = (wet[a] | wet[b]).sum()
        if u > 30 and np.logical_xor(wet[a], wet[b]).sum() / u < 0.02:
            frozen += 1
    area = np.array([float(wet[i].mean()) for i in sc])
    d = np.diff(area)
    parity = 0.0
    if len(d) > 4 and d.std() > 1e-9:
        parity = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    wet_enough = float(np.mean(area)) > 0.001
    return {
        "wet_area_mean_pct": round(100 * float(np.mean(area)), 3),
        "churn_scan_pct": round(float(np.median(flips_s)), 1) if flips_s else None,
        "churn_interp_ratio": (round(float(np.median(flips_i)) / max(float(np.median(flips_s)), 1e-6), 2)
                               if flips_i and flips_s else None),
        "parity_lag1": round(parity, 2),
        "freeze_frac": round(frozen / max(len(sc) - 1, 1), 2),
        "assessable": wet_enough,
    }


def gauge_bias(rates, times, bounds, box, gauge_dir):
    """Served mean rate vs gauge mm over the newest fully-covered clock hour."""
    hours = sorted(glob.glob(str(pathlib.Path(gauge_dir) / "*.json")))
    if not hours:
        return None
    hour = pathlib.Path(hours[-1]).stem
    h0 = dt.datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=dt.UTC).timestamp()
    idx = [i for i, t in enumerate(times) if h0 < t <= h0 + 3600 and t % 300 == 0]
    if len(idx) < 8:
        return None
    W, S, E, N = bounds
    h, w = rates.shape[1:]
    rows = json.loads(pathlib.Path(hours[-1]).read_text())
    diffs = []
    for la, lo, mm, _src in rows:
        if not (box[0] <= lo <= box[2] and box[1] <= la <= box[3]) or mm <= 0.25:
            continue
        c = int((lo - W) / (E - W) * w)
        r = int((N - la) / (N - S) * h)
        if not (0 <= r < h and 0 <= c < w):
            continue
        ours_mm = float(np.nansum([rates[i, r, c] / 12.0 for i in idx]))
        diffs.append(ours_mm - mm)
    if len(diffs) < 5:
        return None
    return round(float(np.mean(diffs)), 2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="/opt/pluvio/serve/observed.npz")
    p.add_argument("--gauge-dir", default="/opt/pluvio/cache/gauges")
    p.add_argument("--out", default="/opt/pluvio/serve/qc_status.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    z = np.load(args.npz)
    times = z["times"].astype("int64")
    rates = z["rates"].astype("float32")
    bounds = tuple(float(x) for x in z["bounds"])

    stale_s = int(dt.datetime.now(dt.UTC).timestamp() - int(times[-1]))
    out = {"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "staleness_s": stale_s, "regions": {}}
    red = stale_s > STALE_WARN_S
    if stale_s > STALE_WARN_S:
        LOG.warning("STALE: newest frame %d s old", stale_s)

    for name, box in REGIONS.items():
        m = region_metrics(rates, times, bounds, box)
        if m is None:
            continue
        gb = gauge_bias(rates, times, bounds, box, args.gauge_dir)
        m["gauge_bias_mm"] = gb
        verdicts = []
        if m["assessable"]:
            if m["churn_scan_pct"] and m["churn_scan_pct"] > CHURN_SCAN_WARN:
                verdicts.append("CHURN")
            if m["churn_interp_ratio"] and m["churn_interp_ratio"] > INTERP_RATIO_WARN:
                verdicts.append("INTERP")
            if m["parity_lag1"] < PARITY_WARN:
                verdicts.append("PARITY-PULSE")
            if m["freeze_frac"] > FREEZE_WARN:
                verdicts.append("FREEZE")
        if gb is not None and abs(gb) > GAUGE_BIAS_WARN:
            verdicts.append("GAUGE-BIAS")
        m["warnings"] = verdicts
        out["regions"][name] = m
        if verdicts:
            red = True
            LOG.warning("%s: %s  %s", name, ",".join(verdicts),
                        json.dumps({k: v for k, v in m.items() if k != "warnings"}))
        else:
            LOG.info("%s ok", name)

    op = pathlib.Path(args.out)
    tmp = op.with_name(op.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(op)
    LOG.info("wrote %s", op)
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
