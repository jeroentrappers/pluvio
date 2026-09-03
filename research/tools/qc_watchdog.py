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

The check math (region_metrics, gauge_bias, evaluate_region) lives in
tools/qc/checks.py, unchanged in behaviour from the original inline version;
this file is the CLI: loads the npz, drives the checks per region, and
writes the legacy-shaped JSON below so the systemd unit and anything
scraping it keep working.

Output: one JSON (region -> metrics + verdicts) written atomically for anything to
scrape, WARN lines in the journal for thresholds crossed, exit code 1 if any region
is red (so the systemd unit shows failed and can be alerted on).

Usage: PYTHONPATH=... python -m tools.qc_watchdog [--npz PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.qc import checks
from tools.qc.thresholds import load_thresholds
from tools.qc.verdict import write_atomic

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="/opt/pluvio/serve/observed.npz")
    p.add_argument("--gauge-dir", default="/opt/pluvio/cache/gauges")
    p.add_argument("--out", default="/opt/pluvio/serve/qc_status.json")
    p.add_argument("--thresholds", default=None,
                    help="path to a thresholds YAML/JSON file "
                         "(default: $PLUVIO_QC_THRESHOLDS or built-in defaults)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    thresholds = load_thresholds(args.thresholds)

    z = np.load(args.npz)
    times = z["times"].astype("int64")
    rates = z["rates"].astype("float32")
    bounds = tuple(float(x) for x in z["bounds"])

    stale_s = int(dt.datetime.now(dt.UTC).timestamp() - int(times[-1]))
    out = {"generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "staleness_s": stale_s, "regions": {}}
    red = stale_s > thresholds.stale_warn_s
    if stale_s > thresholds.stale_warn_s:
        LOG.warning("STALE: newest frame %d s old", stale_s)

    for name, box in REGIONS.items():
        m = checks.region_metrics(rates, times, bounds, box)
        if m is None:
            continue
        gb = checks.gauge_bias(rates, times, bounds, box, args.gauge_dir)
        m["gauge_bias_mm"] = gb
        verdicts = checks.evaluate_region(m, gb, thresholds)
        m["warnings"] = verdicts
        out["regions"][name] = m
        if verdicts:
            red = True
            LOG.warning("%s: %s  %s", name, ",".join(verdicts),
                        json.dumps({k: v for k, v in m.items() if k != "warnings"}))
        else:
            LOG.info("%s ok", name)

    op = write_atomic(args.out, out)
    LOG.info("wrote %s", op)
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
