"""Durably archive every forecast/nowcast run — verification needs history.

The serving npz holds only the LATEST issue; without this, "how good was the
nowcast we published at 14:05?" is unanswerable a day later. Copies each new
issue (deduped by issue_epoch) into a date-partitioned tree, rates downcast to
f16 (compression ~2x, mm/h precision unaffected for verification).

Usage: python -m tools.forecast_archive [--out-root /mnt/storagebox/forecast_archive]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys

import numpy as np

LOG = logging.getLogger("pluvio.forecast_archive")

SOURCES = {
    "forecast": "/opt/pluvio/serve/model_forecast.npz",
    "nowcast": "/opt/pluvio/serve/model_nowcast.npz",
    # 4.1 side path (composite-driven 5-min issues); archived so the nightly
    # scoreboard can compare it with the served nowcast on real serving data.
    "lowlatency": "/opt/pluvio/serve/lowlatency_nowcast.npz",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="/mnt/storagebox/forecast_archive")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for kind, src in SOURCES.items():
        sp = pathlib.Path(src)
        if not sp.exists():
            continue
        try:
            z = np.load(sp, allow_pickle=False)
            issue = int(z["issue_epoch"])
        except Exception as exc:
            LOG.warning("%s unreadable (%s)", src, exc)
            continue
        ts = dt.datetime.fromtimestamp(issue, dt.UTC)
        day_dir = pathlib.Path(args.out_root) / f"{ts:%Y/%m/%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        dest = day_dir / f"{kind}_{ts:%H%M}.npz"
        if dest.exists():
            continue
        payload = {k: z[k] for k in z.files}
        payload["rates"] = payload["rates"].astype("float16")
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp.replace(dest)
        LOG.info("archived %s issue %s", kind, ts.isoformat())
    return 0


if __name__ == "__main__":
    sys.exit(main())
