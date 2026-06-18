"""DWD ICON-D2 (~2.2 km) total precipitation — free high-resolution NWP anchor.

DWD publishes ICON-D2 GRIB2 openly (no key) at opendata.dwd.de, including a
**regular-lat-lon** variant that rasterio/GDAL reads directly. We pull `tot_prec`
(accumulated mm from run start; differenced at build time, like AIFS) for the
latest run. The open-data mirror is rolling (~24–48 h) so this is **forward-only**
— it accumulates a high-res NWP history from now (the free alternative to
Open-Meteo's paywalled historical archive).

Output: one GRIB2 per step, `icon_d2_tot_prec_<YYYYmmddTHHZ>_+<step>h.grib2`.

    python collectors/fetch_icon_d2.py --max-step 24 --out data/icon_d2
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import logging
import pathlib
import sys
import urllib.request

LOG = logging.getLogger("pluvio.fetch_icon_d2")
BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
RUNS = (0, 3, 6, 9, 12, 15, 18, 21)  # ICON-D2 runs every 3 h


def _url(run: dt.datetime, step: int) -> str:
    hh = f"{run.hour:02d}"
    stamp = run.strftime("%Y%m%d") + hh
    fn = f"icon-d2_germany_regular-lat-lon_single-level_{stamp}_{step:03d}_2d_tot_prec.grib2.bz2"
    return f"{BASE}/{hh}/tot_prec/{fn}"


def _exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:
        return False


def latest_run(now: dt.datetime) -> dt.datetime | None:
    """Most recent ICON-D2 run whose step-0 tot_prec is published (publication
    lags the run by ~2–3 h). Scans back up to 4 runs (~12 h)."""
    cand = now.replace(minute=0, second=0, microsecond=0)
    for _ in range(8):
        if cand.hour in RUNS and _exists(_url(cand, 0)):
            return cand
        cand -= dt.timedelta(hours=1)
    return None


def fetch_run(out_dir: pathlib.Path, run: dt.datetime, steps) -> int:
    n = 0
    for step in steps:
        target = out_dir / f"icon_d2_tot_prec_{run.strftime('%Y%m%dT%HZ')}_+{step}h.grib2"
        if target.exists() and target.stat().st_size > 0:
            n += 1
            continue
        url = _url(run, step)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = bz2.decompress(r.read())
            tmp = target.with_suffix(".grib2.part")
            tmp.write_bytes(data)
            tmp.rename(target)
            n += 1
        except Exception as exc:  # noqa: BLE001 — one bad step shouldn't kill the run
            LOG.warning("step +%dh failed (%s)", step, exc)
            target.unlink(missing_ok=True)
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-step", type=int, default=24, help="forecast horizon, hours (ICON-D2 ≤48)")
    p.add_argument("--step", type=int, default=1, help="step cadence, hours")
    p.add_argument("--out", default="data/icon_d2")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = latest_run(dt.datetime.now(dt.UTC))
    if run is None:
        LOG.error("no published ICON-D2 run found"); return 1
    steps = list(range(0, args.max_step + 1, args.step))
    n = fetch_run(out_dir, run, steps)
    LOG.info("done: %d/%d steps for ICON-D2 run %s → %s", n, len(steps), run.strftime("%Y%m%dT%HZ"), out_dir)
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
