"""KNMI HARMONIE-AROME (UWCW, ~2 km) rainfall — high-res NWP anchor, BACKFILLABLE.

KNMI's Data Platform (KDP) serves the UWCW HARMONIE-AROME NL 2 km forecasts as
per-run NetCDF on a regular 0.05° grid, and — unlike ICON-D2/Open-Meteo open data
— **retains the archive back to 2024-08** (verified), so it pairs with the full
OPERA truth window. We pull the 1-hour rainfall-accumulation field per run.

KDP is a two-step API: GET …/files/<name>/url returns a temporary presigned URL,
then download that. Auth = the Open Data API key (env KNMI_OPEN_DATA_KEY).
NetCDF is regular lat/lon → the zarr builder's rasterio reproject reads it like
ERA5/Open-Meteo. One file per run: `harmonie_rain_<YYYYmmddTHH>.nc`.

    python collectors/fetch_harmonie.py --mode backfill --start 2024-08 --end now --out data/harmonie
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import sys
import time
import urllib.request

LOG = logging.getLogger("pluvio.fetch_harmonie")
KDP = "https://api.dataplatform.knmi.nl/open-data/v1"
DATASET, VERSION = "uwcw_extra_lv_ha43_nl_2km", "1.0"
VAR = "rainfall-accumulation-01h-hagl"
RUNS = (0, 3, 6, 9, 12, 15, 18, 21)  # UWCW HARMONIE runs every 3 h


def _fname(run: dt.datetime) -> str:
    return f"uwcw_ha43_bess_0p05deg_{VAR}_{run:%Y%m%dT%H}.nc"


def _download_url(key: str, filename: str) -> str | None:
    req = urllib.request.Request(
        f"{KDP}/datasets/{DATASET}/versions/{VERSION}/files/{filename}/url",
        headers={"Authorization": key})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("temporaryDownloadUrl")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def fetch_run(out_dir: pathlib.Path, key: str, run: dt.datetime) -> bool:
    target = out_dir / f"harmonie_rain_{run:%Y%m%dT%H}.nc"
    if target.exists() and target.stat().st_size > 0:
        return False
    url = _download_url(key, _fname(run))
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        tmp = target.with_suffix(".nc.part"); tmp.write_bytes(data); tmp.rename(target)
        LOG.info("wrote %s (%.1f MB)", target.name, len(data) / 1e6)
        return True
    except Exception as exc:  # noqa: BLE001
        LOG.warning("download %s failed (%s)", run, exc); return False


def _runs(start, end):
    def parse(s):
        if s == "now":
            return dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
        return dt.datetime.strptime(s, "%Y-%m").replace(tzinfo=dt.UTC)
    t, e = parse(start), parse(end)
    while t < e:
        if t.hour in RUNS:
            yield t
        t += dt.timedelta(hours=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["forward", "backfill"], default="forward")
    p.add_argument("--start", default="2024-08"); p.add_argument("--end", default="now")
    p.add_argument("--out", default="data/harmonie")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    key = os.environ.get("KNMI_OPEN_DATA_KEY", "")
    if not key:
        LOG.error("KNMI_OPEN_DATA_KEY not set"); return 2
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.mode == "forward":
        now = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
        for back in range(0, 12):  # scan back to the latest published run
            run = now - dt.timedelta(hours=back)
            if run.hour in RUNS and fetch_run(out, key, run):
                LOG.info("forward: got run %s", run); return 0
        LOG.warning("forward: no run fetched"); return 1

    n = 0
    for run in _runs(args.start, args.end):
        if fetch_run(out, key, run):
            n += 1
    LOG.info("backfill done: %d runs → %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
