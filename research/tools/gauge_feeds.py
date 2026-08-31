"""Hourly gauge accumulations from every open network in the serving domain.

One JSON per clock hour: [[lat, lon, mm, source], ...] — the input the operational
Appendix-B gauge adjustment consumes. Sources (all open, all proven fetchable):

    KNMI  NL  10-min `rg` rate [mm/h]      -> mm += rg / 6 over the hour's 6 files
    KMI   BE  aws_10min precip_quantity    -> sum of the hour's 10-min totals [mm]
    DWD   DE  10-min 'now' zips, RWS_10    -> sum of the hour's 6 totals [mm]
    EA    UK  15-min readings, value       -> sum of the hour's 4 totals [mm]

Every source is optional: whatever answers gets written, with per-source counts
logged — a missing network degrades coverage, not the run. The DWD leg downloads
~1400 tiny zips; they are cached and refreshed only when stale, so an hourly timer
costs one burst, not four.

Usage:
    python -m tools.gauge_feeds --hour 2026083118 --out /opt/pluvio/cache/gauges/2026083118.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import logging
import pathlib
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("pluvio.gauge_feeds")

DWD_BASE = ("https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
            "climate/10_minutes/precipitation/now/")
DWD_CACHE = pathlib.Path("/opt/pluvio/cache/dwd_gauges")
EA_BASE = "https://environment.data.gov.uk/flood-monitoring"
EA_CACHE = pathlib.Path("/opt/pluvio/cache/ea_gauges")


def hour_stamps(hour: str):
    t0 = dt.datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=dt.UTC)
    return t0, t0 + dt.timedelta(hours=1)


def knmi_hour(hour):
    from tools.gauge_validate import fetch_knmi_10min, read_gauges
    t0, _ = hour_stamps(hour)
    acc = {}
    for k in range(1, 7):                       # files are end-time labelled
        t = t0 + dt.timedelta(minutes=10 * k)
        p = fetch_knmi_10min(t.strftime("%Y%m%dT%H%M"))
        if p is None:
            continue
        for st, la, lo, rate in read_gauges(p):
            key = (round(la, 4), round(lo, 4))
            mm, n = acc.get(key, (0.0, 0))
            acc[key] = (mm + rate / 6.0, n + 1)
    return [[la, lo, mm, "knmi"] for (la, lo), (mm, n) in acc.items() if n >= 4]


def kmi_hour(hour):
    t0, t1 = hour_stamps(hour)
    url = ("https://opendata.meteo.be/service/aws/ows?service=WFS&version=2.0.0"
           "&request=GetFeature&typeName=aws:aws_10min"
           "&outputFormat=application/json&count=20000&CQL_FILTER="
           f"timestamp%3E%27{t0:%Y-%m-%dT%H:%M:%S}Z%27%20AND%20"
           f"timestamp%3C%3D%27{t1:%Y-%m-%dT%H:%M:%S}Z%27")
    try:
        feats = json.load(urllib.request.urlopen(url, timeout=240))["features"]
    except Exception as exc:
        LOG.warning("KMI unavailable (%s)", exc)
        return []
    acc = {}
    for x in feats:
        p = x["properties"]
        q = p.get("precip_quantity")
        c = (x.get("geometry") or {}).get("coordinates")
        if q is None or not c:
            continue
        key = (round(float(c[1]), 4), round(float(c[0]), 4))
        mm, n = acc.get(key, (0.0, 0))
        acc[key] = (mm + float(q), n + 1)
    return [[la, lo, mm, "kmi"] for (la, lo), (mm, n) in acc.items() if n >= 4]


def _dwd_stations():
    meta = DWD_CACHE / "stations.txt"
    if not meta.exists() or (dt.datetime.now(dt.UTC).timestamp()
                             - meta.stat().st_mtime) > 86400:
        DWD_CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            DWD_BASE + "zehn_now_rr_Beschreibung_Stationen.txt", meta)
    coords = {}
    for ln in meta.read_text(encoding="latin-1").splitlines()[2:]:
        p = ln.split()
        if len(p) >= 6:
            try:
                coords[p[0].zfill(5)] = (float(p[4]), float(p[5]))
            except ValueError:
                pass
    return coords


def dwd_hour(hour):
    coords = _dwd_stations()
    DWD_CACHE.mkdir(parents=True, exist_ok=True)
    # refresh stale zips in one parallel burst (curl handles the fan-out)
    stale = []
    now = dt.datetime.now(dt.UTC).timestamp()
    for sid in coords:
        z = DWD_CACHE / f"10minutenwerte_nieder_{sid}_now.zip"
        if not z.exists() or now - z.stat().st_mtime > 1800:
            stale.append(f"10minutenwerte_nieder_{sid}_now.zip")
    if stale:
        subprocess.run(
            ["xargs", "-P", "12", "-I", "{}", "sh", "-c",
             f"curl -sf -m 60 -o {DWD_CACHE}/{{}} {DWD_BASE}{{}} || true"],
            input="\n".join(stale).encode(), timeout=600)
    t0, t1 = hour_stamps(hour)
    lo_s = (t0 + dt.timedelta(minutes=10)).strftime("%Y%m%d%H%M")
    hi_s = t1.strftime("%Y%m%d%H%M")
    out = []
    for sid, (la, lo) in coords.items():
        z = DWD_CACHE / f"10minutenwerte_nieder_{sid}_now.zip"
        if not z.exists():
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                txt = zf.read(zf.namelist()[0]).decode("latin-1")
        except Exception:
            continue
        mm, n = 0.0, 0
        for row in csv.DictReader(io.StringIO(txt), delimiter=";"):
            s = row.get("MESS_DATUM", "").strip()
            if lo_s <= s <= hi_s:
                try:
                    v = float(row["RWS_10"])
                except (KeyError, ValueError):
                    continue
                if v >= 0:
                    mm += v
                    n += 1
        if n >= 4:
            out.append([la, lo, mm, "dwd"])
    return out


def _ea_stations():
    meta = EA_CACHE / "stations.json"
    if not meta.exists() or (dt.datetime.now(dt.UTC).timestamp()
                             - meta.stat().st_mtime) > 86400:
        EA_CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            EA_BASE + "/id/stations?parameter=rainfall&_limit=2000", meta)
    items = json.load(open(meta))["items"]
    return {s["notation"]: (float(s["lat"]), float(s["long"]))
            for s in items if s.get("lat") and s.get("long")}


def ea_hour(hour):
    coords = _ea_stations()
    t0, t1 = hour_stamps(hour)
    EA_CACHE.mkdir(parents=True, exist_ok=True)
    ids = list(coords)
    script = (f'curl -sf -m 60 "{EA_BASE}/id/stations/{{}}/readings.csv'
              f'?parameter=rainfall&since={t0:%Y-%m-%dT%H:%M:%S}Z&_limit=12" '
              f'-o {EA_CACHE}/{{}}.csv || true')
    subprocess.run(["xargs", "-P", "12", "-I", "{}", "sh", "-c", script],
                   input="\n".join(ids).encode(), timeout=900)
    out = []
    for sid, (la, lo) in coords.items():
        fp = EA_CACHE / f"{sid}.csv"
        if not fp.exists():
            continue
        mm, n = 0.0, 0
        try:
            for row in csv.DictReader(open(fp)):
                try:
                    te = dt.datetime.strptime(row["dateTime"],
                                              "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
                    v = float(row["value"])
                except (KeyError, ValueError, TypeError):
                    continue
                if t0 < te <= t1 and v >= 0:
                    mm += v
                    n += 1
        except Exception:
            continue
        if n >= 3:
            out.append([la, lo, mm, "ea"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hour", required=True, help="YYYYmmddHH (UTC clock hour)")
    p.add_argument("--out", required=True)
    p.add_argument("--sources", default="knmi,kmi,dwd,ea")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fns = {"knmi": knmi_hour, "kmi": kmi_hour, "dwd": dwd_hour, "ea": ea_hour}
    rows = []
    for src in args.sources.split(","):
        try:
            got = fns[src](args.hour)
        except Exception as exc:
            LOG.warning("%s failed (%s)", src, exc)
            got = []
        LOG.info("%s: %d gauges", src, len(got))
        rows.extend(got)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(rows))
    tmp.replace(out)
    LOG.info("wrote %s: %d gauges total", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
