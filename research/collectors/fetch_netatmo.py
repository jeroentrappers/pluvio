"""Netatmo crowdsourced surface observations (public weather stations).

Netatmo's `Getpublicdata` returns every *public* personal weather station in a
lat/lon region — temperature, humidity, pressure, and (on some) rain/wind.
Across Benelux that's thousands of stations, ~20x the density of the official
KMI/KNMI AWS networks. We poll a tiled bbox, take the latest reading per
station, and append to a Parquet whose schema matches the AWS collectors so
`build_zarr` folds them into the same IDW surface channels.

Auth: OAuth2 refresh-token grant. The durable credential is
``NETATMO_REFRESH_TOKEN`` (from the app's Token Generator, scope read_station);
access tokens last ~3h and are minted per run. If Netatmo rotates the refresh
token, we write the new one back to .env.

Forward-only: Getpublicdata has no historical archive, so this accumulates from
first run (like ALARO / no backfill into the 2024-2026 training window).

Output schema (one row per station per poll), aligned with fetch_kmi_aws.py:
    timestamp   datetime UTC   (poll time, floored to 10 min — one frame/poll)
    lat, lon    float
    station_id  str
    pressure                  hPa, sea-level (Netatmo reports calibrated)
    temp_dry_shelter_avg      °C
    humidity_rel_shelter_avg  %
    wind_speed_10m            m/s   (Netatmo gives km/h; converted)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

LOG = logging.getLogger("pluvio.fetch_netatmo")

TOKEN_URL = "https://api.netatmo.com/oauth2/token"
PUBLICDATA_URL = "https://api.netatmo.com/api/getpublicdata"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


# ───────────────────────────────────────────────────────────── env / auth

def _read_env() -> dict[str, str]:
    """Parse .env ourselves — values contain '|', which breaks shell sourcing."""
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _persist_refresh_token(new_token: str) -> None:
    """Rewrite NETATMO_REFRESH_TOKEN in .env if Netatmo rotated it."""
    lines = ENV_PATH.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("NETATMO_REFRESH_TOKEN="):
            lines[i] = f"NETATMO_REFRESH_TOKEN={new_token}"
            break
    ENV_PATH.write_text("\n".join(lines) + "\n")
    LOG.info("Netatmo rotated the refresh token; updated .env")


def get_access_token(client: httpx.Client, env: dict[str, str]) -> str:
    cid, sec, rt = (env.get("NETATMO_CLIENT_ID"), env.get("NETATMO_CLIENT_SECRET"),
                    env.get("NETATMO_REFRESH_TOKEN"))
    if not all((cid, sec, rt)):
        raise SystemExit("NETATMO_CLIENT_ID / _CLIENT_SECRET / _REFRESH_TOKEN must be set in .env")
    r = client.post(TOKEN_URL, data={
        "grant_type": "refresh_token", "refresh_token": rt,
        "client_id": cid, "client_secret": sec}, timeout=30)
    r.raise_for_status()
    tok = r.json()
    new_rt = tok.get("refresh_token")
    if new_rt and new_rt != rt:
        _persist_refresh_token(new_rt)
    return tok["access_token"]


# ───────────────────────────────────────────────────────── public-data parse

def _tiles(bbox: tuple[float, float, float, float], step: float):
    """Yield (lat_sw, lon_sw, lat_ne, lon_ne) sub-boxes covering bbox."""
    minx, miny, maxx, maxy = bbox  # lon_min, lat_min, lon_max, lat_max
    lat = miny
    while lat < maxy:
        lon = minx
        while lon < maxx:
            yield (lat, lon, min(lat + step, maxy), min(lon + step, maxx))
            lon += step
        lat += step


# Plausibility gates — crowdsourced sensors are noisy (miscalibrated pressure,
# dead RH sensors reading 0). Out-of-range values become null so a single bad
# station can't skew the local IDW; the rest of that station's fields survive.
_QC = {
    "temp_dry_shelter_avg": (-40.0, 50.0),
    "humidity_rel_shelter_avg": (1.0, 100.0),   # 0% ⇒ dead sensor
    "pressure": (870.0, 1085.0),                 # generous sea-level band; drops e.g. 747
    "wind_speed_10m": (0.0, 60.0),
}


def _qc(field: str, v):
    if v is None:
        return None
    lo, hi = _QC[field]
    return v if lo <= v <= hi else None


def _extract_station(s: dict) -> dict | None:
    """Pull lat/lon + latest T/RH/P/wind out of one Getpublicdata station."""
    place = s.get("place", {})
    loc = place.get("location")
    if not loc:
        return None
    row = {"station_id": s.get("_id"), "lon": float(loc[0]), "lat": float(loc[1]),
           "pressure": None, "temp_dry_shelter_avg": None,
           "humidity_rel_shelter_avg": None, "wind_speed_10m": None}
    for _mac, m in s.get("measures", {}).items():
        # Temperature/humidity/pressure modules: {"res": {ts: [vals...]}, "type": [...]}
        if "type" in m and "res" in m and m["res"]:
            types = m["type"]
            latest_ts = max(m["res"].keys(), key=int)
            vals = m["res"][latest_ts]
            for t, v in zip(types, vals):
                if t == "temperature":
                    row["temp_dry_shelter_avg"] = v
                elif t == "humidity":
                    row["humidity_rel_shelter_avg"] = v
                elif t == "pressure":
                    row["pressure"] = v
        # Wind module: flat keys, km/h → m/s
        if "wind_strength" in m and m["wind_strength"] is not None:
            row["wind_speed_10m"] = m["wind_strength"] / 3.6
    for f in _QC:
        row[f] = _qc(f, row[f])
    # keep only stations with at least one usable field after QC
    if all(row[k] is None for k in
           ("pressure", "temp_dry_shelter_avg", "humidity_rel_shelter_avg", "wind_speed_10m")):
        return None
    return row


def fetch_region(client: httpx.Client, token: str,
                 tile: tuple[float, float, float, float]) -> list[dict]:
    lat_sw, lon_sw, lat_ne, lon_ne = tile
    r = client.get(PUBLICDATA_URL, params={
        "lat_ne": lat_ne, "lon_ne": lon_ne, "lat_sw": lat_sw, "lon_sw": lon_sw,
        "filter": "true"}, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code != 200:
        LOG.warning("  tile %s: HTTP %s %s", tile, r.status_code, r.text[:120])
        return []
    out = []
    for s in r.json().get("body", []):
        row = _extract_station(s)
        if row:
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/aws/netatmo_10min.parquet")
    parser.add_argument(
        "--bbox", default="2.0,49.0,8.0,54.0",
        help="minx,miny,maxx,maxy EPSG:4326. Default = Benelux + borders, the "
             "populated core of the analysis grid.")
    parser.add_argument("--tile-deg", type=float, default=1.0,
                        help="Tile size in degrees (Netatmo caps results per call).")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Pause between tile calls to respect rate limits.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    bbox = tuple(float(x) for x in args.bbox.split(","))
    env = _read_env()
    # One frame per poll: floor poll time to 10 min so build_zarr groups them.
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    stamp = now.replace(minute=(now.minute // 10) * 10)

    with httpx.Client(http2=True) as client:
        token = get_access_token(client, env)
        by_station: dict[str, dict] = {}
        tiles = list(_tiles(bbox, args.tile_deg))
        for i, tile in enumerate(tiles):
            for row in fetch_region(client, token, tile):
                by_station[row["station_id"]] = row  # dedup across overlapping tiles
            if args.sleep:
                time.sleep(args.sleep)
        LOG.info("Collected %d unique stations across %d tiles", len(by_station), len(tiles))

    if not by_station:
        LOG.warning("No stations returned — nothing written.")
        return 1

    df = pd.DataFrame(list(by_station.values()))
    df["timestamp"] = pd.Timestamp(stamp).tz_localize(None)
    df = df[["timestamp", "station_id", "lat", "lon", "pressure",
             "temp_dry_shelter_avg", "humidity_rel_shelter_avg", "wind_speed_10m"]]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp", "station_id"], keep="last")
    df.to_parquet(out, index=False)
    LOG.info("Wrote %d rows (%d this poll @ %s) → %s",
             len(df), len(by_station), stamp.isoformat(), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
