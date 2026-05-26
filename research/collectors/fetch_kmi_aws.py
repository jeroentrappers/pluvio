"""KMI Automatic Weather Station (AWS) 10-minute observations.

WFS endpoint:   https://opendata.meteo.be/service/aws/wfs
Feature type:   aws:aws_10min
Cadence:        10 minutes
Coverage:       ~30 KMI stations across Belgium

Fields kept (subset of the full schema — extend as needed):
  code                       int      station identifier
  the_geom                   point    longitude / latitude
  timestamp                  datetime UTC
  pressure                   decimal  hPa, mean-sea-level corrected
  temp_dry_shelter_avg       decimal  °C  — air temperature at 1.5 m
  humidity_rel_shelter_avg   decimal  %   — relative humidity at 1.5 m
  wind_speed_10m             decimal  m/s — 10-m wind speed
  wind_direction             decimal  ° from north
  wind_gusts_speed           decimal  m/s — peak gust in the 10-min interval
  precip_quantity            decimal  mm  — 10-min accumulation
  temp_grass_pt100_avg       decimal  °C  — grass-level temperature (frost indicator)

Output: a single Parquet partitioned by date, appended on every run. One row
per (timestamp, station_code). Tidy enough to read with `pandas.read_parquet`
or `duckdb.sql("SELECT * FROM read_parquet('aws_10min.parquet')")`.

WFS chunks responses to ~100 features per page, so we paginate via
startIndex+count. The geoserver exposes a JSON-encoded GetFeature endpoint
that's much easier to parse than GML.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

LOG = logging.getLogger("pluvio.fetch_kmi_aws")

WFS_URL = "https://opendata.meteo.be/service/aws/wfs"
USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"

WANTED_FIELDS = (
    "code",
    "timestamp",
    "pressure",
    "temp_dry_shelter_avg",
    "humidity_rel_shelter_avg",
    "wind_speed_10m",
    "wind_direction",
    "wind_gusts_speed",
    "precip_quantity",
    "temp_grass_pt100_avg",
)


def fetch_all(
    client: httpx.Client,
    count: int,
    cql_filter: str | None,
) -> dict:
    """Fetch up to ``count`` features in a single request.

    Two GeoServer quirks influence the URL we hand-build:
      1. The CQL parser breaks if `:` inside ISO timestamp literals is
         percent-encoded; we keep `:` unencoded.
      2. ``startIndex`` + ``cql_filter`` returns 400 (GeoServer bug). Since
         one day of BE AWS data is ~4k features and a month ~130k, a single
         request is enough — we bump ``count`` instead of paging.
    """
    import urllib.parse

    parts = [
        "service=WFS",
        "version=2.0.0",
        "request=GetFeature",
        "typenames=aws:aws_10min",
        "outputFormat=application/json",
        f"count={count}",
    ]
    if cql_filter:
        parts.append("cql_filter=" + urllib.parse.quote(cql_filter, safe=":"))
    url = WFS_URL + "?" + "&".join(parts)
    r = client.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def flatten(feature: dict) -> dict:
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates") if geom else (None, None)
    props = feature.get("properties", {})
    row = {
        "lon": coords[0] if coords else None,
        "lat": coords[1] if coords else None,
    }
    for key in WANTED_FIELDS:
        row[key] = props.get(key)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Earliest observation timestamp to pull (UTC ISO). Defaults to the "
            "last 24 hours so a daily cron picks up the latest."
        ),
    )
    parser.add_argument("--end", default=None, help="Latest observation timestamp.")
    parser.add_argument("--out", default="data/aws/kmi_aws_10min.parquet")
    parser.add_argument(
        "--max-features",
        type=int,
        default=200_000,
        help="GeoServer count cap. 30 stations × 10-min steps × 30 days ≈ 130k.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = _parse_iso(args.start) if args.start else now - timedelta(hours=24)
    end = _parse_iso(args.end) if args.end else now
    if end < start:
        LOG.error("--end must be after --start")
        return 2

    # CQL filter on the timestamp field. GeoServer's ECQL doesn't accept the
    # OGC `DURING` keyword on its WFS endpoint — `BETWEEN '…' AND '…'` does,
    # but only when the literals are *single-quoted* ISO strings.
    start_s = start.isoformat().replace("+00:00", "Z")
    end_s = end.isoformat().replace("+00:00", "Z")
    cql = f"timestamp BETWEEN '{start_s}' AND '{end_s}'"

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        LOG.info("Fetching window [%s, %s]", start, end)
        page = fetch_all(client, args.max_features, cql)
        feats = page.get("features", []) or []
        rows = [flatten(f) for f in feats]
        if len(rows) >= args.max_features:
            LOG.warning(
                "Hit max-features=%d; window may be truncated. Re-run with smaller windows.",
                args.max_features,
            )

    if not rows:
        LOG.warning("No features returned for window [%s, %s]", start, end)
        return 0

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "code"]).reset_index(drop=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["timestamp", "code"], keep="last")
            .sort_values(["timestamp", "code"])
            .reset_index(drop=True)
        )

    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("Wrote %d rows (%d stations × %d timesteps) → %s",
             len(df), df["code"].nunique(), df["timestamp"].nunique(), out_path)
    return 0


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s = s + "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
