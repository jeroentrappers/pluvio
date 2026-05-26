"""Option A — live poller for KMI's `getForecasts`.

Designed to run on a cron / systemd timer / Fly machine every 10 minutes.
Each poll appends one row per (location, frame) to a Parquet partitioned
by date. After a week of data you can compute lead-time-stratified skill.

Schema (see also `notebooks/01_verification.ipynb`):

    poll_ts            : timestamp_us, UTC          → when *we* fetched
    location           : string                     → human key, e.g. "brussels"
    lat, lon           : float64                    → as queried
    frame_ts           : timestamp_us, UTC          → the forecast/observation time
    lead_min           : int32                      → minutes between poll_ts and frame_ts
    value_mm_per_h     : float32                    → KMI's per-location precip rate
    image_url          : string                     → pre-signed CDN URL (rots daily)

Pair the same `(location, frame_ts)` across polls to extract the prediction
trajectory and (eventually) the observation when `lead_min` ≤ 0.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import datetime, timezone

import httpx
import pandas as pd

from collectors._kmi_signing import sign

LOG = logging.getLogger("pluvio.collect_kmi_live")

LOCATIONS = {
    "brussels": (50.8503, 4.3517),
    "antwerp": (51.2194, 4.4025),
    "ghent": (51.0543, 3.7174),
    "liege": (50.6326, 5.5797),
    "charleroi": (50.4108, 4.4446),
    "bruges": (51.2093, 3.2247),
    "arlon": (49.6839, 5.8167),
    "ostend": (51.2247, 2.9069),
}

USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"


def fetch_one(client: httpx.Client, lat: float, lon: float) -> dict:
    params = {
        "s": "getForecasts",
        "k": sign("getForecasts"),
        "lat": round(lat, 6),
        "long": round(lon, 6),
    }
    r = client.get(
        "https://app.meteo.be/services/appv4/",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def rows_from_response(
    response: dict,
    location: str,
    lat: float,
    lon: float,
    poll_ts: datetime,
) -> list[dict]:
    seq = response.get("animation", {}).get("sequence") or []
    out: list[dict] = []
    for frame in seq:
        try:
            ts = datetime.fromisoformat(frame["time"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        out.append(
            {
                "poll_ts": poll_ts,
                "location": location,
                "lat": lat,
                "lon": lon,
                "frame_ts": ts,
                "lead_min": int((ts - poll_ts).total_seconds() // 60),
                # wire: mm/10min → mm/h (×6) so it matches the Dart side
                "value_mm_per_h": float(frame.get("value") or 0) * 6.0,
                "image_url": frame.get("uri"),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locations",
        default="brussels",
        help=f"Comma-separated keys from {sorted(LOCATIONS)}",
    )
    parser.add_argument(
        "--out",
        default="data/kmi_live.parquet",
        help="Output Parquet file. Appended on each run.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = [k.strip() for k in args.locations.split(",") if k.strip()]
    poll_ts = datetime.now(timezone.utc).replace(microsecond=0)

    new_rows: list[dict] = []
    with httpx.Client(http2=True) as client:
        for key in selected:
            if key not in LOCATIONS:
                LOG.warning("Unknown location %r; skipping", key)
                continue
            lat, lon = LOCATIONS[key]
            try:
                payload = fetch_one(client, lat, lon)
            except httpx.HTTPError as exc:
                LOG.warning("Fetch failed for %s: %s", key, exc)
                continue
            new_rows.extend(rows_from_response(payload, key, lat, lon, poll_ts))
            LOG.info("Captured %d frames for %s", len(new_rows), key)

    if not new_rows:
        LOG.error("No rows captured this poll; nothing written.")
        return 1

    df_new = pd.DataFrame(new_rows)
    if out_path.exists():
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("Wrote %d new rows (total: %d) → %s", len(df_new), len(df), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
