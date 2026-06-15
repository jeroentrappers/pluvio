"""KNMI 10-minute automatic-weather-station observations (the Netherlands).

The Belgian KMI AWS network (~14 stations) is collected by fetch_kmi_aws.py;
this adds KNMI's ~50 NL stations north of the border. Source is the KNMI Data
Platform dataset ``10-minute-in-situ-meteorological-observations`` v1.0 (the old
``Actuele10mindataKNMIstations`` was deprecated in 2025). One NetCDF (HDF5) file
per 10-min slot, ~70 stations each, history back to 2012 — so this backfills the
training window, unlike the forward-only crowd/Netatmo source.

We map KNMI's variables onto the shared AWS parquet schema so build_zarr pools
KMI(BE) + KNMI(NL) + Netatmo into the same aws_* surface channels:
    ta  Air Temperature 1-min mean  [°C]  → temp_dry_shelter_avg
    rh  Relative Humidity 1-min mean [%]  → humidity_rel_shelter_avg
    pp  Air Pressure at MSL 1-min mean[hPa]→ pressure   (MSL, matches BE & Netatmo)
    ff  Wind Speed at 10 m mean     [m/s] → wind_speed_10m

The .nc files themselves aren't kept — we stream, parse, and append to Parquet.
Requires KNMI_API_KEY in .env (same key as the radar collector; shared quota).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import h5py
import httpx
import numpy as np
import pandas as pd

# Reuse the KNMI Data Platform plumbing (auth, paged listing, rate-limit backoff).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collectors.fetch_knmi_archive import (  # noqa: E402
    API_ROOT, _client, list_files_in_window, rl_get,
)

LOG = logging.getLogger("pluvio.fetch_knmi_aws")

DATASET, VERSION = "10-minute-in-situ-meteorological-observations", "1.0"
PREFIX = "KMDS__OPER_P___10M_OBS_L2_"
EPOCH_1950 = datetime(1950, 1, 1, tzinfo=timezone.utc)

# KNMI var → our shared AWS column.
VAR_MAP = {
    "ta": "temp_dry_shelter_avg",
    "rh": "humidity_rel_shelter_avg",
    "pp": "pressure",
    "ff": "wind_speed_10m",
}

# KNMI also reports the BES (Caribbean) islands — drop anything outside a
# superset of the analysis grid so only NL + border stations are kept.
GRID_BBOX = (-1.0, 48.0, 12.0, 57.0)  # minx, miny, maxx, maxy
COLUMNS = ["timestamp", "station_id", "lat", "lon", "pressure",
           "temp_dry_shelter_avg", "humidity_rel_shelter_avg", "wind_speed_10m"]


def _read_env_key() -> str:
    env_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("KNMI_API_KEY="):
                return line.split("=", 1)[1].strip()
    import os
    return os.environ.get("KNMI_API_KEY", "").strip()


def _download_bytes(client: httpx.Client, filename: str) -> bytes:
    """Fetch one file's signed URL and return its bytes (no disk write)."""
    r = rl_get(client, f"{API_ROOT}/{DATASET}/versions/{VERSION}/files/{filename}/url")
    r.raise_for_status()
    url = r.json()["temporaryDownloadUrl"]
    resp = httpx.get(url, timeout=httpx.Timeout(120.0))  # signed URL: no auth header
    resp.raise_for_status()
    return resp.content


def _parse_nc(blob: bytes) -> pd.DataFrame | None:
    """Parse one 10-min obs NetCDF (HDF5) into rows matching the AWS schema."""
    import io
    with h5py.File(io.BytesIO(blob), "r") as f:
        t = float(np.asarray(f["time"])[0])
        ts = EPOCH_1950 + timedelta(seconds=t)
        n = f["lat"].shape[0]
        data = {
            "timestamp": pd.Timestamp(ts).tz_convert(None),
            "lat": np.asarray(f["lat"][:], dtype="float64"),
            "lon": np.asarray(f["lon"][:], dtype="float64"),
        }
        sid = np.asarray(f["station"][:])
        data["station_id"] = [s.decode() if isinstance(s, bytes) else str(s) for s in sid]
        for var, col in VAR_MAP.items():
            data[col] = (np.asarray(f[var][:, 0], dtype="float64")
                         if var in f else np.full(n, np.nan))
    df = pd.DataFrame(data)
    minx, miny, maxx, maxy = GRID_BBOX
    in_grid = df.lat.between(miny, maxy) & df.lon.between(minx, maxx)
    has_data = df[list(VAR_MAP.values())].notna().any(axis=1)
    keep = in_grid & has_data
    return df[keep].reset_index(drop=True) if keep.any() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO date or datetime.")
    parser.add_argument("--end", required=True, help="UTC ISO date or datetime.")
    parser.add_argument("--cadence-minutes", type=int, default=10,
                        help="Subsample (native 10 min). Use 30 to cut backfill volume.")
    parser.add_argument("--out", default="data/aws/knmi_aws_10min.parquet")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    key = _read_env_key()
    if not key:
        LOG.error("KNMI_API_KEY not set (.env)."); return 2
    start, end = _iso(args.start), _iso(args.end)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with _client(key) as client:
        files = list_files_in_window(client, DATASET, VERSION, PREFIX, start, end,
                                     cadence_minutes=args.cadence_minutes)
        LOG.info("%d files in window [%s, %s] @ %d-min cadence",
                 len(files), start, end, args.cadence_minutes)
        if not files:
            return 0
        frames, done, failed = [], 0, 0
        for fn in files:
            try:
                df = _parse_nc(_download_bytes(client, fn))
                if df is not None:
                    frames.append(df)
                done += 1
            except (httpx.HTTPError, OSError, KeyError) as exc:
                failed += 1
                LOG.warning("  %s: %s", fn, exc)
            if done % 500 == 0 and done:
                LOG.info("  %d/%d files parsed (%d rows so far)", done, len(files),
                         sum(len(x) for x in frames))

    if not frames:
        LOG.warning("no rows parsed"); return 1
    new = pd.concat(frames, ignore_index=True)[COLUMNS]
    if out.exists():
        prev = pd.read_parquet(out)
        prev["timestamp"] = pd.to_datetime(prev["timestamp"])
        new = pd.concat([prev, new], ignore_index=True)
    new = new.drop_duplicates(subset=["timestamp", "station_id"], keep="last")
    new.to_parquet(out, index=False)
    LOG.info("Wrote %d rows (%d files ok, %d failed) → %s", len(new), done, failed, out)
    return 0


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
