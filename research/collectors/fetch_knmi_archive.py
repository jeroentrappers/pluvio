"""Option B — pair KNMI's archived 2-hour radar nowcast with the observation-
corrected ground truth, retrospectively. The whole pipeline lives in one file
so it's auditable.

Datasets (both CC BY 4.0):
  radar_forecast / 2.0           — 5-min nowcast up to 2 h ahead, HDF5 per run
  nl-rdr-data-rtcor-5m / 1.0     — 5-min observation-corrected radar, HDF5

The forecast files are stamped at the *issue* time; each file contains a
single run (`forecast` group with 25 timesteps from +0 to +120 min). The
observation files are stamped at the *observation* time, one per 5-min slot.

After download we'll convert each lead-time → observation pair into a tidy
parquet that the verification notebook can ingest.

Requires a free API key in the env: KNMI_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

LOG = logging.getLogger("pluvio.fetch_knmi")

API_ROOT = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        http2=True,
        headers={"Authorization": api_key, "User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def list_files(
    client: httpx.Client,
    dataset: str,
    version: str,
    start: datetime,
    end: datetime,
) -> list[str]:
    """Page through ``files`` and return the filenames falling within
    [start, end]. The API supports ``begin``/``end`` filters but they apply to
    upload time, not data time — so we filter ourselves on the filenames,
    which are themselves timestamped (e.g. ``RAD_NL25_RAC_5M_202605260800.h5``).
    """
    files: list[str] = []
    next_page: str | None = None
    base = f"{API_ROOT}/{dataset}/versions/{version}/files"
    while True:
        params = {"maxKeys": 500}
        if next_page is not None:
            params["nextPageToken"] = next_page
        r = client.get(base, params=params)
        r.raise_for_status()
        body = r.json()
        for entry in body.get("files", []):
            name: str = entry["filename"]
            ts = _parse_ts_from_filename(name)
            if ts is None:
                continue
            if start <= ts <= end:
                files.append(name)
        next_page = body.get("nextPageToken")
        if not next_page or not body.get("isTruncated"):
            break
    return files


_FILENAME_STAMP_LEN = 12  # YYYYMMDDHHMM


def _parse_ts_from_filename(name: str) -> datetime | None:
    """Find the YYYYMMDDHHMM stamp KNMI embeds in file names."""
    import re

    m = re.search(r"(\d{12})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def download(
    client: httpx.Client,
    dataset: str,
    version: str,
    filename: str,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """KNMI returns a short-lived signed URL; we follow it to a CDN bucket."""
    target = out_dir / filename
    if target.exists():
        return target
    url = f"{API_ROOT}/{dataset}/versions/{version}/files/{filename}/url"
    r = client.get(url)
    r.raise_for_status()
    signed_url = r.json()["temporaryDownloadUrl"]
    # Don't send the auth header to the CDN — it makes signed URLs reject.
    with httpx.stream("GET", signed_url, timeout=httpx.Timeout(120.0)) as resp:
        resp.raise_for_status()
        with target.open("wb") as fp:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                fp.write(chunk)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO date or datetime")
    parser.add_argument("--end", required=True, help="UTC ISO date or datetime")
    parser.add_argument("--out", default="data/knmi", help="Output directory")
    parser.add_argument(
        "--dataset",
        choices=("radar_forecast", "nl-rdr-data-rtcor-5m"),
        default="radar_forecast",
    )
    parser.add_argument("--version", default=None, help="Defaults to canonical version per dataset")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    load_dotenv()

    api_key = os.environ.get("KNMI_API_KEY", "").strip()
    if not api_key:
        LOG.error("KNMI_API_KEY not set. Get one at developer.dataplatform.knmi.nl.")
        return 2

    start = _parse_iso(args.start)
    end = _parse_iso(args.end)
    if end < start:
        LOG.error("--end must be on/after --start")
        return 2

    version = args.version or {"radar_forecast": "2.0", "nl-rdr-data-rtcor-5m": "1.0"}[args.dataset]
    out_dir = pathlib.Path(args.out) / args.dataset / version
    out_dir.mkdir(parents=True, exist_ok=True)

    with _client(api_key) as client:
        LOG.info("Listing %s v%s in [%s, %s]…", args.dataset, version, start, end)
        names = list_files(client, args.dataset, version, start, end)
        LOG.info("Found %d files. Downloading to %s", len(names), out_dir)
        for name in tqdm(names, unit="file"):
            try:
                download(client, args.dataset, version, name, out_dir)
            except httpx.HTTPError as exc:
                LOG.warning("Failed %s: %s", name, exc)
            # gentle pacing
            time.sleep(0.05)
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
