"""Option B — pull KNMI's archived radar nowcast + observation-corrected
ground truth, retrospectively.

Datasets (both CC BY 4.0, dataset names verified live on 2026-05-26):
  radar_forecast / 2.0          → RAD_NL25_RAC_FM_<YYYYMMDDHHMM>.h5
                                  one issue time per file, 25 lead steps inside.
  nl_rdr_data_rtcor_5m / 1.0    → RAD_NL25_RAC_RT_<YYYYMMDDHHMM>.h5
                                  one observation per file (5-min accumulation).

The KNMI listing API returns files alphabetically, so without a hint it
starts at the oldest file in the archive. We use ``startAfterFilename`` to
jump straight to the requested date window — turns a multi-thousand-page
walk into a single page.

Requires a free API key in the env: ``KNMI_API_KEY``.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

LOG = logging.getLogger("pluvio.fetch_knmi")

API_ROOT = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
USER_AGENT = "pluvio-research/0.1 (+https://github.com/appmire/pluvio)"

DATASETS = {
    # alias        → (dataset_id, version, filename_prefix)
    "radar_forecast": ("radar_forecast", "2.0", "RAD_NL25_RAC_FM_"),
    "rtcor": ("nl_rdr_data_rtcor_5m", "1.0", "RAD_NL25_RAC_RT_"),
    # also accept the canonical dataset IDs directly
    "nl_rdr_data_rtcor_5m": ("nl_rdr_data_rtcor_5m", "1.0", "RAD_NL25_RAC_RT_"),
}


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": api_key, "User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def _start_after_filename(prefix: str, start: datetime) -> str:
    """Build the alphabetically-just-before filename for the listing API."""
    # The filename embeds the *end* of the 5-min window for RT files and the
    # issue time for FM files. Subtracting one minute jumps us to the file
    # immediately before the requested start, then pagination returns
    # everything from `start` onwards.
    one_min_before = start - timedelta(minutes=1)
    stamp = one_min_before.strftime("%Y%m%d%H%M")
    return f"{prefix}{stamp}.h5"


def list_files_in_window(
    client: httpx.Client,
    dataset: str,
    version: str,
    prefix: str,
    start: datetime,
    end: datetime,
) -> list[str]:
    base = f"{API_ROOT}/{dataset}/versions/{version}/files"
    files: list[str] = []
    next_token: str | None = None
    after = _start_after_filename(prefix, start)
    end_ts_str = end.strftime("%Y%m%d%H%M")
    while True:
        # The API rejects startAfterFilename + nextPageToken together (400).
        # Use startAfterFilename only to seed the first page; once the server
        # hands us a nextPageToken, page with that alone.
        params: dict[str, object] = {"maxKeys": 500}
        if next_token:
            params["nextPageToken"] = next_token
        else:
            params["startAfterFilename"] = after
        r = client.get(base, params=params)
        r.raise_for_status()
        body = r.json()
        for entry in body.get("files", []):
            name = entry["filename"]
            if not name.startswith(prefix):
                continue
            stamp = _stamp_from_filename(name)
            if stamp is None:
                continue
            if stamp > end_ts_str:
                return files
            files.append(name)
        next_token = body.get("nextPageToken")
        if not next_token or not body.get("isTruncated"):
            break
    return files


def _stamp_from_filename(name: str) -> str | None:
    m = re.search(r"(\d{12})", name)
    return m.group(1) if m else None


def download(
    client: httpx.Client,
    dataset: str,
    version: str,
    filename: str,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    target = out_dir / filename
    if target.exists():
        return target
    url = f"{API_ROOT}/{dataset}/versions/{version}/files/{filename}/url"
    r = client.get(url)
    r.raise_for_status()
    signed_url = r.json()["temporaryDownloadUrl"]
    # Don't send the API auth header to the CDN — signed URLs reject it.
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
    parser.add_argument(
        "--dataset",
        default="radar_forecast",
        choices=sorted(DATASETS),
        help="Dataset alias; canonical IDs also accepted.",
    )
    parser.add_argument("--out", default="data/knmi", help="Output directory root")
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

    dataset_id, version, prefix = DATASETS[args.dataset]
    out_dir = pathlib.Path(args.out) / dataset_id / version
    out_dir.mkdir(parents=True, exist_ok=True)

    with _client(api_key) as client:
        LOG.info("Listing %s v%s in [%s, %s]…", dataset_id, version, start, end)
        names = list_files_in_window(client, dataset_id, version, prefix, start, end)
        LOG.info("Found %d files. Downloading to %s", len(names), out_dir)
        for name in tqdm(names, unit="file"):
            try:
                download(client, dataset_id, version, name, out_dir)
            except httpx.HTTPError as exc:
                LOG.warning("Failed %s: %s", name, exc)
            time.sleep(0.02)
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
