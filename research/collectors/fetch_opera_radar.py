"""EUMETNET OPERA pan-European radar composite — the model's truth source.

Pulls the OPERA `COMP` products (instantaneous rain **RATE**, 1-h accumulation
**ACRR**, with their quality band) from the open European Weather Cloud buckets
on CloudFerro — anonymous, no key, CC-BY-4.0:

    24-h rolling cache : https://s3.waw3-1.cloudferro.com/openradar-24h
    archive (2012→)    : https://s3.waw3-1.cloudferro.com/openradar-archive

Layout: ``YYYY/MM/DD/OPERA/COMP/OPERA@<YYYYMMDDTHHMM>@0@<PRODUCT>.tiff`` at
15-min cadence. The product token drifts across eras (``RATE`` today,
``QIND_RATE`` in older years), so we match by regex, not exact name.

The native composite is a 1900×2200 LAEA grid at 2 km covering all of Europe.
We don't need Europe — we **windowed-crop** each COG over our BeNeLux+context
bbox via GDAL ``/vsicurl`` (reads only the needed COG tiles), so both transfer
and storage are ~15× smaller and the whole archive fits in tens of GB.

    # forward (latest few cycles), default bbox + RATE:
    python collectors/fetch_opera_radar.py --mode forward --out data/opera
    # backfill a date range from the archive:
    python collectors/fetch_opera_radar.py --mode backfill \
        --start 2024-08-01 --end 2024-09-01 --bucket archive --out data/opera
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pathlib
import re
import sys

import httpx

LOG = logging.getLogger("pluvio.fetch_opera")

BUCKETS = {
    "24h": "https://s3.waw3-1.cloudferro.com/openradar-24h",
    "archive": "https://s3.waw3-1.cloudferro.com/openradar-archive",
}
# BeNeLux + upstream context (North Sea / France / Germany / UK edge). Wider
# than the verified domain on purpose — the model needs to see weather advect in.
DEFAULT_BBOX = (-2.0, 47.0, 12.0, 56.0)  # W, S, E, N (EPSG:4326)


def list_day_keys(client: httpx.Client, base: str, day: dt.date, product: str) -> dict[str, str]:
    """Map issue-timestamp 'YYYYmmddTHHMM' → object key, for `product` on `day`."""
    prefix = f"{day:%Y/%m/%d}/OPERA/COMP/"
    pat = re.compile(rf"OPERA@(\d{{8}}T\d{{4}})@0@([A-Z_]*{product})\.tiff$")
    out: dict[str, str] = {}
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = client.get(f"{base}/", params=params, timeout=60)
        r.raise_for_status()
        body = r.text
        for key in re.findall(r"<Key>([^<]+)</Key>", body):
            m = pat.search(key)
            if m:
                out.setdefault(m.group(1), key)  # first match wins (prefer plain RATE)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        token = m.group(1) if (m and "<IsTruncated>true</IsTruncated>" in body) else None
        if not token:
            break
    return out


def crop_cog(base: str, key: str, bbox: tuple[float, float, float, float], out_path: pathlib.Path) -> bool:
    """Windowed-read the COG over bbox (WGS84) and write a small cropped GeoTIFF.
    Returns True on success."""
    import numpy as np
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tiff,.tif")
    url = f"/vsicurl/{base}/{key}"
    w, s, e, n = bbox
    try:
        with rasterio.open(url) as ds:
            xs, ys = warp_transform("EPSG:4326", ds.crs, [w, e, e, w], [s, s, n, n])
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)
            data = ds.read(window=win)  # all bands (RATE + quality)
            prof = ds.profile.copy()
            prof.update(
                width=data.shape[2],
                height=data.shape[1],
                transform=ds.window_transform(win),
                compress="deflate",
                predictor=2,
                driver="GTiff",
                tiled=True,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.tiff")
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(data)
        os.replace(tmp, out_path)
        return True
    except Exception as exc:  # noqa: BLE001 — one bad timestep shouldn't stop the run
        LOG.warning("crop failed %s (%s)", key, exc)
        return False


def _disk_free_gb(path: pathlib.Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def collect(
    base: str,
    products: list[str],
    bbox: tuple[float, float, float, float],
    out_dir: pathlib.Path,
    days: list[dt.date],
    min_free_gb: float,
) -> dict:
    n_ok = n_skip = n_miss = 0
    with httpx.Client(http2=False) as client:
        for day in days:
            if _disk_free_gb(out_dir) < min_free_gb:
                LOG.error("disk below %.0f GB free — stopping", min_free_gb)
                break
            for product in products:
                keys = list_day_keys(client, base, day, product)
                if not keys:
                    LOG.info("%s %s: no keys", day, product)
                    continue
                for ts, key in sorted(keys.items()):
                    out_path = out_dir / product / f"{day:%Y/%m/%d}" / f"{ts}_{product}.tiff"
                    if out_path.exists():
                        n_skip += 1
                        continue
                    if crop_cog(base, key, bbox, out_path):
                        n_ok += 1
                    else:
                        n_miss += 1
                LOG.info("%s %s: %d cropped (%d skipped, %d failed)", day, product, n_ok, n_skip, n_miss)
    return {"written": n_ok, "skipped": n_skip, "failed": n_miss}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["forward", "backfill"], default="forward")
    p.add_argument("--bucket", choices=list(BUCKETS), default=None,
                   help="default: 24h for forward, archive for backfill")
    p.add_argument("--products", default="RATE", help="comma list, e.g. RATE,ACRR")
    p.add_argument("--start", help="backfill start date YYYY-MM-DD")
    p.add_argument("--end", help="backfill end date YYYY-MM-DD (exclusive)")
    p.add_argument("--forward-days", type=int, default=1, help="forward: how many recent days to scan")
    p.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)), help="W,S,E,N EPSG:4326")
    p.add_argument("--out", default="data/opera")
    p.add_argument("--min-free-gb", type=float, default=20.0, help="stop if disk drops below this")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    bbox = tuple(float(x) for x in args.bbox.split(","))
    products = [x.strip() for x in args.products.split(",") if x.strip()]
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket = args.bucket or ("archive" if args.mode == "backfill" else "24h")
    base = BUCKETS[bucket]

    today = dt.datetime.now(dt.UTC).date()
    if args.mode == "forward":
        days = [today - dt.timedelta(days=i) for i in range(args.forward_days)]
    else:
        if not args.start:
            LOG.error("--start required for backfill")
            return 2
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end) if args.end else today + dt.timedelta(days=1)
        days = [start + dt.timedelta(days=i) for i in range((end - start).days)]

    LOG.info("OPERA collect: mode=%s bucket=%s products=%s days=%d bbox=%s out=%s",
             args.mode, bucket, products, len(days), bbox, out_dir)
    summary = collect(base, products, bbox, out_dir, days, args.min_free_gb)
    LOG.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
