"""DWD RADKLIM-YW — gauge-adjusted radar precipitation (independent truth/verif).

RADKLIM (reprocessed RADOLAN, version 2017.002) is a high-quality, gauge-adjusted
radar QPE on the 1 km national grid, 5-min, archived 2001→present (verified to
2026). We use it as an **independent high-quality precipitation truth** — both a
cross-check target for training and a verification reference distinct from OPERA.
Coverage is Germany + margins, so over our BeNeLux+context domain it informs the
eastern part (E-BE/NL/LUX + W-DE).

Source: opendata.dwd.de/.../5_minutes/radolan/reproc/2017_002/bin/<year>/ — one
tar per month of 5-min RADOLAN-binary (.bin) composites. We parse with wradlib,
convert to mm/h, regrid onto a regular lat/lon GeoTIFF over the domain (so the
zarr builder's rasterio reproject consumes it like OPERA/MTG), one file per slot.

    python collectors/fetch_radklim.py --start 2024-08 --end 2024-10 --out data/radklim
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import pathlib
import re
import sys
import tarfile
import urllib.request

import numpy as np

LOG = logging.getLogger("pluvio.fetch_radklim")
BASE = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/reproc/2017_002/bin"
# Target regular lat/lon grid over BeNeLux+context (~0.02° ≈ 2 km, near native 1 km).
DEFAULT_BBOX = (-2.0, 47.0, 12.0, 56.0)  # W,S,E,N
DEFAULT_RES = 0.02
TS_RE = re.compile(r"(\d{10})")  # YYMMDDHHMM in RADOLAN filenames


def _months(start, end):
    def parse(s):
        if s == "now":
            t = dt.datetime.now(dt.UTC); return t.year, t.month
        y, m = s.split("-"); return int(y), int(m)
    (y0, m0), (y1, m1) = parse(start), parse(end)
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _target_grid(bbox, res):
    w, s, e, n = bbox
    lons = np.arange(w, e + 1e-9, res)
    lats = np.arange(n, s - 1e-9, -res)  # north→south
    return lats, lons


def _month_tar_urls(year, month):
    """List candidate tar URLs for a month (DWD names vary: .tar / .tar.gz)."""
    import html
    listing = urllib.request.urlopen(f"{BASE}/{year}/", timeout=60).read().decode(errors="ignore")
    hrefs = re.findall(r'href="([^"]+)"', listing)
    mm = f"{month:02d}"
    return [f"{BASE}/{year}/{html.unescape(h)}" for h in hrefs
            if h.endswith((".tar", ".tar.gz")) and (f"{year}{mm}" in h or f"-{mm}." in h or f"_{year}{mm}" in h)]


def build_month(year, month, lats, lons, out_dir, cadence_min):
    import wradlib as wrl
    from scipy.spatial import cKDTree

    urls = _month_tar_urls(year, month)
    if not urls:
        LOG.warning("%04d-%02d: no tar found", year, month); return 0
    TLON, TLAT = np.meshgrid(lons, lats)
    tgt_pts = np.column_stack([TLON.ravel(), TLAT.ravel()])
    written, last = 0, None
    nn_idx = nn_far = None  # precomputed once: source→target nearest-neighbour map

    # RADKLIM monthly tar (uncompressed) → daily .tar.gz → 5-min .bin composites.
    for url in urls:
        LOG.info("  fetching %s", url.rsplit("/", 1)[-1])
        raw = urllib.request.urlopen(url, timeout=600).read()
        mmode = "r:gz" if url.endswith(".gz") else "r:"
        bins = []  # (ts, bytes) kept at cadence
        with tarfile.open(fileobj=io.BytesIO(raw), mode=mmode) as mtf:
            for daymem in sorted(mtf.getmembers(), key=lambda m: m.name):
                if not daymem.isfile() or not daymem.name.endswith((".tar.gz", ".tgz", ".tar")):
                    continue
                dmode = "r:gz" if daymem.name.endswith((".gz", ".tgz")) else "r:"
                with tarfile.open(fileobj=io.BytesIO(mtf.extractfile(daymem).read()), mode=dmode) as dtf:
                    for binmem in sorted(dtf.getmembers(), key=lambda m: m.name):
                        if not binmem.isfile():
                            continue
                        m = TS_RE.search(binmem.name)
                        if not m:
                            continue
                        ts = dt.datetime.strptime(m.group(1), "%y%m%d%H%M").replace(tzinfo=dt.UTC)
                        if last is not None and (ts - last).total_seconds() / 60 < cadence_min:
                            continue
                        last = ts
                        bins.append((ts, dtf.extractfile(binmem).read()))

        for ts, payload in bins:
            target = out_dir / f"radklim_yw_{ts:%Y%m%dT%H%M}.tiff"
            if target.exists():
                continue
            try:
                data, attrs = wrl.io.read_radolan_composite(io.BytesIO(payload))
                # MASKED array → fill mask FIRST (np.asarray drops it, leaking nodata
                # flags as huge negatives). wradlib ALREADY applies `precision`, so
                # `data` is mm per 5-min — multiply only by 12 (5-min → mm/h). The
                # previous `* precision * 12` double-applied precision → values ~100×
                # too low (verified 2026-06-18: median wet 0.01 vs correct 0.96 mm/h).
                arr = np.ma.filled(data, np.nan).astype("float64")
                cl = attrs.get("cluttermask")        # drop flagged clutter pixels
                if cl is not None and len(cl):
                    arr.flat[cl] = np.nan
                arr = np.clip(arr * 12.0, 0.0, 250.0).astype("float32")  # mm/h, cap residual clutter
                if nn_idx is None:  # build the RADOLAN→target NN map ONCE (grid is fixed)
                    rg = wrl.georef.get_radolan_grid(*arr.shape, wgs84=True)
                    src = np.column_stack([rg[..., 0].ravel(), rg[..., 1].ravel()])
                    dist, nn_idx = cKDTree(src).query(tgt_pts)
                    nn_far = dist > 0.05  # target cells >~5 km from any RADOLAN cell → outside
                out = arr.ravel()[nn_idx]
                out[nn_far] = 0.0
                _write_tiff(target, np.nan_to_num(out.reshape(TLON.shape), nan=0.0).astype("float32"),
                            lats, lons)
                written += 1
            except Exception as exc:  # noqa: BLE001
                LOG.warning("  parse @ %s failed (%s)", ts, exc)
    LOG.info("%04d-%02d: wrote %d slots", year, month, written)
    return written


def _write_tiff(path, arr, lats, lons):
    import rasterio
    from rasterio.transform import from_origin
    res_lon = abs(lons[1] - lons[0]); res_lat = abs(lats[1] - lats[0])
    transform = from_origin(lons[0] - res_lon / 2, lats[0] + res_lat / 2, res_lon, res_lat)
    tmp = path.with_suffix(".tiff.part")
    with rasterio.open(tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=np.nan) as dst:
        dst.write(arr, 1)
    tmp.rename(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2024-08"); p.add_argument("--end", default="now")
    p.add_argument("--bbox", default=",".join(map(str, DEFAULT_BBOX)))
    p.add_argument("--res-deg", type=float, default=DEFAULT_RES)
    p.add_argument("--cadence-min", type=int, default=15, help="keep one slot per N min (OPERA is 15)")
    p.add_argument("--out", default="data/radklim")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(float(x) for x in args.bbox.split(","))
    lats, lons = _target_grid(bbox, args.res_deg)
    LOG.info("RADKLIM-YW: %s..%s grid %d×%d → %s", args.start, args.end, len(lats), len(lons), out)
    n = 0
    for y, m in _months(args.start, args.end):
        try:
            n += build_month(y, m, lats, lons, out, args.cadence_min)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%04d-%02d failed (%s)", y, m, exc)
    LOG.info("done: %d slots in %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
