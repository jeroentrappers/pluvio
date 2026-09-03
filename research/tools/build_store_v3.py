"""Build the v3 training store: ONE regular lat/lon grid, full Benelux.

Why v3 (2026-09-02 input-validation night):
  * the KNMI-stereo legacy grid mixed geometries — radar/truth on the TRIMMED
    765->700 extent, aux regridded to the UNTRIMMED extent (up to ~0.5 deg
    internal misalignment). Both true geometries are now known exactly, so
    this rebuild heals the store instead of inheriting the damage;
  * the serving box must cover the whole Netherlands; the old analysis grid
    already reaches 55.97N (the trim cut the SOUTH at 49.93N), so NL data
    exists historically — only S-Belgium inputs (48.9-49.93N) are NaN before
    the ingestion fix.

Grid: BOX = (1.5, 48.9, 7.5, 54.2), N=192 -> ~3.1 x 3.4 km cells,
row 0 = north. Everything f16, zarr v2 (GPU node is python3.10/zarr2),
chunks (16, ...).

Sources (per issue of the legacy store):
  radar, aux   sampled from the legacy arrays at each new cell's lat/lon via
               the analysis grid's inverse mapping — TRIMMED extent for
               radar/truth-side arrays, UNTRIMMED for aux (their actual
               georeference).
  truth        south of 52.5N: the 768-grid QPE day zarrs (sharp, 4x native);
               north of 52.5N: the legacy rtcor truth reprojected;
               where both exist the QPE composite wins.
  static_*     reprojected once from the legacy store (trimmed mapping).

Phases:  --create   then chunk-aligned  --range LO HI  shards (multiples of 16).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import sys

import cv2
import numpy as np
import pyproj
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.grid import Grid  # noqa: E402

LOG = logging.getLogger("pluvio.build_store_v3")

BOX = (1.5, 48.9, 7.5, 54.2)          # west, south, east, north
N_OUT = 192
F16 = "float16"
QPE_BOUNDS = (1.5, 48.9, 7.5, 52.5)   # the 768 research grid
QPE_N = 768

_PROJ4 = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +x_0=0 +y_0=0"
_CORNERS_LONLAT = [(0.0, 49.362064), (0.0, 55.973602),
                   (10.856453, 55.388973), (9.0093, 48.8953)]


def _legacy_index_maps(h: int, w: int):
    """(row, col) float index maps from the new grid's lat/lon into the legacy
    analysis grid — one pair for the TRIMMED extent (radar/truth), one for the
    UNTRIMMED extent (aux, which were regridded before the trim fix)."""
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", _PROJ4, always_xy=True)
    xs, ys = [], []
    for lon, lat in _CORNERS_LONLAT:
        x, y = to_xy.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    wst, sth, est, nth = BOX
    lons = np.linspace(wst, est, N_OUT)
    lats = np.linspace(nth, sth, N_OUT)     # row 0 = north
    LON, LAT = np.meshgrid(lons, lats)
    gx, gy = to_xy.transform(LON, LAT)

    def maps(y_bottom: float):
        # legacy grid: rows linspace(ymax -> y_bottom, h), cols linspace(xmin -> xmax, w)
        rr = (ymax - gy) / (ymax - y_bottom) * (h - 1)
        cc = (gx - xmin) / (xmax - xmin) * (w - 1)
        return rr.astype("float32"), cc.astype("float32")

    y_trim = ymax - (700.0 / 765.0) * (ymax - ymin)
    return maps(y_trim), maps(ymin)          # (trimmed, untrimmed)


def _remap(field: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    return cv2.remap(np.ascontiguousarray(field, dtype="float32"), cc, rr,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=float("nan"))


def _truth_rac(epoch: int) -> np.ndarray | None:
    """Sharp truth on the new grid from the native 1-km RAC/RTCOR tars.

    The deep QPE day-zarr archive turned out to be retention-pruned (~3 days),
    so the single consistent historical truth is KNMI RTCOR (2019->, whole
    domain incl. all of NL and most of BE; the far-SW corner of the box falls
    outside the native domain and stays NaN). Reuses tools.knmi_rtcor: true
    stereographic per-cell mapping, calibrated, gauge-adjusted product.
    """
    import tools.knmi_rtcor as kr

    stamp = dt.datetime.fromtimestamp(int(epoch), dt.UTC).strftime("%Y%m%dT%H%M")
    try:
        r = kr.rate(stamp, BOX, (N_OUT, N_OUT))
    except Exception as exc:
        LOG.warning("rac truth failed for %s: %s", stamp, exc)
        return None
    return None if r is None else r.astype("float32")


def _truth_qpe(qpe_root: pathlib.Path, epoch: int) -> np.ndarray | None:
    """Sharp truth over the southern (QPE) part of the box, on the new grid."""
    ts = dt.datetime.fromtimestamp(int(epoch), dt.UTC)
    zp = qpe_root / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    if not zp.exists():
        return None
    try:
        root = zarr.open_group(str(zp), mode="r")
        slot = int(round((epoch % 86400) / 300))
        if not 0 <= slot < root["rate"].shape[0]:
            return None
        rate = np.asarray(root["rate"][slot], dtype="float32")
        if not np.isfinite(rate).any():
            return None
    except Exception as exc:
        LOG.warning("qpe read failed %s: %s", zp, exc)
        return None
    wst, sth, est, nth = BOX
    qW, qS, qE, qN = QPE_BOUNDS
    out = np.full((N_OUT, N_OUT), np.nan, dtype="float32")
    lats = np.linspace(nth, sth, N_OUT)
    lons = np.linspace(wst, est, N_OUT)
    rows = np.where(lats <= qN)[0]
    if rows.size == 0:
        return None
    # Pre-decimate 768 -> 384 by block-mean (remap has no INTER_AREA; going
    # straight to ~3 km with LINEAR would alias the sharp field), then sample.
    r2 = np.nan_to_num(rate).reshape(384, 2, 384, 2).mean(axis=(1, 3))
    rr = ((qN - lats[rows]) / (qN - qS) * (r2.shape[0] - 1)).astype("float32")
    cc = ((lons - qW) / (qE - qW) * (r2.shape[1] - 1)).astype("float32")
    CC, RR = np.meshgrid(cc, rr)
    sub = cv2.remap(r2.astype("float32"), CC, RR, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    out[rows[0]:rows[-1] + 1] = sub
    return out


def prefetch_tars(t: np.ndarray) -> None:
    """Download every daily RAC tar the issue list needs, sequentially —
    shards then run compute-only against a warm cache (no download races)."""
    import tools.knmi_rtcor as kr

    stamps = sorted({dt.datetime.fromtimestamp(int(e), dt.UTC).strftime("%Y%m%dT%H%M")
                     for e in t})
    days = sorted({(s[:8], s) for s in stamps})
    seen = set()
    got = miss = 0
    for i, (_day, stamp) in enumerate(days):
        tar = kr._tar_for(stamp)
        key = None if tar is None else tar.name
        if key in seen:
            continue
        seen.add(key)
        got += tar is not None
        miss += tar is None
        if (got + miss) % 25 == 0:
            LOG.info("prefetch: %d tars ok, %d missing", got, miss)
    LOG.info("prefetch complete: %d ok, %d missing", got, miss)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="/opt/pluvio/zarr/timeseries.zarr")
    p.add_argument("--qpe", default="/mnt/storagebox/qpe")
    p.add_argument("--out", required=True)
    p.add_argument("--create", action="store_true")
    p.add_argument("--prefetch", action="store_true",
                   help="download all needed RAC tars sequentially, then exit")
    p.add_argument("--range", nargs=2, type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    src = zarr.open_group(args.src, mode="r")
    names = list(src.array_keys())
    per_issue = [k for k in names if src[k].ndim >= 1 and src[k].shape[0] > 1000]
    n = min(src[k].shape[0] for k in per_issue)
    h, w = src["radar"].shape[-2:]
    (rr_t, cc_t), (rr_u, cc_u) = _legacy_index_maps(h, w)

    if args.prefetch:
        t = np.asarray(src["issue_time"][:n]).astype("int64")
        if t.max() > 10**12:
            t //= 1000
        prefetch_tars(t)
        return 0

    if args.create:
        dst = zarr.open_group(args.out, mode="w", zarr_format=2)
        dst.attrs.update(dict(src.attrs))
        dst.attrs.update({"grid_n": N_OUT, "bounds": list(BOX), "store_version": 3,
                          "grid": "regular lat/lon, row 0 = north"})
        # Grid contract (1.1): every store carries its own georeference, so
        # consumers read it instead of assuming a box. Written last so it
        # always wins if the legacy `bounds`/`grid_n` keys above ever drift.
        dst.attrs.update(Grid.regular(BOX, (N_OUT, N_OUT)).to_attrs())
        for k in names:
            a = src[k]
            if k not in per_issue:
                if a.ndim == 2:   # static — trimmed mapping like radar
                    data = _remap(np.nan_to_num(np.asarray(a[:], dtype="float32")), rr_t, cc_t)
                    arr = dst.create_array(k, data=data.astype(F16), chunks="auto")
                else:
                    arr = dst.create_array(k, data=np.asarray(a[:]), chunks="auto")
                arr.attrs.update(dict(a.attrs))
                continue
            if a.ndim == 1:
                arr = dst.create_array(k, data=np.asarray(a[:n]), chunks="auto")
            else:
                shape = (n,) + a.shape[1:-2] + (N_OUT, N_OUT)
                chunks = (16,) + a.shape[1:-2] + (N_OUT, N_OUT)
                arr = dst.create_array(k, shape=shape, dtype=F16, chunks=chunks,
                                       fill_value=np.nan)
            arr.attrs.update(dict(a.attrs))
        LOG.info("created %s: %d issues, %dx%d over %s", args.out, n, N_OUT, N_OUT, BOX)
        return 0

    assert args.range, "--range LO HI required unless --create"
    lo, hi = args.range
    hi = min(hi, n)
    dst = zarr.open_group(args.out, mode="a")
    t = np.asarray(src["issue_time"][:n]).astype("int64")
    if t.max() > 10**12:
        t //= 1000
    qpe_root = pathlib.Path(args.qpe)
    B = 16
    done = 0
    for s0 in range(lo, hi, B):
        s1 = min(s0 + B, hi)
        for k in per_issue:
            a = src[k]
            if a.ndim == 1:
                continue
            if k == "truth":
                block = []
                for i in range(s0, s1):
                    rac = _truth_rac(t[i])
                    if rac is None:
                        # fall back to the legacy (coarse) truth reprojection
                        rac = _remap(np.asarray(a[i], dtype="float32"), rr_t, cc_t)
                    q = _truth_qpe(qpe_root, t[i])   # last ~3 days: composite wins
                    if q is not None:
                        use_q = np.isfinite(q)
                        rac[use_q] = q[use_q]
                    block.append(rac)
                block = np.stack(block)
            elif a.ndim == 3:
                # every 3-D per-issue array is an aux channel: they were
                # regridded to the UNTRIMMED extent (their real georeference)
                block = np.stack([_remap(np.asarray(a[i], dtype="float32"), rr_u, cc_u)
                                  for i in range(s0, s1)])
            elif a.ndim == 4:
                block = np.stack([
                    np.stack([_remap(np.asarray(a[i, j], dtype="float32"), rr_t, cc_t)
                              for j in range(a.shape[1])])
                    for i in range(s0, s1)
                ])
            else:
                continue
            dst[k][s0:s1] = block.astype(F16)
        done += s1 - s0
        if done % 320 == 0 or s1 >= hi:
            LOG.info("range [%d,%d): %d/%d done", lo, hi, done, hi - lo)
    LOG.info("shard [%d,%d) complete", lo, hi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
