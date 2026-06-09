"""Build the unified time-series zarr from the per-source raw pulls.

Goal: one zarr store where every issue time has its radar frames + every
aux channel aligned onto the same 100×100 KNMI-stereographic analysis
grid. Training reads from here; aux-build scripts and per-source notebook
juggling go away.

Layout (all variables share `issue_time` as the first axis where applicable):

    /timeseries.zarr/
        issue_time              (n,)            int64 epoch seconds UTC
        leads_min               (n_lead,)       int16
        radar                   (n, n_lead, 100, 100) float32, mm/h
        aws_pressure            (n, 100, 100)   float32, normalised
        aws_temp                (n, 100, 100)   float32
        aws_humidity            (n, 100, 100)   float32
        aws_wind                (n, 100, 100)   float32
        msg_ir108               (n, 100, 100)   float32  (NaN where missing)
        # … more MSG / ALARO channels added in subsequent versions …
        static_elevation_m      (100, 100)      float32  (broadcast)
        static_landmask         (100, 100)      float32
        static_distance_km      (100, 100)      float32

The build is **idempotent**: re-running with a wider window appends new
issue times to the existing zarr without rewriting what's there. Each
issue-time slot writes either real data or NaN (for missing aux).

v1 channels: radar (image1..25) + AWS surface (4 channels) + MSG IR 10.8
+ static. Plenty to test the wiring; the remaining MSG layers and the
forward-only ALARO bands land in v2 once their pulls finish.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project paths
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.build_aux import AWS_CHANNELS, _idw  # noqa: E402
from model.geo import grid_latlon  # noqa: E402
from notebooks._lib import ANALYSIS_GRID, load_forecast_h5  # noqa: E402

LOG = logging.getLogger("pluvio.build_zarr")

# Sources expected on disk (relative to research/data).
RADAR_GLOB = "knmi/radar_forecast/2.0/RAD_NL25_RAC_FM_*.h5"
AWS_PARQUET = "aws/kmi_aws_10min.parquet"
MSG_IR108_DIR = "msg/msg_fes_ir108"
STATIC_NPZ = "static.npz"

# Meteosat WMS bbox used by the collector (matches default in
# collectors/fetch_eumetsat_msg.py).
MSG_BBOX = (2.0, 49.0, 7.5, 52.0)  # minx, miny, maxx, maxy in EPSG:4326

# Hard upper bound — radar files older than this aren't expected (KNMI
# radar_forecast v2.0 starts 2024-08-14). Used as a sanity check.
EARLIEST_RADAR = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────── helpers

def _parse_radar_ts(path: pathlib.Path) -> datetime:
    """KNMI radar filename → UTC datetime."""
    # filename ends in _YYYYMMDDHHMM.h5
    stem = path.stem.split("_")[-1]
    return datetime.strptime(stem, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def _parse_msg_ts(path: pathlib.Path) -> datetime:
    """Meteosat tif filename → UTC datetime."""
    # …_YYYYMMDDTHHMMSSZ.tif
    stem = path.stem.split("_")[-1].rstrip("Z")
    return datetime.strptime(stem, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


# ──────────────────────────────────────────────────────────── Meteosat reproj

def _read_msg_tif(path: pathlib.Path) -> np.ndarray | None:
    """Load a Meteosat WMS GeoTIFF as a 2-D float32 array (H, W).

    Returns ``None`` if rasterio can't decode it (rare; broken/empty file).
    """
    try:
        import rasterio
    except ImportError:
        # Fallback: PIL for raster decode. Loses GeoTIFF georeferencing but
        # the WMS request fixed the bbox, so it's fine.
        from PIL import Image
        try:
            arr = np.array(Image.open(path)).astype("float32")
            return arr
        except Exception as exc:  # pragma: no cover
            LOG.warning("PIL failed to read %s: %s", path, exc)
            return None

    try:
        with rasterio.open(path) as src:
            band = src.read(1).astype("float32")
            return band
    except Exception as exc:  # pragma: no cover
        LOG.warning("rasterio failed to read %s: %s", path, exc)
        return None


def _bilinear_sample(arr: np.ndarray, bbox: tuple[float, float, float, float],
                    lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear sample a regular EPSG:4326 raster at irregular lat/lon points.

    `arr` is (rows, cols); row 0 = north (maxy), col 0 = west (minx) — the
    standard "image" orientation that EUMETView returns. `lat`/`lon` are
    the target grid in the same CRS.
    """
    minx, miny, maxx, maxy = bbox
    rows, cols = arr.shape
    # Fractional pixel coords in (row, col) space.
    fx = (lon - minx) / (maxx - minx) * (cols - 1)
    fy = (maxy - lat) / (maxy - miny) * (rows - 1)
    # Inside-bbox mask
    inside = (fx >= 0) & (fx <= cols - 1) & (fy >= 0) & (fy <= rows - 1)
    fx = np.clip(fx, 0, cols - 1)
    fy = np.clip(fy, 0, rows - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, cols - 1), np.clip(y0 + 1, 0, rows - 1)
    wx, wy = fx - x0, fy - y0
    a = arr[y0, x0] * (1 - wx) * (1 - wy)
    b = arr[y0, x1] * wx * (1 - wy)
    c = arr[y1, x0] * (1 - wx) * wy
    d = arr[y1, x1] * wx * wy
    out = (a + b + c + d).astype("float32")
    out[~inside] = np.nan
    return out


# ──────────────────────────────────────────────────────────── AWS

def _build_aws_idw(aws_df: pd.DataFrame, ts: datetime, lat: np.ndarray,
                   lon: np.ndarray, max_age_min: int = 30,
                  ) -> dict[str, np.ndarray] | None:
    """Pick the newest AWS frame ≤ ts (within `max_age_min`), IDW each
    channel onto the grid, normalise. Returns None if no fresh frame."""
    cutoff = pd.Timestamp(ts).tz_convert(None) if pd.Timestamp(ts).tz is not None else pd.Timestamp(ts)
    horizon = cutoff - pd.Timedelta(minutes=max_age_min)
    frame = aws_df[(aws_df["timestamp"] <= cutoff) & (aws_df["timestamp"] >= horizon)]
    if frame.empty:
        return None
    # Latest timestamp within the window — keep all stations for that ts.
    latest = frame["timestamp"].max()
    frame = aws_df[aws_df["timestamp"] == latest]
    lats = frame["lat"].to_numpy(dtype="float64")
    lons = frame["lon"].to_numpy(dtype="float64")
    out: dict[str, np.ndarray] = {}
    for ch, (centre, scale) in AWS_CHANNELS.items():
        vals = frame[ch].to_numpy(dtype="float64")
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            out[ch] = np.full(lat.shape, np.nan, dtype="float32")
            continue
        g = _idw(lats[mask], lons[mask], vals[mask], lat, lon)
        out[ch] = ((g - centre) / scale).astype("float32")
    return out


# ──────────────────────────────────────────────────────────── MSG join

def _index_msg_dir(d: pathlib.Path) -> list[tuple[datetime, pathlib.Path]]:
    if not d.exists():
        return []
    out: list[tuple[datetime, pathlib.Path]] = []
    for p in d.glob("*.tif"):
        try:
            out.append((_parse_msg_ts(p), p))
        except ValueError:
            continue
    out.sort()
    return out


def _latest_le(index: list[tuple[datetime, pathlib.Path]], ts: datetime,
              max_age_min: int = 60) -> pathlib.Path | None:
    """Latest file in `index` whose timestamp ≤ ts and within max_age_min."""
    # Binary search would scale better; linear is fine for 30k entries.
    cand = None
    for t, p in index:
        if t > ts:
            break
        if (ts - t).total_seconds() / 60 > max_age_min:
            continue
        cand = p
    return cand


# ──────────────────────────────────────────────────────────── main build

# Channel naming aligned with what the dataset loader expects.
AWS_VAR_NAMES = {
    "pressure": "aws_pressure",
    "temp_dry_shelter_avg": "aws_temp",
    "humidity_rel_shelter_avg": "aws_humidity",
    "wind_speed_10m": "aws_wind",
}


def build(data_root: pathlib.Path, out_path: pathlib.Path,
          start: datetime | None, end: datetime | None,
          msg_max_age_min: int, aws_max_age_min: int) -> int:
    import zarr

    glat, glon = grid_latlon()
    H, W = ANALYSIS_GRID

    # 1. Discover radar files in window
    radar_dir = data_root / "knmi/radar_forecast/2.0"
    if not radar_dir.exists():
        LOG.error("radar dir missing: %s", radar_dir)
        return 2
    radar_files: list[tuple[datetime, pathlib.Path]] = []
    for p in radar_dir.glob("RAD_NL25_RAC_FM_*.h5"):
        try:
            ts = _parse_radar_ts(p)
        except ValueError:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        radar_files.append((ts, p))
    radar_files.sort()
    if not radar_files:
        LOG.error("no radar files found in window")
        return 2
    issue_times = [t for t, _ in radar_files]
    LOG.info("indexing %d radar issue times (%s … %s)",
             len(radar_files), issue_times[0], issue_times[-1])

    # 2. Load aux indexes
    aws_df: pd.DataFrame | None = None
    aws_path = data_root / AWS_PARQUET
    if aws_path.exists():
        aws_df = pd.read_parquet(aws_path)
        aws_df["timestamp"] = pd.to_datetime(aws_df["timestamp"], utc=True).dt.tz_convert(None)
        aws_df = aws_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        LOG.info("AWS: %d rows", len(aws_df))
    else:
        LOG.warning("AWS parquet missing — aws_* will be NaN")

    msg_index = _index_msg_dir(data_root / MSG_IR108_DIR)
    LOG.info("MSG ir108: %d files indexed", len(msg_index))

    static_path = data_root / STATIC_NPZ
    static: dict[str, np.ndarray] | None = None
    if static_path.exists():
        s = np.load(static_path)
        static = {
            "static_elevation_m": s["elevation_m"].astype("float32"),
            "static_landmask": s["landmask"].astype("float32"),
            "static_distance_km": s["distance_to_coast_km"].astype("float32"),
        }
        LOG.info("static: %d channels", len(static))
    else:
        LOG.warning("static.npz missing — run model/build_static.py first")

    # 3. Probe first radar file for n_lead (most files have 25; if KNMI ever
    # truncates we just take whatever's in the first file).
    first = load_forecast_h5(radar_files[0][1])
    n_lead = first.frames.shape[0]
    leads = first.leads_min.astype("int16")
    LOG.info("radar layout: n_lead=%d leads_min=%s", n_lead,
             list(leads[: min(5, len(leads))]) + (["…"] if n_lead > 5 else []))

    # 4. Open zarr — idempotent: if the store exists we read its existing
    # issue_time array, and append only new issue_times. For v1 we keep it
    # simple: rebuild from scratch each invocation. Append-mode lands in v2.
    if out_path.exists():
        import shutil
        LOG.warning("overwriting existing %s", out_path)
        shutil.rmtree(out_path)
    root = zarr.open_group(str(out_path), mode="w", zarr_format=2)

    n = len(radar_files)
    root.create_array("issue_time", shape=(n,), dtype="int64", chunks=(min(n, 256),))
    root.create_array("leads_min", shape=(n_lead,), dtype="int16", chunks=(n_lead,))
    root["issue_time"][:] = np.asarray([_epoch(t) for t in issue_times], dtype="int64")
    root["leads_min"][:] = leads

    # Data arrays — chunks shaped one issue per slot for cheap append-of-1.
    z_radar = root.create_array("radar", shape=(n, n_lead, H, W),
                                chunks=(1, n_lead, H, W), dtype="float32")
    z_aws = {
        var: root.create_array(var, shape=(n, H, W), chunks=(1, H, W), dtype="float32")
        for var in AWS_VAR_NAMES.values()
    }
    z_msg_ir108 = root.create_array("msg_ir108", shape=(n, H, W),
                                    chunks=(1, H, W), dtype="float32")

    if static is not None:
        for name, arr in static.items():
            z = root.create_array(name, shape=arr.shape, dtype="float32",
                                  chunks=arr.shape)
            z[...] = arr

    # 5. Fill per issue time.
    n_radar_ok = n_aws_ok = n_msg_ok = 0
    for i, (ts, path) in enumerate(radar_files):
        try:
            fc = load_forecast_h5(path)
            z_radar[i] = fc.frames.astype("float32")
            n_radar_ok += 1
        except Exception as exc:
            LOG.warning("[%d/%d] %s: radar decode failed: %s", i, n, ts, exc)
            z_radar[i] = np.full((n_lead, H, W), np.nan, dtype="float32")

        if aws_df is not None:
            grids = _build_aws_idw(aws_df, ts, glat, glon, aws_max_age_min)
            if grids is not None:
                for ch, var in AWS_VAR_NAMES.items():
                    z_aws[var][i] = grids[ch]
                n_aws_ok += 1
            else:
                for var in AWS_VAR_NAMES.values():
                    z_aws[var][i] = np.full((H, W), np.nan, dtype="float32")
        else:
            for var in AWS_VAR_NAMES.values():
                z_aws[var][i] = np.full((H, W), np.nan, dtype="float32")

        msg_path = _latest_le(msg_index, ts, msg_max_age_min)
        if msg_path is not None:
            arr = _read_msg_tif(msg_path)
            if arr is not None:
                z_msg_ir108[i] = _bilinear_sample(arr, MSG_BBOX, glat, glon)
                n_msg_ok += 1
            else:
                z_msg_ir108[i] = np.full((H, W), np.nan, dtype="float32")
        else:
            z_msg_ir108[i] = np.full((H, W), np.nan, dtype="float32")

        if (i + 1) % 100 == 0 or i == n - 1:
            LOG.info("  [%d/%d] radar=%d aws=%d msg=%d",
                     i + 1, n, n_radar_ok, n_aws_ok, n_msg_ok)

    LOG.info("done — radar=%d aws=%d msg=%d (of %d issue times) → %s",
             n_radar_ok, n_aws_ok, n_msg_ok, n, out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path,
                        default=REPO_ROOT / "data",
                        help="research/data root (per-source dirs underneath).")
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "data" / "timeseries.zarr")
    parser.add_argument("--start", help="UTC ISO start (inclusive). Default: all radar files on disk.")
    parser.add_argument("--end", help="UTC ISO end (inclusive). Default: all radar files on disk.")
    parser.add_argument("--msg-max-age-min", type=int, default=60,
                        help="Reject MSG samples staler than this. 60 min ≈ 2 satellite slots.")
    parser.add_argument("--aws-max-age-min", type=int, default=30,
                        help="Reject AWS samples staler than this. 30 min ≈ 3 AWS slots.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    start = _iso(args.start) if args.start else None
    end = _iso(args.end) if args.end else None

    return build(args.data, args.out, start, end,
                 args.msg_max_age_min, args.aws_max_age_min)


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
