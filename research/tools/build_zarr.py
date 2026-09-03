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
        msg_ir108 …  msg_rdt    (n, 100, 100)   float32  (NaN where missing)
        alaro_precip … alaro_*  (n, 100, 100)   float32  (NaN where missing)
        static_elevation_m      (100, 100)      float32  (broadcast)
        static_landmask         (100, 100)      float32
        static_distance_km      (100, 100)      float32

The build supports two modes:
  * full rebuild (default) — wipe and re-create from scratch.
  * ``--append`` — read the existing store's ``issue_time``, and write only
    radar issue-times not already present (extending every per-issue array
    along axis 0). Channels added since the last build are back-filled with
    NaN for the pre-existing slots. Lets a daily cron grow the store cheaply.

── On the raster channels (MSG + ALARO) ──────────────────────────────────
Both come from WMS ``GetMap`` as *rendered* GeoTIFFs, not physical values:

  * Grayscale renders (msg ir108/wv062, all ALARO scalar fields): bands 1-3
    are identical luminance → we take **band 1** as the (monotonic) signal.
  * Colour-mapped renders (msg gii_kindex/gii_liftedindex/cth): RGB encodes
    the value through a colour ramp, so band 1 alone is NOT monotonic. We
    still take band 1 as a usable proxy for now; proper colour-map inversion
    (or feeding R,G,B as three channels and letting the convnet decode it) is
    a documented follow-up — see CHANNELS notes.
  * msg rdt: the signal lives in the **alpha** band (0/255 cell mask), read
    as a 0/1 field. Bilinear resampling onto the analysis grid then turns it
    into a [0,1] cell-*coverage* fraction (soft edges) — which is the
    "proximity field" the aux design wanted, not a hard mask.

Undecodable files (a broken/empty raster, or — historically — a WMS
``ServiceExceptionReport`` XML body saved with a ``.tif`` extension) are
tolerated: the slot becomes NaN. The collector no longer writes such bodies
(it pulls with the ``raster`` style and verifies content-type), so all 9
ALARO layers including Surface_CAPE / Mean_sea_level_pressure are in the
registry below.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project paths
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# moved into build() — the truth-backfill path must not drag in
# the aux/collector dependency web (httpx etc.):
# from model.build_aux import AWS_CHANNELS  # noqa: E402
from model.geo import grid_latlon, log_resolved_geometry  # noqa: E402
# moved into build() — the truth-backfill path must not drag in
# the aux/collector dependency web (httpx etc.):
# from notebooks._lib import ANALYSIS_GRID, load_forecast_h5  # noqa: E402

LOG = logging.getLogger("pluvio.build_zarr")

# Sources expected on disk (relative to research/data).
RADAR_GLOB = "knmi/radar_forecast/2.0/RAD_NL25_RAC_FM_*.h5"
# All surface-obs parquets share one schema (timestamp, lat, lon, + AWS_CHANNELS
# fields); they're concatenated into one station pool so KMI(BE) + KNMI(NL) +
# Netatmo(crowd) densify the same aws_* channels rather than adding new ones.
AWS_GLOB = "aws/*.parquet"
STATIC_NPZ = "static.npz"

# WMS bbox the collectors request, EPSG:4326 (minx, miny, maxx, maxy).
# Must match collectors/fetch_eumetsat_msg.py and tools/pull_forward.sh (alaro).
# NB: v1 hard-coded the *old narrow* MSG bbox (2.0,49.0,7.5,52.0); that
# mis-georeferenced every MSG sample. The current pulls use the wide bbox.
MSG_BBOX = (0.0, 48.5, 11.0, 56.0)
ALARO_BBOX = (0.0, 48.5, 11.0, 56.0)
# Copernicus OSTIA SST GeoTIFFs (collectors/fetch_sst.py) — real °C, not rendered.
SST_BBOX = (-1.0, 48.0, 11.0, 57.0)

# Hard upper bound — radar files older than this aren't expected (KNMI
# radar_forecast v2.0 starts 2024-08-14). Used as a sanity check.
EARLIEST_RADAR = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────── channel registry

@dataclasses.dataclass(frozen=True)
class RasterChannel:
    """One reprojected raster aux channel sourced from WMS GeoTIFFs."""
    var: str            # zarr variable name (and the loader's channel name)
    kind: str           # "msg" | "alaro" — selects the on-disk dir layout
    layer: str          # source layer id as it appears in the filename
    bbox: tuple[float, float, float, float]
    max_age_min: int    # reject samples staler than this vs the issue time
    mode: str = "lum"   # "lum" → band 1 raw; "alpha" → last band > 0 as 0/1

    def dir_and_pattern(self, data_root: pathlib.Path) -> tuple[pathlib.Path, str]:
        if self.kind == "msg":
            return data_root / "msg" / f"msg_fes_{self.layer}", "*.tif"
        if self.kind == "sst":
            return data_root / "sst", "sst_*.tif"
        # ALARO is a flat dir; files are alaro_<layer>_<stamp>.tif
        return data_root / "alaro", f"alaro_{self.layer}_*.tif"


# MSG: 60-min max-age ≈ 2 satellite slots (15-min native).
MSG_CHANNELS = [
    RasterChannel("msg_ir108", "msg", "ir108", MSG_BBOX, 60, "lum"),
    RasterChannel("msg_wv062", "msg", "wv062", MSG_BBOX, 60, "lum"),
    RasterChannel("msg_gii_kindex", "msg", "gii_kindex", MSG_BBOX, 60, "lum"),
    RasterChannel("msg_gii_liftedindex", "msg", "gii_liftedindex", MSG_BBOX, 60, "lum"),
    RasterChannel("msg_cth", "msg", "cth", MSG_BBOX, 60, "lum"),
    RasterChannel("msg_rdt", "msg", "rdt", MSG_BBOX, 60, "alpha"),
]

# ALARO: hourly NWP, forward-only. 90-min max-age covers the worst-case gap
# between an issue time and the nearest hourly valid step. All 9 layers are
# pulled with the WMS `raster` style (grayscale), so band-1 luminance is a
# clean monotonic render — see collectors/fetch_alaro_24h.py for why.
ALARO_CHANNELS = [
    RasterChannel("alaro_precip", "alaro", "Total_precipitation", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_cloud", "alaro", "Inst_flx_Tot_Cld_cover", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_wind_u", "alaro", "10_m_u__wind_component", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_wind_v", "alaro", "10_m_v__wind_component", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_rh", "alaro", "2m_Relative_humidity", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_t2m", "alaro", "2_m_temperature", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_td2m", "alaro", "2_m_dewpoint_temperature", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_cape", "alaro", "Surface_CAPE", ALARO_BBOX, 90, "lum"),
    RasterChannel("alaro_mslp", "alaro", "Mean_sea_level_pressure", ALARO_BBOX, 90, "lum"),
]

# SST: daily, gap-free L4; 36h max-age so every issue time finds that day's (or
# the previous day's) field. Real-valued °C GeoTIFF → "lum" reads band 1 as-is.
SST_CHANNELS = [
    # OSTIA publishes day D around D+2 06:00 UTC (measured 2026-09-03): a 36 h
    # window meant every issue was appended before its SST existed and the
    # channel was NaN store-wide. SST changes slowly; 96 h is safe.
    RasterChannel("sst", "sst", "ostia", SST_BBOX, 5760, "lum"),
]

RASTER_CHANNELS = MSG_CHANNELS + ALARO_CHANNELS + SST_CHANNELS


# ──────────────────────────────────────────────────────────── helpers

def _parse_radar_ts(path: pathlib.Path) -> datetime:
    """KNMI radar filename → UTC datetime."""
    # filename ends in _YYYYMMDDHHMM.h5
    stem = path.stem.split("_")[-1]
    return datetime.strptime(stem, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def _parse_tif_ts(path: pathlib.Path) -> datetime:
    """WMS GeoTIFF filename → UTC datetime.

    Works for both MSG (…_YYYYMMDDTHHMMSSZ.tif) and ALARO
    (alaro_<layer>_YYYYMMDDTHHMMSSZ.tif): the timestamp is always the last
    underscore-separated token.
    """
    stem = path.stem.split("_")[-1].rstrip("Z")
    return datetime.strptime(stem, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


# ──────────────────────────────────────────────── raster decode + reproj

def _read_raster_bands(path: pathlib.Path) -> np.ndarray | None:
    """Load a WMS GeoTIFF as a (bands, H, W) float32 array.

    Returns ``None`` if the file can't be decoded (broken/empty file, or a
    WMS XML error body saved with a .tif extension).
    """
    try:
        import rasterio
        try:
            with rasterio.open(path) as src:
                return src.read().astype("float32")  # (bands, H, W)
        except Exception as exc:
            LOG.debug("rasterio failed on %s: %s", path.name, exc)
            return None
    except ImportError:
        # Fallback: PIL. Loses georeferencing but the WMS request fixed the
        # bbox, so it's fine. PIL gives (H, W) or (H, W, bands).
        from PIL import Image
        try:
            arr = np.asarray(Image.open(path)).astype("float32")
        except Exception as exc:
            LOG.debug("PIL failed on %s: %s", path.name, exc)
            return None
        if arr.ndim == 2:
            return arr[None, ...]
        return np.moveaxis(arr, -1, 0)  # (bands, H, W)


def _extract_channel(bands: np.ndarray, mode: str) -> np.ndarray:
    """Reduce a (bands, H, W) rendered raster to a single (H, W) signal."""
    if mode == "alpha":
        # Signal is the alpha (last) band: >0 where a feature is drawn.
        return (bands[-1] > 0).astype("float32")
    # "lum": band 1 of the rendered image (luminance / colour-ramp proxy).
    return bands[0].astype("float32")


def _bilinear_sample(arr: np.ndarray, bbox: tuple[float, float, float, float],
                    lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear sample a regular EPSG:4326 raster at irregular lat/lon points.

    `arr` is (rows, cols); row 0 = north (maxy), col 0 = west (minx) — the
    standard "image" orientation that WMS returns. `lat`/`lon` are the target
    grid in the same CRS.
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

def _idw_knn(lats: np.ndarray, lons: np.ndarray, vals: np.ndarray,
             glat: np.ndarray, glon: np.ndarray, k: int = 8,
             power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting via a KD-tree over the k nearest stations.

    The dense Netatmo pool (~10k stations) makes the naive all-pairs IDW
    (O(H·W·N)) ~140 s per issue time; k-NN brings it to milliseconds. Distances
    are in degrees — fine for IDW over this small mid-latitude domain.
    """
    from scipy.spatial import cKDTree
    pts = np.column_stack([lats, lons])
    tree = cKDTree(pts)
    grid = np.column_stack([glat.ravel(), glon.ravel()])
    kk = min(k, len(vals))
    dist, idx = tree.query(grid, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / (dist ** power + 1e-12)
    g = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
    return g.reshape(glat.shape)


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
    # Latest reading PER STATION within the window (not just the single global
    # max timestamp) — so multiple sources reporting at slightly different times
    # all contribute, and each station counts once.
    frame = (frame.sort_values("timestamp")
                  .drop_duplicates(subset=["lat", "lon"], keep="last"))
    lats = frame["lat"].to_numpy(dtype="float64")
    lons = frame["lon"].to_numpy(dtype="float64")
    from model.build_aux import AWS_CHANNELS  # lazy: see module header

    out: dict[str, np.ndarray] = {}
    for ch, (centre, scale) in AWS_CHANNELS.items():
        vals = frame[ch].to_numpy(dtype="float64")
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            out[ch] = np.full(lat.shape, np.nan, dtype="float32")
            continue
        g = _idw_knn(lats[mask], lons[mask], vals[mask], lat, lon)
        out[ch] = ((g - centre) / scale).astype("float32")
    return out


# ──────────────────────────────────────────────────────────── raster join

def _index_raster_dir(d: pathlib.Path, pattern: str,
                      ) -> list[tuple[datetime, pathlib.Path]]:
    """Sorted (timestamp, path) index of GeoTIFFs matching `pattern` in `d`."""
    if not d.exists():
        return []
    out: list[tuple[datetime, pathlib.Path]] = []
    for p in d.glob(pattern):
        try:
            out.append((_parse_tif_ts(p), p))
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


# ──────────────────────────────────────────────────────────── zarr arrays

def _ensure_per_issue_array(root, name: str, per_issue_shape: tuple,
                            n_total: int, existing_n: int):
    """Return a (n_total, *per_issue_shape) float32 array, creating or
    resizing as needed.

    * fresh build  (existing_n == 0): create at n_total.
    * append       (existing_n  > 0): if the var exists, resize to n_total;
      if it's new (added to the registry since the last build), create it and
      back-fill the pre-existing [:existing_n] slots with NaN.
    """
    shape = (n_total,) + per_issue_shape
    chunks = (1,) + per_issue_shape
    if name in root:
        z = root[name]
        z.resize(shape)
        return z
    z = root.create_array(name, shape=shape, chunks=chunks, dtype="float32")
    if existing_n > 0:
        z[:existing_n] = np.full((existing_n,) + per_issue_shape, np.nan, dtype="float32")
    return z


# ──────────────────────────────────────────────────────────── main build

# Channel naming aligned with what the dataset loader expects.
AWS_VAR_NAMES = {
    "pressure": "aws_pressure",
    "temp_dry_shelter_avg": "aws_temp",
    "humidity_rel_shelter_avg": "aws_humidity",
    "wind_speed_10m": "aws_wind",
}


def _truth_frame(mode: str, ts: datetime, bounds, shape):
    """Training-truth analysis at `ts` from the chosen source, or None.

    v2 curriculum (docs/training_run_v2.md): pretrain against RTCOR (KNMI's
    gauge-adjusted 5-min product, tars 2019->), fine-tune against OUR archived
    composite (/mnt/storagebox/qpe). Kept as a SEPARATE per-issue array so the
    operational-nowcast channel (RAC_FM) is untouched and old stores stay valid.
    """
    stamp = ts.strftime("%Y%m%dT%H%M")
    if mode == "rtcor":
        from tools import knmi_rtcor as kr
        try:
            return kr.rate(stamp, bounds, shape)
        except Exception:
            return None
    if mode == "qpe":
        try:
            import cv2
            import zarr as _z
            day = _z.open_group(
                f"/mnt/storagebox/qpe/{ts:%Y/%m}/{ts:%d}.zarr", mode="r")
            times = np.asarray(day["times"][:], dtype="int64")
            idx = np.where(times == int(ts.timestamp()))[0]
            if idx.size == 0:
                return None
            rate = np.asarray(day["rate"][int(idx[0])], dtype="float32")
            return cv2.resize(rate, (shape[1], shape[0]),
                              interpolation=cv2.INTER_AREA)
        except Exception:
            return None
    return None


def truth_backfill(out_path: pathlib.Path, mode: str, batch: int = 0,
                   shard: str = "") -> int:
    """Fill the 'truth' array for issues ALREADY in the store (newest first).

    The store predates the truth pipeline (35k issues, 2024-08->), so the
    curriculum needs truth written retroactively. Newest-first so a smoke train
    has data within the hour while history fills behind; grouped by day so each
    RTCOR tar downloads once. NaN-marked slots are retried on a later pass;
    slots already finite are skipped, so the job is resumable and idempotent.
    """
    import zarr
    from notebooks._lib import ANALYSIS_GRID  # noqa: F401  (grid consistency)

    root = zarr.open_group(str(out_path), mode="a")
    n = int(root["issue_time"].shape[0])
    H, W = root["radar"].shape[2], root["radar"].shape[3]
    if "truth" not in root.array_keys():
        z = root.create_array("truth", shape=(n, H, W), dtype="float32",
                              chunks=(1, H, W))
        z[:] = np.nan
        LOG.info("created truth array (%d, %d, %d)", n, H, W)
    zt = root["truth"]
    if int(zt.shape[0]) < n:
        zt.resize((n, H, W))
    glat, glon = grid_latlon()
    bounds = (float(glon.min()), float(glat.min()),
              float(glon.max()), float(glat.max()))
    epochs = np.asarray(root["issue_time"][:], dtype="int64")
    order = np.argsort(epochs)[::-1]                  # newest first
    # --shard i/N: the job is network+decode bound (one RTCOR day-tar per ~288
    # slots), so N processes on DISJOINT DAYS write disjoint zarr chunks and
    # scale nearly linearly. Sharding by day keeps each tar owned by one worker.
    if shard:
        i_s, n_s = (int(x) for x in shard.split("/"))
        days = (epochs[order] // 86400)
        order = order[days % n_s == i_s]
        LOG.info("shard %d/%d: %d issues", i_s, n_s, len(order))
    done = skipped = failed = 0
    for i in order:
        if batch and done + failed >= batch:
            break
        if np.isfinite(np.asarray(zt[int(i)])).any():
            skipped += 1
            continue
        ts = datetime.fromtimestamp(int(epochs[i]), tz=timezone.utc)
        tf = _truth_frame(mode, ts, bounds, (H, W))
        if tf is not None:
            zt[int(i)] = tf.astype("float32")
            done += 1
        else:
            failed += 1
        if (done + failed) % 200 == 0:
            LOG.info("truth backfill: %d written, %d unavailable, %d already had",
                     done, failed, skipped)
    LOG.info("truth backfill finished: %d written, %d unavailable, %d already had",
             done, failed, skipped)
    return 0


def build(data_root: pathlib.Path, out_path: pathlib.Path,
          start: datetime | None, end: datetime | None,
          msg_max_age_min: int, aws_max_age_min: int,
          append: bool, truth: str = "none") -> int:
    import zarr

    from model.build_aux import AWS_CHANNELS  # noqa: E402
    from notebooks._lib import ANALYSIS_GRID, load_forecast_h5  # noqa: E402

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

    # 1b. Append mode: open existing store and drop already-present issue times.
    existing_n = 0
    root = None
    if append and out_path.exists():
        root = zarr.open_group(str(out_path), mode="a")
        existing_epochs = set(int(x) for x in root["issue_time"][:])
        existing_n = len(existing_epochs)
        before = len(radar_files)
        radar_files = [(t, p) for (t, p) in radar_files
                       if _epoch(t) not in existing_epochs]
        LOG.info("append: store has %d issue times; %d/%d radar files are new",
                 existing_n, len(radar_files), before)
        if not radar_files:
            LOG.info("nothing new to append — done")
            return 0
    elif append:
        LOG.warning("--append but %s doesn't exist; doing a full build", out_path)
        append = False

    issue_times = [t for t, _ in radar_files]
    LOG.info("indexing %d new radar issue times (%s … %s)",
             len(radar_files), issue_times[0], issue_times[-1])

    # 2. Load aux indexes — concat every surface-obs parquet into one pool.
    aws_df: pd.DataFrame | None = None
    aws_paths = sorted(data_root.glob(AWS_GLOB))
    if aws_paths:
        parts = []
        for p in aws_paths:
            try:
                d = pd.read_parquet(p)
                d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(None)
            except Exception as exc:
                # A collector may be mid-write (parquet writes aren't atomic);
                # skip rather than abort the whole build.
                LOG.warning("AWS source %-28s unreadable, skipping: %s", p.name, exc)
                continue
            parts.append(d)
            LOG.info("AWS source %-28s %d rows", p.name, len(d))
        if parts:
            aws_df = pd.concat(parts, ignore_index=True).dropna(subset=["timestamp"])
            aws_df = aws_df.sort_values("timestamp")
            LOG.info("AWS pool: %d rows from %d source(s)", len(aws_df), len(parts))
        else:
            LOG.warning("no readable AWS parquet — aws_* will be NaN")
    else:
        LOG.warning("no AWS parquet found — aws_* will be NaN")

    # One on-disk index per raster channel.
    raster_index: dict[str, list[tuple[datetime, pathlib.Path]]] = {}
    for ch in RASTER_CHANNELS:
        d, pattern = ch.dir_and_pattern(data_root)
        idx = _index_raster_dir(d, pattern)
        raster_index[ch.var] = idx
        LOG.info("%-22s %6d files indexed (%s)", ch.var, len(idx), d)

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

    # 3. Probe first radar file for n_lead.
    first = load_forecast_h5(radar_files[0][1])
    n_lead = first.frames.shape[0]
    leads = first.leads_min.astype("int16")
    LOG.info("radar layout: n_lead=%d leads_min=%s", n_lead,
             list(leads[: min(5, len(leads))]) + (["…"] if n_lead > 5 else []))

    # 4. Open / extend the zarr.
    k = len(radar_files)            # new issue times to write
    n_total = existing_n + k        # final length along axis 0
    base = existing_n               # write new slots at [base : base+k]

    if not append:
        if out_path.exists():
            import shutil
            LOG.warning("overwriting existing %s", out_path)
            shutil.rmtree(out_path)
        root = zarr.open_group(str(out_path), mode="w", zarr_format=2)
        root.create_array("leads_min", shape=(n_lead,), dtype="int16", chunks=(n_lead,))
        root["leads_min"][:] = leads
        root.create_array("issue_time", shape=(n_total,), dtype="int64",
                          chunks=(min(n_total, 256),))
    else:
        # issue_time grows; leads_min must match.
        if int(root["leads_min"].shape[0]) != n_lead:
            LOG.error("append: existing leads_min (%d) != current (%d) — aborting",
                      int(root["leads_min"].shape[0]), n_lead)
            return 2
        root["issue_time"].resize((n_total,))

    root["issue_time"][base:n_total] = np.asarray(
        [_epoch(t) for t in issue_times], dtype="int64")

    # Per-issue data arrays.
    z_radar = _ensure_per_issue_array(root, "radar", (n_lead, H, W), n_total, base)
    z_truth = (_ensure_per_issue_array(root, "truth", (H, W), n_total, base)
               if truth != "none" else None)
    z_aws = {var: _ensure_per_issue_array(root, var, (H, W), n_total, base)
             for var in AWS_VAR_NAMES.values()}
    z_raster = {ch.var: _ensure_per_issue_array(root, ch.var, (H, W), n_total, base)
                for ch in RASTER_CHANNELS}

    # Static arrays — written once (never per-issue), only on a fresh build.
    if static is not None and not append:
        for name, arr in static.items():
            z = root.create_array(name, shape=arr.shape, dtype="float32", chunks=arr.shape)
            z[...] = arr

    # 5. Fill per (new) issue time.
    n_radar_ok = n_aws_ok = 0
    n_raster_ok = {ch.var: 0 for ch in RASTER_CHANNELS}
    for j, (ts, path) in enumerate(radar_files):
        i = base + j
        try:
            fc = load_forecast_h5(path)
            z_radar[i] = fc.frames.astype("float32")
            n_radar_ok += 1
        except Exception as exc:
            LOG.warning("[%d/%d] %s: radar decode failed: %s", j + 1, k, ts, exc)
            z_radar[i] = np.full((n_lead, H, W), np.nan, dtype="float32")

        if z_truth is not None:
            glat_b = (float(glon.min()), float(glat.min()),
                      float(glon.max()), float(glat.max()))
            tf = _truth_frame(truth, ts, glat_b, (H, W))
            z_truth[i] = (tf.astype("float32") if tf is not None
                          else np.full((H, W), np.nan, dtype="float32"))

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

        for ch in RASTER_CHANNELS:
            max_age = msg_max_age_min if ch.kind == "msg" else ch.max_age_min
            rpath = _latest_le(raster_index[ch.var], ts, max_age)
            grid = None
            if rpath is not None:
                bands = _read_raster_bands(rpath)
                if bands is not None:
                    sig = _extract_channel(bands, ch.mode)
                    grid = _bilinear_sample(sig, ch.bbox, glat, glon)
            if grid is not None:
                z_raster[ch.var][i] = grid
                n_raster_ok[ch.var] += 1
            else:
                z_raster[ch.var][i] = np.full((H, W), np.nan, dtype="float32")

        if (j + 1) % 100 == 0 or j == k - 1:
            rsum = " ".join(f"{v.split('_', 1)[-1]}={n_raster_ok[v]}"
                            for v in list(n_raster_ok)[:3])
            LOG.info("  [%d/%d] radar=%d aws=%d %s …",
                     j + 1, k, n_radar_ok, n_aws_ok, rsum)

    LOG.info("done — wrote %d issue times (total now %d): radar=%d aws=%d",
             k, n_total, n_radar_ok, n_aws_ok)
    for ch in RASTER_CHANNELS:
        LOG.info("    %-22s %d/%d slots have data", ch.var, n_raster_ok[ch.var], k)
    LOG.info("→ %s", out_path)
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
    parser.add_argument("--truth-backfill", action="store_true",
                        help="fill 'truth' for issues already in the store "
                             "(newest first, resumable) and exit")
    parser.add_argument("--truth-shard", default="",
                        help="i/N — process only day-shard i of N (parallel backfill)")
    parser.add_argument("--truth-batch", type=int, default=0,
                        help="max issues per backfill invocation (0 = all)")
    parser.add_argument("--truth", choices=["none", "rtcor", "qpe"], default="none",
                        help="write a per-issue training-truth array from RTCOR "
                             "tars or the QPE archive (docs/training_run_v2.md)")
    parser.add_argument("--append", action="store_true",
                        help="Append only radar issue-times not already in the "
                             "store (extends every per-issue array). Falls back "
                             "to a full build if the store doesn't exist yet.")
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
    log_resolved_geometry()

    start = _iso(args.start) if args.start else None
    end = _iso(args.end) if args.end else None

    if args.truth_backfill:
        if args.truth == "none":
            parser.error("--truth-backfill requires --truth rtcor|qpe")
        return truth_backfill(args.out, args.truth, args.truth_batch, args.truth_shard)
    return build(args.data, args.out, start, end,
                 args.msg_max_age_min, args.aws_max_age_min, args.append,
                 truth=args.truth)


def _iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "T" not in s:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


if __name__ == "__main__":
    sys.exit(main())
