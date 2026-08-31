"""Produce the observed-rainfall artifact `observed.npz` for the history mode.

The forecast pipeline answers "what is coming"; this answers "what actually fell",
straight from the gauge-validated QPE chain (research/gpu_results/rtcor_replication):
fuzzy dual-pol clutter removal, K_dp attenuation, VPR, quality-weighted sweep merge
(local-1200), height-aware compositing across every radar the feeds carry — the
configuration that ties KNMI RTCOR on rain detection at 1 km.

Latency budget, measured 2026-08-31: KNMI/DWD volumes land ~3-5 min after scan time,
the Belgian radars via the OPERA 24-h cache ~12 min. Each cycle therefore targets the
newest 10-min stamp at least OBS_LAG_MIN old and backfills up to BACKFILL_PER_RUN
missing older stamps, so the rolling window self-heals after outages and fills itself
after a fresh deploy within ~an hour.

Output (atomic tmp→rename, like model_forecast.npz):

    observed.npz
      times   int64   (n,)         epoch seconds, ascending (newest last)
      rates   float16 (n, H, W)    mm/h on the Belgium serving bounds
      bounds  float64 (4,)         [W, S, E, N] — matches backend DEFAULT_BOUNDS
      grid    int64   (2,)         (H, W)

The rolling store keeps one .npy per stamp under --store; frames older than
--window-min are pruned. The serving grid here is ~1 km (416x400 over Belgium) —
independent of PLUVIO_GRID_N, which belongs to the nowcast model and must stay 256.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pathlib
import sys
import tempfile

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("pluvio.produce_observed")

# Serving geometry and radar set are configurable: the product started Belgium-only
# (1.5-7.5E, 48.9-52.5N at ~1 km) and grew to the widest box the verified radar set
# covers. Every radar added beyond the original eight passed the post-a1gate
# verification gate (temporal self-consistency + geographic agreement vs the existing
# composite) before inclusion — see tools/verify_radar.
def _env_tuple(name, default, cast=float):
    raw = os.environ.get(name, "")
    return tuple(cast(x) for x in raw.split(",")) if raw else default

BE_BOUNDS = _env_tuple("PLUVIO_OBS_BOUNDS", (1.5, 48.9, 7.5, 52.5))     # W, S, E, N
OBS_SHAPE = tuple(int(x) for x in _env_tuple("PLUVIO_OBS_SHAPE", (400, 416), int))
RADARS = tuple(os.environ.get(
    "PLUVIO_OBS_RADARS",
    "nlhrw,nldhl,behel,bejab,bewid,deess,denhb,deasb").split(","))
OBS_LAG_MIN = 15                            # newest stamp we dare target
BACKFILL_PER_RUN = int(os.environ.get("PLUVIO_OBS_BACKFILL", "6"))
# Frames younger than this get recomputed when they were built with an incomplete radar
# set — the Belgian files arrive ~12 min late, DWD a few minutes, so completeness for a
# stamp typically settles within half an hour.
UPGRADE_WINDOW_MIN = int(os.environ.get("PLUVIO_OBS_UPGRADE_MIN", "60"))
# Below this the frame LOOKS different from its neighbours (coverage and merge change),
# which reads as flicker in the animation — better a shorter window than an erratic one.
MIN_RADARS = 5


def _champion_env():
    os.environ.setdefault("PLUVIO_SWEEP_MERGE", "local")
    os.environ.setdefault("PLUVIO_LOCAL_DH", "1200")
    # VPR temporal smoothing stays OFF here. Measured 2026-08-31: per-radar fields are
    # frame-to-frame stable WITHOUT state (nlhrw 6.35→6.29% wet across the very pair
    # where the served window ballooned +59%), while the EMA made frames
    # history-dependent — and when the 5-min timer fired during a manual rebuild, two
    # writers interleaved their profile-state sequences and produced frames whose
    # brightness depended on WHICH process computed them. Determinism beats a
    # smoothing whose target flicker was never observed.
    os.environ["PLUVIO_VPR_STATE"] = ""


# Per-radar field cache from the PREVIOUS stamp (runs process chronologically), for
# the two-scan persistence check below. radar -> (stamp, rate_field_unmasked)
_PREV: dict = {}


FILL_MODE = os.environ.get("PLUVIO_OBS_FILL", "comp")
FILL_CACHE = pathlib.Path(os.environ.get("PLUVIO_OBS_FILL_CACHE",
                                         "/opt/pluvio/cache/opera_comp"))
FILL_BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
FILL_LOOKBACK_SLOTS = 3          # <=15 min behind; measured publish latency ~4 min


def _opera_fill(stamp: str):
    """Pan-European OPERA composite as a FILL layer outside our own radar coverage.

    The bucket\'s OPERA/COMP DBZH product is the full 4400x3800 1-km LAEA European
    composite at 5-min cadence (lon -39.6..57.8, lat 31.7..73.9), published ~4 min
    after scan time. Countries that never share single-site volumes through the open
    exchange (UK partially, IE, PL, Nordics, parts of IT/AT) exist ONLY here, so this
    is what turns the served history from a 12-radar region into continental coverage.

    Strictly a fill: wherever our own composite has coverage — including its explicit
    zeros — the own value wins. OPERA pixels are converted dBZ -> rate with the same
    Marshall-Palmer pair as the main chain; the raster\'s internal mask (out of any
    radar\'s reach) stays NaN so uncovered areas render transparent, not dry.
    """
    if FILL_MODE != "comp":
        return None
    import urllib.request

    t0 = dt.datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=dt.UTC)
    path = None
    for k in range(FILL_LOOKBACK_SLOTS + 1):
        t = t0 - dt.timedelta(minutes=5 * k)
        fn = f"OPERA@{t:%Y%m%d}T{t:%H%M}@0@DBZH.tiff"
        cand = FILL_CACHE / fn
        if cand.exists() and cand.stat().st_size > 0:
            path = cand
            break
        FILL_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = FILL_CACHE / (fn + ".part")
        try:
            with urllib.request.urlopen(f"{FILL_BUCKET}/{t:%Y/%m/%d}/OPERA/COMP/{fn}",
                                        timeout=90) as r, open(tmp, "wb") as fh:
                fh.write(r.read())
            tmp.replace(cand)
            path = cand
            break
        except Exception:
            tmp.unlink(missing_ok=True)
    if path is None:
        LOG.warning("no OPERA COMP within %d min of %s — frame served without fill",
                    5 * FILL_LOOKBACK_SLOTS, stamp)
        return None
    # opportunistic cache prune: the bucket only holds 24 h anyway
    cutoff = dt.datetime.now(dt.UTC).timestamp() - 36 * 3600
    for f in FILL_CACHE.glob("OPERA@*.tiff"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        with rasterio.open(path) as src:
            dbz = src.read(1, masked=True)
            tr, crs = src.transform, src.crs
        rate = np.zeros(dbz.shape, "float32")
        wet = ~dbz.mask & (dbz.data > 7.0)              # ~0.1 mm/h Marshall-Palmer
        rate[wet] = (10.0 ** (dbz.data[wet].astype("float32") / 10.0) / 200.0) ** (1.0 / 1.6)
        rate[dbz.mask] = np.nan
        w, sth, e, n = BE_BOUNDS
        dst = np.full(OBS_SHAPE, np.nan, "float32")
        reproject(rate, dst, src_transform=tr, src_crs=crs,
                  dst_transform=from_bounds(w, sth, e, n, OBS_SHAPE[1], OBS_SHAPE[0]),
                  dst_crs=CRS.from_epsg(4326), resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        return dst
    except Exception as exc:
        LOG.warning("OPERA fill failed for %s (%s)", stamp, exc)
        return None


UKMO_BUCKET = "https://met-office-radar-obs-data.s3.eu-west-2.amazonaws.com"
UKMO_CACHE = pathlib.Path(os.environ.get("PLUVIO_OBS_UKMO_CACHE",
                                         "/opt/pluvio/cache/ukmo_comp"))


def _ukmo_fill(stamp: str):
    """UK Met Office national composite as a SECOND fill layer over the British Isles.

    UKMO contributes to neither the open single-site exchange nor the OPERA composite,
    so the pan-EU fill has only the ~35% of the UK that French/Irish/Benelux radars
    reach. Their own composite IS open though: s3://met-office-radar-obs-data (CC
    BY-SA), 1725x2175 at 1 km on an OSGB transverse-Mercator grid covering Britain and
    Ireland, 15-min cadence, measured ~14-min latency. Values are float32 mm/h with -1
    as nodata — no gain/offset dance.

    15-min cadence against our 5-min frames: each frame takes the newest slot at or
    before its stamp (<=30 min back). The late-data upgrade loop re-pulls frames whose
    fill was missing, so a temporarily absent slot heals rather than sticks.
    """
    if FILL_MODE != "comp":
        return None
    import urllib.request

    t0 = dt.datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=dt.UTC)
    t0 -= dt.timedelta(minutes=t0.minute % 15)
    path = None
    for k in range(3):
        t = t0 - dt.timedelta(minutes=15 * k)
        fn = f"{t:%Y%m%d%H%M}_ODIM_ng_radar_rainrate_composite_1km_UK.h5"
        cand = UKMO_CACHE / fn
        if cand.exists() and cand.stat().st_size > 0:
            path = cand
            break
        UKMO_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = UKMO_CACHE / (fn + ".part")
        try:
            with urllib.request.urlopen(
                    f"{UKMO_BUCKET}/radar/{t:%Y/%m/%d}/{fn}", timeout=90) as r,                     open(tmp, "wb") as fh:
                fh.write(r.read())
            tmp.replace(cand)
            path = cand
            break
        except Exception:
            tmp.unlink(missing_ok=True)
    if path is None:
        return None
    cutoff = dt.datetime.now(dt.UTC).timestamp() - 36 * 3600
    for f in UKMO_CACHE.glob("*_UK.h5"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
    try:
        import h5py
        import pyproj
        import rasterio  # noqa: F401  (registers the env for the warp below)
        from rasterio.crs import CRS
        from rasterio.transform import Affine, from_bounds
        from rasterio.warp import Resampling, reproject

        with h5py.File(path, "r") as f:
            w = f["where"].attrs
            arr = f["dataset1"]["data1"]["data"][:].astype("float32")
            projdef = w["projdef"]
            if isinstance(projdef, bytes):
                projdef = projdef.decode()
            ux, uy = pyproj.Proj(projdef)(float(w["UL_lon"]), float(w["UL_lat"]))
            xs, ys = float(w["xscale"]), float(w["yscale"])
        arr[arr < 0] = np.nan                               # -1 = nodata
        src_tr = Affine(xs, 0.0, ux, 0.0, -ys, uy)
        bw, bs, be, bn = BE_BOUNDS
        dst = np.full(OBS_SHAPE, np.nan, "float32")
        reproject(arr, dst, src_transform=src_tr, src_crs=CRS.from_proj4(projdef),
                  dst_transform=from_bounds(bw, bs, be, bn, OBS_SHAPE[1], OBS_SHAPE[0]),
                  dst_crs=CRS.from_epsg(4326), resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        # Trust the national composite only over the British Isles: outside that box
        # its far-range view loses to OPERA\'s continental radars.
        lon = np.linspace(bw, be, OBS_SHAPE[1])[None, :]
        lat = np.linspace(bn, bs, OBS_SHAPE[0])[:, None]
        dst[~((lon <= 2.2) & (lat >= 49.5))] = np.nan
        return dst
    except Exception as exc:
        LOG.warning("UKMO fill failed for %s (%s)", stamp, exc)
        return None


def _persistence_filter(radar, stamp, rate, shape, bounds):
    """Two-scan confirmation for radars without dual-pol moments.

    Measured over Flanders (bejab's home footprint): bejab's wet mask flips 99% of its
    cells between consecutive scans at 0.3 mm/h — statistically uncorrelated, i.e.
    noise, while the dual-pol-QC'd nlhrw flips 52% on the same convective day. With no
    polarimetric variables to classify on, the physical discriminator left is TIME:
    rain persists and advects, noise decorrelates. Echo must appear in this scan AND
    (within 8 km, the 100 km/h motion bound) in the previous one; uncorrelated noise
    survives as p-squared, real moving rain survives the dilation. Costs one scan of
    onset delay for genuinely new cells. Display chain only — the QPE archive keeps
    the unfiltered physics.
    """
    from scipy import ndimage
    from tools import rtcor_chain as rc
    from tools.radar_single_site import polar_to_grid

    prev = _PREV.get(radar)
    want_prev = (dt.datetime.strptime(stamp, "%Y%m%dT%H%M")
                 - dt.timedelta(minutes=5)).strftime("%Y%m%dT%H%M")
    if prev is None or prev[0] != want_prev:
        try:
            sw = rc.read_sweeps_any(radar, want_prev)
            prev_rate = (rc.single_radar_h(sw, shape, bounds, polar_to_grid)[0]
                         if sw else None)
        except Exception:
            prev_rate = None
    else:
        prev_rate = prev[1]
    _PREV[radar] = (stamp, rate.copy())
    if prev_rate is None:
        return rate                      # nothing to confirm against — pass through
    confirmed = ndimage.binary_dilation(
        np.nan_to_num(prev_rate, nan=0.0) > 0.1, iterations=8)
    return np.where((np.nan_to_num(rate, nan=0.0) > 0.1) & ~confirmed, 0.0, rate)


def compose(stamp: str):
    """One champion composite on the serving grid -> (rate | None, n_radars).

    The radar count travels with the frame: frames built before every radar's file
    arrived get RECOMPUTED on later runs (see main), because mixing frames of varying
    completeness is exactly what made regions blink in and out of the served history —
    measured: stored wet-fraction swung 8.4 ↔ 12.5% between adjacent 5-min frames while
    deterministic full-set recomputes of the same stamps read 7.8→8.5→8.5→8.3→10.1.
    """
    from tools.radar_single_site import polar_to_grid
    from tools import rtcor_chain as rc

    per = []
    for r in RADARS:
        try:
            sw = rc.read_sweeps_any(r, stamp)
            if sw:
                rate, q, h = rc.single_radar_h(sw, OBS_SHAPE, BE_BOUNDS, polar_to_grid)
                if sw[0].get("rhohv") is None:      # no dual-pol: persistence QC
                    rate = _persistence_filter(r, stamp, rate, OBS_SHAPE, BE_BOUNDS)
                per.append((rate, q, h))
        except Exception as exc:            # a broken radar must not sink the frame
            LOG.debug("%s unusable at %s (%s)", r, stamp, exc)
    if len(per) < MIN_RADARS:
        LOG.warning("only %d radars at %s — skipping frame", len(per), stamp)
        return None, len(per), False
    rate, _ = rc.composite_by_height(per, OBS_SHAPE)
    # temporal consistency: isolated single-cell blinkers dominate the perceived
    # noise between frames; the validated speckle filter removes them.
    rate = rc.speckle(rate)
    rate = _despeckle_area(rate)
    fill = _opera_fill(stamp)
    ukmo = _ukmo_fill(stamp)
    if ukmo is not None:                    # British Isles: national composite wins
        fill = ukmo if fill is None else np.where(np.isfinite(ukmo), ukmo, fill)
    if fill is not None:
        gap = ~np.isfinite(rate) & np.isfinite(fill)
        rate = np.where(gap, fill, rate)
    tags = ("" if fill is None else " +fill") + ("" if ukmo is None else "+uk")
    LOG.info("  %s: %d radars%s, wet %.2f%%", stamp, len(per), tags,
             100 * float(np.nanmean(rate > 0.1)))
    fill_ok = (FILL_MODE != "comp") or (fill is not None and ukmo is not None)
    return rate.astype("float16"), len(per), fill_ok


def _despeckle_area(rate, min_cells: int = 8, core_mm_h: float = 1.0):
    """Remove tiny trace-only echo clusters — display-grade physical QC.

    The 3x3 speckle filter spans 81 km2 on the 3 km research grid but only 9 km2 on
    this ~1 km serving grid, so 4-8-cell clusters slip through and flicker frame to
    frame — measured as hundreds of scattered gain components per 5-min step, centred
    on the Belgian radars (the only ones without dual-pol QC, and the ones whose
    calibration harmonisation lifted the noise floor). A rain feature smaller than
    ~8 km2 that nowhere reaches core_mm_h is noise; a small INTENSE cell (young
    convection) is kept. The permanent QPE archive stays unfiltered.
    """
    from scipy import ndimage

    wet = np.nan_to_num(rate, nan=0.0) > 0.1
    labels, n = ndimage.label(wet)
    if not n:
        return rate
    sizes = np.bincount(labels.ravel())
    maxes = ndimage.maximum(np.nan_to_num(rate, nan=0.0), labels, np.arange(1, n + 1))
    kill_ids = [i + 1 for i in range(n)
                if sizes[i + 1] < min_cells and maxes[i] < core_mm_h]
    if kill_ids:
        kill = np.isin(labels, kill_ids)
        rate = np.where(kill, 0.0, rate)
    return rate


MAX_FLOW_PX = 8.0        # ~100 km/h over 5 min at ~1 km pixels — nothing real is faster


def _flow(prev, cur):
    """Farneback optical flow prev->cur, made robust for radar fields.

    Raw Farneback on a speckly, mostly-flat rain field produces garbage vectors that
    remap then acts on — measured: interpolants with wet-area jumps LARGER than the
    scans they sit between (10.2 pp per 100 s). Three standard remedies: blur the
    input (flow sees structure, not speckle), smooth the flow field, and clamp
    magnitudes to physically possible cell motion.
    """
    import cv2

    a = np.nan_to_num(prev.astype("float32"), nan=0.0)
    b = np.nan_to_num(cur.astype("float32"), nan=0.0)
    fa = ((np.log10(a + 0.05) + 1.4) * 60.0).clip(0, 255).astype("uint8")
    fb = ((np.log10(b + 0.05) + 1.4) * 60.0).clip(0, 255).astype("uint8")
    fa = cv2.GaussianBlur(fa, (0, 0), 2.5)
    fb = cv2.GaussianBlur(fb, (0, 0), 2.5)
    flow = cv2.calcOpticalFlowFarneback(fa, fb, None, 0.5, 4, 35, 3, 7, 1.5, 0)
    flow[..., 0] = cv2.blur(flow[..., 0], (21, 21))
    flow[..., 1] = cv2.blur(flow[..., 1], (21, 21))
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    scale = np.where(mag > MAX_FLOW_PX, MAX_FLOW_PX / np.maximum(mag, 1e-6), 1.0)
    flow[..., 0] *= scale
    flow[..., 1] *= scale
    return flow


def _interpolate(prev, cur, flow, f):
    """One Eq.-4 cross-faded interpolant at fraction f of the way prev->cur.

    Every SCAN stays exact in the served sequence; only the in-between frames are
    synthesized. This is the substance of the reference animations' smoothness (the
    processing-cdn PNGs are frames of an interpolated video): at 1 km pixels a cell
    moves 2-3 px per 5-min scan, so raw playback teleports it — interpolants make it
    slide. Accumulating instead of interpolating was tried first and REJECTED: painting
    the motion track fragments the wet contour (gain components 600 -> 797) and halves
    peak rates, the same stationary-pixel trap as the temporal median.
    """
    import cv2

    # Semi-Lagrangian nearest-scan advection, NOT a cross-fade. Cross-fading two rain
    # fields makes every interpolant a weighted UNION of both wet masks: where a cell
    # moved, ghost rain at 33-67% weight appears mid-gap and the next exact scan
    # deletes it — a sawtooth measured at 5 pp median wet-delta per 100 s (raw scans:
    # 1.4 pp per 300 s). Advecting the nearest scan keeps every frame a plausible
    # displaced field; the handover at half-way happens where the two scans align
    # best, because each has been advected exactly half the motion.
    a = np.nan_to_num(prev.astype("float32"), nan=0.0)
    b = np.nan_to_num(cur.astype("float32"), nan=0.0)
    h, w = a.shape
    gy, gx = np.mgrid[0:h, 0:w].astype("float32")
    if f < 0.5:
        out = cv2.remap(a, gx - flow[..., 0] * f, gy - flow[..., 1] * f,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    else:
        out = cv2.remap(b, gx + flow[..., 0] * (1.0 - f), gy + flow[..., 1] * (1.0 - f),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.where(np.isfinite(prev) | np.isfinite(cur), out, np.nan)


def wanted_stamps(window_min: int) -> list[str]:
    """The 10-min stamps the window should hold, oldest→newest."""
    now = dt.datetime.now(dt.UTC)
    newest = now - dt.timedelta(minutes=OBS_LAG_MIN)
    # 5-min cadence: every feed we composite is 5-min native, and 10-min stepping
    # reads as jumpy motion (cells move 3-6 km per step at 1 km pixels).
    newest = newest.replace(minute=(newest.minute // 5) * 5, second=0, microsecond=0)
    out = []
    t = newest - dt.timedelta(minutes=window_min)
    while t <= newest:
        out.append(t.strftime("%Y%m%dT%H%M"))
        t += dt.timedelta(minutes=5)
    return out


def _exclusive_lock():
    """One producer at a time. A timer tick and a manual backfill running together
    interleave frame computation (measured: the +59% wet-area balloon), so every run
    takes this lock or exits."""
    import fcntl
    fh = open("/run/pluvio-observed.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        LOG.info("another producer run holds the lock — exiting")
        raise SystemExit(0)
    return fh


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="/opt/pluvio/serve/observed_frames")
    p.add_argument("--out", default="/opt/pluvio/serve/observed.npz")
    p.add_argument("--window-min", type=int, default=180)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _champion_env()
    _lock = _exclusive_lock()  # noqa: F841 — held for process lifetime

    store = pathlib.Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    want = wanted_stamps(args.window_min)

    # prune frames that fell out of the window
    keep = set(want)
    for f in list(store.glob("*.npy")) + list(store.glob("*.json")):
        if f.stem not in keep:
            f.unlink(missing_ok=True)

    # Work list: missing frames plus recent frames that were built incomplete (a late
    # radar file upgrades them). CHRONOLOGICAL order — the VPR temporal smoothing is an
    # EMA and must see volumes in time order to be causal.
    n_full = len(RADARS)
    now = dt.datetime.now(dt.UTC)
    work = []
    for stamp in want:                                   # oldest -> newest
        f = store / f"{stamp}.npy"
        meta = store / f"{stamp}.json"
        if not f.exists():
            work.append(stamp)
            continue
        try:
            import json as _json
            m = _json.loads(meta.read_text())
            nrad = m.get("n_radars", n_full)
            had_fill = m.get("fill", True)                # legacy sidecars: assume yes
        except Exception:
            nrad, had_fill = 0, False                     # unknown provenance: rebuild
        age_min = (now - dt.datetime.strptime(stamp, "%Y%m%dT%H%M")
                   .replace(tzinfo=dt.UTC)).total_seconds() / 60
        if (nrad < n_full or (FILL_MODE and not had_fill)) \
                and age_min <= UPGRADE_WINDOW_MIN:
            work.append(stamp)
    for stamp in work[:BACKFILL_PER_RUN]:
        rate, nrad, had_fill = compose(stamp)
        if rate is not None:
            import json as _json
            tmp = store / f".{stamp}.tmp.npy"
            np.save(tmp, rate)
            tmp.replace(store / f"{stamp}.npy")
            (store / f"{stamp}.json").write_text(
                _json.dumps({"n_radars": nrad, "fill": had_fill}))

    frames = sorted(store.glob("*.npy"))
    if not frames:
        LOG.error("no observed frames available")
        return 1
    times, rates = [], []
    for f in frames:
        try:
            arr = np.load(f)
        except Exception:
            continue
        if arr.shape != OBS_SHAPE:
            f.unlink(missing_ok=True)       # grid changed — stale frame
            continue
        times.append(int(dt.datetime.strptime(f.stem, "%Y%m%dT%H%M")
                         .replace(tzinfo=dt.UTC).timestamp()))
        rates.append(arr)
    # Served sequence: every scan exact, plus two motion-compensated interpolants per
    # 5-min gap (~100 s visual cadence). Raw frames and the QPE archive stay untouched.
    out_t, out_r = [times[0]], [rates[0].astype("float16")]
    for i in range(1, len(rates)):
        prev, cur = rates[i - 1].astype("float32"), rates[i].astype("float32")
        if times[i] - times[i - 1] == 300:
            fl = _flow(prev, cur)
            for f in (1.0 / 3.0, 2.0 / 3.0):
                out_t.append(int(times[i - 1] + f * 300))
                out_r.append(_interpolate(prev, cur, fl, f).astype("float16"))
        out_t.append(times[i])
        out_r.append(rates[i].astype("float16"))
    times, rates = out_t, out_r
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    np.savez(tmp,
             times=np.asarray(times, dtype="int64"),
             rates=np.stack(rates),
             bounds=np.asarray(BE_BOUNDS, dtype="float64"),
             grid=np.asarray(OBS_SHAPE, dtype="int64"))
    tmp.replace(out)
    out.chmod(0o644)
    LOG.info("wrote %s: %d frames, %s → %s", out, len(times),
             dt.datetime.fromtimestamp(times[0], dt.UTC).strftime("%H:%M"),
             dt.datetime.fromtimestamp(times[-1], dt.UTC).strftime("%H:%M"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
