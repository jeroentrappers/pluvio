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

# Backend serving bounds (backend cache.DEFAULT_BOUNDS) at ~1 km.
BE_BOUNDS = (1.5, 48.9, 7.5, 52.5)          # W, S, E, N
OBS_SHAPE = (400, 416)                      # (H, W): 3.6 deg lat, 6.0 deg lon
RADARS = ("nlhrw", "nldhl", "behel", "bejab", "bewid", "deess", "denhb", "deasb")
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
    # temporal smoothing of the per-radar VPR — see rtcor_chain.estimate_vpr
    os.environ.setdefault("PLUVIO_VPR_STATE", "/opt/pluvio/serve/observed_state")


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
                per.append(rc.single_radar_h(sw, OBS_SHAPE, BE_BOUNDS, polar_to_grid))
        except Exception as exc:            # a broken radar must not sink the frame
            LOG.debug("%s unusable at %s (%s)", r, stamp, exc)
    if len(per) < MIN_RADARS:
        LOG.warning("only %d radars at %s — skipping frame", len(per), stamp)
        return None, len(per)
    rate, _ = rc.composite_by_height(per, OBS_SHAPE)
    # temporal consistency: isolated single-cell blinkers dominate the perceived
    # noise between frames; the validated speckle filter removes them.
    rate = rc.speckle(rate)
    LOG.info("  %s: %d radars, wet %.2f%%", stamp, len(per),
             100 * float(np.nanmean(rate > 0.1)))
    return rate.astype("float16"), len(per)


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
            nrad = _json.loads(meta.read_text()).get("n_radars", n_full)
        except Exception:
            nrad = 0                                      # unknown provenance: rebuild
        age_min = (now - dt.datetime.strptime(stamp, "%Y%m%dT%H%M")
                   .replace(tzinfo=dt.UTC)).total_seconds() / 60
        if nrad < n_full and age_min <= UPGRADE_WINDOW_MIN:
            work.append(stamp)
    for stamp in work[:BACKFILL_PER_RUN]:
        rate, nrad = compose(stamp)
        if rate is not None:
            import json as _json
            tmp = store / f".{stamp}.tmp.npy"
            np.save(tmp, rate)
            tmp.replace(store / f"{stamp}.npy")
            (store / f"{stamp}.json").write_text(_json.dumps({"n_radars": nrad}))

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
