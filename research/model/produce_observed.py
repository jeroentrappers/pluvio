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
BACKFILL_PER_RUN = 4                        # bound each cycle's runtime
MIN_RADARS = 3


def _champion_env():
    os.environ.setdefault("PLUVIO_SWEEP_MERGE", "local")
    os.environ.setdefault("PLUVIO_LOCAL_DH", "1200")


def compose(stamp: str) -> np.ndarray | None:
    """One champion composite on the serving grid, or None if too few radars."""
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
        return None
    rate, _ = rc.composite_by_height(per, OBS_SHAPE)
    LOG.info("  %s: %d radars, wet %.2f%%", stamp, len(per),
             100 * float(np.nanmean(rate > 0.1)))
    return rate.astype("float16")


def wanted_stamps(window_min: int) -> list[str]:
    """The 10-min stamps the window should hold, oldest→newest."""
    now = dt.datetime.now(dt.UTC)
    newest = now - dt.timedelta(minutes=OBS_LAG_MIN)
    newest = newest.replace(minute=(newest.minute // 10) * 10, second=0, microsecond=0)
    out = []
    t = newest - dt.timedelta(minutes=window_min)
    while t <= newest:
        out.append(t.strftime("%Y%m%dT%H%M"))
        t += dt.timedelta(minutes=10)
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
    for f in store.glob("*.npy"):
        if f.stem not in keep:
            f.unlink(missing_ok=True)

    # compute missing frames, newest first, bounded per run
    missing = [s for s in reversed(want) if not (store / f"{s}.npy").exists()]
    for stamp in missing[:BACKFILL_PER_RUN]:
        rate = compose(stamp)
        if rate is not None:
            tmp = store / f".{stamp}.tmp.npy"
            np.save(tmp, rate)
            tmp.replace(store / f"{stamp}.npy")

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
