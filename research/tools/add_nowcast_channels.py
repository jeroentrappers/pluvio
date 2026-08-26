"""Append optical-flow advection + growth/decay tendency channels to a seamless zarr.

Adds two derived inputs to an *existing* store (built by tools/build_seamless_zarr.py),
computed purely from `opera_rate` — no /mnt/storagebox dependency, so it runs on the GPU
node against the training zarr directly:

    oflow_rate    (n, n_now, H, W) f16  LK-advected radar at each nowcast lead (0..120 min)
    oflow_leads   (n_now,)         i16  the nowcast lead axis for oflow_rate
    rate_tendency (n, H, W)        f16  per-pixel OLS slope of log1p(rate) over the recent window

Why: the c15 model loses FSS to optical-flow because it must learn advection implicitly from
raw frames. Feeding the LK-advected field lets the net learn a *residual* on a sharp,
correctly-displaced prior (→ closes the FSS gap); the tendency carries the growth/decay signal
advection can't. The LK motion uses a longer window (~10 frames / 150 min) than the 6 raw frames
the model sees — a larger runway consumed by the motion estimator, not the input stack.

Reuses `model.classical.optical_flow_nowcast` (the same pysteps LK engine that is the eval bar,
so the prior is consistent with the baseline). Per-issue failures fall back to 0-motion
persistence — never abort the whole build.

    python -m tools.add_nowcast_channels --zarr nowcast_mm_c15_0724_v2.zarr        # full
    python -m tools.add_nowcast_channels --zarr <z> --limit 200                    # test slice
    python -m tools.add_nowcast_channels --zarr <z> --force                        # recompute
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.classical import optical_flow_nowcast  # noqa: E402

LOG = logging.getLogger("pluvio.add_nowcast_channels")

NOWCAST_MAX_MIN = 120        # advection prior only over the radar-skilful regime
MOTION_WINDOW = 10           # frames fed to the LK motion estimate (~150 min @ 15-min cadence)
TENDENCY_WINDOW = 6          # frames for the log1p-rate trend (~90 min)
HISTORY_TOL_S = 300          # a frame counts as "K steps back" within ±5 min

_G: dict = {}                # per-worker state (zarr handles), keyed after fork



def _create(root, name, **kw):
    """Create an array, working on both zarr major versions.

    zarr 2 exposes ``Group.create_dataset``; zarr 3 replaced it with
    ``Group.create_array`` and removed the old name. The two tools in this pipeline
    were written against different majors — build_seamless_zarr.py uses
    create_array (zarr 3), this used create_dataset (zarr 2) — which is why the
    training flow needed tools/zarr_v3_to_v2.py wedged between them. Serving runs
    both steps in ONE image, so dispatch instead of pinning: the serving path then
    needs no v3→v2 conversion at all.
    """
    fn = getattr(root, "create_array", None) or root.create_dataset
    return fn(name, **kw)


def _sorted_history(issue_epoch: np.ndarray, step_s: int):
    """For each issue index, the list of prior issue indices (oldest→newest, incl. self)
    that form a contiguous ~step_s cadence chain, up to MOTION_WINDOW long. Chain breaks
    at any gap > step_s + tolerance (radar outage / sequence start)."""
    order = np.argsort(issue_epoch)
    inv = {int(orig): p for p, orig in enumerate(order)}
    sorted_e = issue_epoch[order]
    hist: list[list[int]] = [[] for _ in range(len(issue_epoch))]
    for orig in range(len(issue_epoch)):
        p = inv[orig]
        chain = [order[p]]
        for q in range(p - 1, max(-1, p - MOTION_WINDOW), -1):
            if abs(int(sorted_e[q + 1]) - int(sorted_e[q]) - step_s) <= HISTORY_TOL_S:
                chain.append(order[q])
            else:
                break
        hist[orig] = [int(x) for x in reversed(chain)]  # oldest→newest
    return hist


def _tendency(window: np.ndarray) -> np.ndarray:
    """Per-pixel OLS slope of log1p(rate) over the last TENDENCY_WINDOW frames (per frame).
    window: (T, H, W) oldest→newest. >0 = intensifying, <0 = decaying."""
    w = window[-TENDENCY_WINDOW:]
    t = w.shape[0]
    if t < 2:
        return np.zeros(window.shape[1:], dtype="float32")
    y = np.log1p(np.maximum(w, 0.0))                    # (t, H, W)
    ts = np.arange(t, dtype="float32")
    tc = ts - ts.mean()
    denom = float((tc ** 2).sum())
    num = np.tensordot(tc, y - y.mean(axis=0), axes=(0, 0))  # (H, W)
    return (num / denom).astype("float32")


def _process(task):
    """Worker: compute oflow_rate[i] (n_now,H,W) and rate_tendency[i] (H,W) for one issue."""
    i, hist_idx = task
    root = _G["root"]
    opera = _G["opera"]
    leads = _G["leads"]          # nowcast leads list
    dt_min = _G["dt_min"]
    H, W = opera.shape[1:]
    window = np.stack([np.nan_to_num(np.asarray(opera[h]), nan=0.0) for h in hist_idx]).astype("float32")
    last = window[-1]
    # advection prior
    try:
        if window.shape[0] >= 2:
            rates, _ = optical_flow_nowcast(window, leads, dt_min=dt_min, prefer_pysteps=True)
        else:
            raise ValueError("history < 2 frames")
    except Exception as exc:  # singular LK on near-dry frames, etc. → persistence
        LOG.debug("issue %d: OF fallback (%s)", i, exc)
        rates = np.repeat(last[None], len(leads), axis=0)
    rates = np.nan_to_num(rates, nan=0.0).clip(0.0, None).astype("float16")
    tend = _tendency(window).astype("float16")
    _G["oflow"][i] = rates
    _G["tend"][i] = tend
    return i


def _init_worker(zarr_path, leads, dt_min):
    import zarr
    root = zarr.open_group(zarr_path, mode="r+")
    _G.update(root=root, opera=root["opera_rate"], oflow=root["oflow_rate"],
              tend=root["rate_tendency"], leads=list(leads), dt_min=dt_min)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--limit", type=int, default=None, help="process only the first N issues (test)")
    p.add_argument("--force", action="store_true", help="recompute even if arrays already exist")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import zarr
    root = zarr.open_group(args.zarr, mode="r+")
    issue_epoch = np.asarray(root["issue_time"][:], dtype="int64")
    n, (H, W) = len(issue_epoch), root["opera_rate"].shape[1:]
    zarr_leads = [int(x) for x in np.asarray(root["leads_min"][:])]
    now_leads = [l for l in zarr_leads if 0 <= l <= NOWCAST_MAX_MIN]
    gaps = np.diff(np.sort(issue_epoch))
    step_s = int(np.median(gaps)) if len(gaps) else 900
    dt_min = max(1, round(step_s / 60))
    LOG.info("zarr=%s n=%d grid=%dx%d nowcast_leads=%s dt=%dmin", args.zarr, n, H, W, now_leads, dt_min)

    existing = set(root.array_keys())
    if {"oflow_rate", "rate_tendency"} <= existing and not args.force:
        LOG.error("oflow_rate/rate_tendency already exist — use --force to recompute. Aborting.")
        return 1
    # (re)create the derived arrays (zarr v2 API; overwrite replaces any prior version).
    # chunks=(1, ...) → each issue is its own chunk, so worker processes write disjoint
    # chunks concurrently with no locking.
    _create(root, "oflow_leads", shape=(len(now_leads),), dtype="int16",
                        overwrite=True)[:] = now_leads
    for name, shape, chunks in (
        ("oflow_rate", (n, len(now_leads), H, W), (1, len(now_leads), H, W)),
        ("rate_tendency", (n, H, W), (1, H, W)),
    ):
        _create(root, name, shape=shape, chunks=chunks, dtype="float16", overwrite=True)

    hist = _sorted_history(issue_epoch, step_s)
    idxs = list(range(n if args.limit is None else min(args.limit, n)))
    tasks = [(i, hist[i]) for i in idxs]
    LOG.info("computing %d issues on %d workers …", len(tasks), args.workers)
    done = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.zarr, now_leads, dt_min)) as pool:
        for _ in pool.imap_unordered(_process, tasks, chunksize=16):
            done += 1
            if done % 1000 == 0:
                LOG.info("  %d/%d", done, len(tasks))
    LOG.info("done: added oflow_rate%s, rate_tendency%s, oflow_leads(%d) to %s",
             (n, len(now_leads), H, W), (n, H, W), len(now_leads), args.zarr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
