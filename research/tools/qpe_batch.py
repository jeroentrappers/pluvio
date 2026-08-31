"""Parallel batch computation of champion composites over many timestamps.

Every evaluation so far ran single-process: ~7 s per slot at 1 km (CIFS HDF5 reads,
fuzzy textures, 21 scatter-grids per radar), 288 slots per day — 35 minutes a day on a
16-core box using one core. Slots are independent, so this fans them out over a
process pool. Geometry caches are per-process: each worker pays one ~10 s warm-up and
then reuses it, so workers should be few and long-lived, not one-per-slot.

Returns float16 rate fields (1.2 MB each at 768) keyed by stamp; quality is sampled to
gauge points inside the worker to avoid shipping a second full grid per slot.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ENV_KEYS = ("PLUVIO_GRID_N", "PLUVIO_SWEEP_MERGE", "PLUVIO_LOCAL_DH", "PLUVIO_GABELLA")


def _worker_init(env):
    for k, v in env.items():
        os.environ[k] = v


def _one(args):
    stamp, radars, gauge_pts = args
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/opt/pluvio/radarproc")
    from model.geo import GRID, bbox
    from tools.radar_single_site import polar_to_grid
    from tools.gauge_validate import sample
    from tools import rtcor_chain as rc, knmi_volume as kv

    bounds = bbox()
    per = []
    for r in radars:
        p = kv.fetch(r, stamp)
        if p is None:
            continue
        try:
            sw = kv.read_all_sweeps(p, max_elangle=6.0)
            if sw:
                per.append(rc.single_radar_h(sw, GRID, bounds, polar_to_grid))
        except Exception:
            pass
    if not per:
        return stamp, None, None
    rate, q = rc.composite_by_height(per, GRID)
    q_at = [float(np.nan_to_num(sample(q, la, lo, bounds, GRID, halo=0), nan=0.0))
            for la, lo in gauge_pts] if gauge_pts else []
    return stamp, rate.astype("float16"), q_at


def composite_many(stamps, radars, gauge_pts=None, workers=8):
    """{stamp: rate float16} plus {stamp: [quality at gauge_pts]} for the given stamps."""
    env = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
    rates, quals = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(env,)) as ex:
        for stamp, rate, q_at in ex.map(_one, [(s, list(radars), gauge_pts or [])
                                               for s in stamps], chunksize=4):
            if rate is not None:
                rates[stamp] = rate
                quals[stamp] = q_at
    return rates, quals
