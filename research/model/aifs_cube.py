"""Build a raw-AIFS precip cube on the analysis grid for the outlook regime.

`produce_forecast._aifs_cube` looks for `build_aifs_cube` here. It turns the
AIFS open-data GRIBs the collector lands (`collectors/fetch_aifs_opendata.py` →
`aifs-single_tp_<run>_+<step>h_fc.grib2`, `tp` accumulated from step 0) into a
``(n_lead, H, W)`` mm/h cube aligned to the requested forecast leads — the NWP
anchor the classical seamless blend uses past the radar horizon, and the input
the learned outlook head downscales.

Mapping is the only subtle bit: each requested lead has a *valid time*
(issue + lead); AIFS forecasts a *step* from its run time. We pick the 6-hourly
step whose accumulation interval (s-Δ, s] contains the valid time, difference
consecutive accumulated-`tp` fields, and divide by the interval hours → mm/h.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import pathlib
import re

import numpy as np

LOG = logging.getLogger("pluvio.aifs_cube")

_FC_RE = re.compile(r"_(\d{8}T\d{2}Z)_\+(\d+)h_")


def interval_step(hours: float, step_h: int, max_step: int) -> int:
    """The accumulation step whose (step-step_h, step] window covers ``hours``
    after the run. Clamped to the run's coverage: leads inside the first window
    use the first step; leads past the horizon carry the final step."""
    if hours <= 0:
        return step_h
    s = math.ceil(hours / step_h) * step_h
    return int(max(step_h, min(s, max_step)))


def _bracketing_steps(hours: float, steps: list[int], step_h: int) -> tuple[int, int, float]:
    """For temporal interpolation: the two AIFS steps whose interval *centres*
    bracket ``hours``, plus the blend weight w (0 → first step, 1 → second).

    Each step s carries the mean rate over (s-step_h, s], which we attribute to
    the interval centre (s - step_h/2). Interpolating between adjacent centres
    turns AIFS's coarse 6-hourly rate into a smooth curve at our finer serve
    cadence — otherwise six consecutive hourly frames are byte-identical and the
    forecast *freezes* on screen instead of evolving. Clamped at both ends.
    """
    centres = [s - step_h / 2.0 for s in steps]
    if hours <= centres[0]:
        return steps[0], steps[0], 0.0
    if hours >= centres[-1]:
        return steps[-1], steps[-1], 0.0
    j = next(k for k in range(1, len(centres)) if centres[k] >= hours)
    w = (hours - centres[j - 1]) / (centres[j] - centres[j - 1])
    return steps[j - 1], steps[j], float(w)


def _scan_runs(aifs_dir: pathlib.Path, model: str, param: str) -> dict[str, dict[int, pathlib.Path]]:
    """{run_id: {step_h: path}} for the deterministic (`fc`) GRIBs present."""
    runs: dict[str, dict[int, pathlib.Path]] = {}
    for p in aifs_dir.glob(f"{model}_{param}_*_+*h_fc.grib2"):
        m = _FC_RE.search(p.name)
        if not m:
            continue
        runs.setdefault(m.group(1), {})[int(m.group(2))] = p
    return runs


def build_aifs_cube(aifs_dir, leads_min, issue_dt, *, model: str = "aifs-single",
                    param: str = "tp") -> np.ndarray:
    """``(n_lead, H, W)`` raw-AIFS mean precip rate (mm/h) on the analysis grid.

    Args:
        aifs_dir: dir holding the AIFS GRIBs (the collector's --out).
        leads_min: forecast lead minutes (same order as the returned axis).
        issue_dt: the forecast issue time (tz-aware UTC); valid = issue + lead.

    Uses the latest run at or before ``issue_dt``. Leads outside that run's
    coverage are NaN (the caller zero-fills; the radar nowcast covers the short
    end, so this only bites the very tail).
    """
    from model.geo import grid_latlon
    from model.nwp_regrid import open_tp, regrid_to

    aifs_dir = pathlib.Path(aifs_dir)
    runs = _scan_runs(aifs_dir, model, param)
    if not runs:
        raise RuntimeError(f"no AIFS {model}/{param} fc GRIBs under {aifs_dir}")

    # Latest run not after the issue time (run_id sorts chronologically).
    usable = [r for r in runs if dt.datetime.strptime(r, "%Y%m%dT%HZ").replace(tzinfo=dt.UTC) <= issue_dt]
    if not usable:
        raise RuntimeError(f"no AIFS run at/before issue {issue_dt.isoformat()}")
    run = max(usable)
    run_dt = dt.datetime.strptime(run, "%Y%m%dT%HZ").replace(tzinfo=dt.UTC)
    steps = sorted(runs[run])
    step_h = min((b - a for a, b in zip(steps, steps[1:])), default=steps[0]) if len(steps) > 1 else steps[0]
    max_step = steps[-1]
    LOG.info("AIFS run %s: %d steps %d…%dh (Δ%dh)", run, len(steps), steps[0], max_step, step_h)

    glat, glon = grid_latlon()
    H, W = glat.shape

    tp_cache: dict[int, np.ndarray] = {0: np.zeros((H, W), "float32")}  # accumulation at step 0 = 0
    rate_cache: dict[int, np.ndarray | None] = {}

    def tp_at(step: int):
        if step in tp_cache:
            return tp_cache[step]
        path = runs[run].get(step)
        if path is None:
            return None
        la, lo, tp = open_tp(path)
        grid = regrid_to(glat, glon, la, lo, tp).astype("float32")
        tp_cache[step] = grid
        return grid

    def rate_at(step: int):
        """Mean mm/h over the (step-step_h, step] accumulation window."""
        if step not in rate_cache:
            hi, lo = tp_at(step), tp_at(step - step_h)
            rate_cache[step] = None if (hi is None or lo is None) else np.clip((hi - lo) / step_h, 0.0, None)
        return rate_cache[step]

    out = np.full((len(leads_min), H, W), np.nan, "float32")
    for i, lead in enumerate(leads_min):
        valid = issue_dt + dt.timedelta(minutes=int(lead))
        hours = (valid - run_dt).total_seconds() / 3600.0
        if hours <= 0:
            continue  # valid before the run — leave NaN (nowcast covers it)
        s0, s1, w = _bracketing_steps(hours, steps, step_h)
        r0, r1 = rate_at(s0), rate_at(s1)
        if r0 is None or r1 is None:
            # fall back to the single covering interval if a neighbour is missing
            r = rate_at(interval_step(hours, step_h, max_step))
            if r is not None:
                out[i] = r
            continue
        out[i] = r0 if w == 0.0 else (1.0 - w) * r0 + w * r1  # smooth temporal blend
    return out
