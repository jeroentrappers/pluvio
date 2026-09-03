"""Advection baseline for the benchmark: persistence advected by the motion
estimated from the two most recent history frames, extrapolated to the
target lead.

A thin wrapper over ``model.motion``, the canonical block-matching flow
estimator shared with the backend's serving-side morph
(``backend/src/pluvio_backend/morph.py`` carries its own copy of the same
algorithm rather than importing this — see that module's docstring for why).
"""

from __future__ import annotations

import numpy as np
from model.motion import DEFAULT_MAX_SHIFT, flow_field, max_shift_px, warp

__all__ = ["DEFAULT_MAX_SHIFT", "advect_forecast", "flow_for_pair", "max_shift_px", "warp"]


def flow_for_pair(a: np.ndarray, b: np.ndarray, *, max_shift: int = DEFAULT_MAX_SHIFT) -> np.ndarray:
    """Coarse displacement field (2, H, W): b's features sit at a's position
    plus this flow (i.e. this is the a→b motion, in pixels).

    Callers must pass finite (NaN-free) ``a``/``b`` — fill any store NaN
    (e.g. outside the radar domain) with 0 first; ``model.motion.block_flow``
    raises clearly otherwise.
    """
    return flow_field(np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32"),
                      max_shift=max_shift)


def advect_forecast(prev: np.ndarray, curr: np.ndarray, lead_min: float,
                     step_min: float, *, max_shift: int = DEFAULT_MAX_SHIFT,
                     flow: np.ndarray | None = None) -> np.ndarray:
    """Advection baseline: extrapolate ``curr`` forward to ``lead_min`` minutes
    ahead, using the flow estimated between ``prev`` and ``curr`` (``step_min``
    minutes apart) scaled linearly to the target lead.

    ``prev``/``curr`` must be finite (NaN filled by the caller). Pass a
    precomputed ``flow`` (from ``flow_for_pair``) to skip re-estimating it —
    the flow only depends on the (prev, curr) pair, not on the lead, so a
    caller scoring several leads from the same issue should compute it once.
    """
    if flow is None:
        flow = flow_for_pair(prev, curr, max_shift=max_shift)
    scale = lead_min / step_min if step_min else 0.0
    fy, fx = flow
    out = warp(curr, scale * fy, scale * fx)
    return np.clip(out, 0.0, None).astype("float32")
