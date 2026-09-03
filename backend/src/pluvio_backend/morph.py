"""Motion-based temporal interpolation between forecast frames.

The observed history plays fluidly because the composite producer emits
motion-interpolated frames every ~100 s. Forecast leads arrive 10 min apart,
so the timeline's cadence (and apparent motion) broke at the t=0 seam. This
module densifies the nowcast band by MORPHING between consecutive predicted
frames: estimate a coarse displacement field, advect both frames toward the
intermediate time, and cross-blend. Linear blending (the old `_interp_lead`)
makes cells fade in place; morphing makes them move.

Pure numpy on purpose: the runtime image is python 3.14-slim and shipping
OpenCV for a cp314 target is a wheel lottery. Displacements between 10-min
forecast frames on the ~6 km model grid are tiny (60 km/h ≈ 1.7 px), so a
block-matching search over ±MAX_SHIFT (7) px with a smoothed flow field
captures the motion that matters.

``_block_flow`` is a self-contained copy of ``research/model/motion.py``'s
block-matching algorithm — not an import, because the runtime image ships
only this ``backend/src`` tree (see the Dockerfile), so it can't depend on
the research package. Keep the two in sync by hand; ``research/model/
motion.py`` is the canonical version and carries the fuller derivation notes.
"""

from __future__ import annotations

import math

import numpy as np

MAX_SHIFT = 7          # px search radius per block — v2 leads are 30 min apart, so 60-80 km/h motion spans ~5-7 px on the 6 km grid
BLOCKS = 4             # BLOCKS x BLOCKS overlapping estimation windows
WET_THR = 0.05         # mm/h — cells that participate in matching
_EPS = 1e-6


def _ncc_score(ref: np.ndarray, cand: np.ndarray) -> float:
    """Mean-subtracted, std-normalised cross correlation — invariant to the
    overall mass/offset of either block (unlike a raw dot product, which is
    biased toward whichever candidate offset happens to overlap the most
    accumulated rain and both overshoots the true displacement and creates
    spurious cross-axis drift)."""
    r = ref.astype("float64")
    c = cand.astype("float64")
    r = r - r.mean()
    c = c - c.mean()
    denom = math.sqrt(float((r * r).sum())) * math.sqrt(float((c * c).sum()))
    if denom < _EPS:
        return -np.inf
    return float((r * c).sum()) / denom


def _block_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coarse displacement field (2, H, W): b ≈ a advected by flow.

    For each of BLOCKS² overlapping windows, find the integer (dy, dx) within
    ±MAX_SHIFT that best aligns a→b (max NCC of log1p fields, wet cells
    only), then bilinearly interpolate the block vectors to full res.

    Raises ``ValueError`` on non-finite input; NaN reaching the block matcher
    would silently zero every block it touches, so the caller must fill it
    (e.g. outside the radar domain) before calling this.
    """
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("_block_flow: non-finite input — fill NaN (e.g. outside the "
                         "radar domain) before calling")
    h, w = a.shape
    la, lb = np.log1p(np.maximum(a, 0.0)), np.log1p(np.maximum(b, 0.0))
    ys = np.linspace(0, h, BLOCKS + 1).astype(int)
    xs = np.linspace(0, w, BLOCKS + 1).astype(int)
    cys = [(ys[i] + ys[i + 1]) // 2 for i in range(BLOCKS)]
    cxs = [(xs[i] + xs[i + 1]) // 2 for i in range(BLOCKS)]
    vy = np.zeros((BLOCKS, BLOCKS), dtype="float32")
    vx = np.zeros((BLOCKS, BLOCKS), dtype="float32")
    m = MAX_SHIFT
    for bi in range(BLOCKS):
        for bj in range(BLOCKS):
            y0, y1 = ys[bi], ys[bi + 1]
            x0, x1 = xs[bj], xs[bj + 1]
            if y1 <= y0 or x1 <= x0:
                continue  # degenerate block (field smaller than BLOCKS on an axis)
            ref = la[y0:y1, x0:x1]
            if float((ref > np.log1p(WET_THR)).mean()) < 0.005:
                continue  # (near-)dry block: no measurable motion, leave 0
            best, bdy, bdx = -np.inf, 0, 0
            for dy in range(-m, m + 1):
                for dx in range(-m, m + 1):
                    yy0, yy1 = y0 + dy, y1 + dy
                    xx0, xx1 = x0 + dx, x1 + dx
                    if yy0 < 0 or xx0 < 0 or yy1 > h or xx1 > w:
                        continue
                    cand = lb[yy0:yy1, xx0:xx1]
                    score = _ncc_score(ref, cand)
                    if score > best:
                        best, bdy, bdx = score, dy, dx
            vy[bi, bj], vx[bi, bj] = bdy, bdx
    # 2-D separable interp: rows over block-centres for each column band, then cols.
    fy = np.zeros((h, w), dtype="float32")
    fx = np.zeros((h, w), dtype="float32")
    col_pos = np.arange(w, dtype="float32")
    row_pos = np.arange(h, dtype="float32")
    # interp along x for each block-row, then along y between block-rows
    fy_rows = np.stack([np.interp(col_pos, cxs, vy[i]) for i in range(BLOCKS)])
    fx_rows = np.stack([np.interp(col_pos, cxs, vx[i]) for i in range(BLOCKS)])
    for c in range(w):
        fy[:, c] = np.interp(row_pos, cys, fy_rows[:, c])
        fx[:, c] = np.interp(row_pos, cys, fx_rows[:, c])
    return np.stack([fy, fx])


def _warp(field: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Sample `field` at (y - dy, x - dx) with bilinear interpolation."""
    h, w = field.shape
    yy, xx = np.meshgrid(np.arange(h, dtype="float32"),
                         np.arange(w, dtype="float32"), indexing="ij")
    sy = np.clip(yy - dy, 0, h - 1)
    sx = np.clip(xx - dx, 0, w - 1)
    y0 = np.floor(sy).astype(int)
    x0 = np.floor(sx).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = sy - y0
    wx = sx - x0
    return ((field[y0, x0] * (1 - wy) * (1 - wx)) +
            (field[y1, x0] * wy * (1 - wx)) +
            (field[y0, x1] * (1 - wy) * wx) +
            (field[y1, x1] * wy * wx)).astype("float32")


def morph_pair(a: np.ndarray, b: np.ndarray, w: float,
               flow: np.ndarray | None = None) -> np.ndarray:
    """Motion-interpolated frame at fraction w ∈ (0, 1) between a and b."""
    if flow is None:
        flow = _block_flow(a, b)
    fy, fx = flow
    a_adv = _warp(a, w * fy, w * fx)                    # a advected forward
    b_adv = _warp(b, -(1 - w) * fy, -(1 - w) * fx)      # b advected backward
    out = (1 - w) * a_adv + w * b_adv
    return np.clip(out, 0.0, None).astype("float32")


def flow_for_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Precompute the displacement field for repeated morph_pair calls."""
    return _block_flow(a, b)
