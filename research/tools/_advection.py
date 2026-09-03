"""Pure-numpy block-matching advection, used for the benchmark's "advection"
baseline (persistence advected by the motion estimated from the two most
recent history frames, extrapolated to the target lead).

This is a standalone port of the block-matching flow estimator in
``backend/src/pluvio_backend/morph.py`` (``_block_flow`` / ``_warp``) — copied
rather than imported so this tool never depends on the backend package. Two
deliberate departures from the backend version, made for benchmark accuracy
rather than serving-time cheapness:

  * the per-block match score is a mean-subtracted, std-normalised cross
    correlation over the block's wet cells, not a raw dot product — a plain
    dot product is biased toward whichever candidate offset has the most
    accumulated mass, which both overshoots the true displacement and creates
    spurious cross-axis drift (~30% magnitude bias, measured on synthetic
    shifts);
  * the search radius (``max_shift``) is a parameter, not a hard-coded
    constant — the backend's morph.py tuned it for its own ~6 km serving
    grid; a benchmark run against a different store (e.g. the ~3.2 km v3
    grid) must derive it from that store's actual pixel size, or a fixed
    search radius silently caps the estimate and biases every downstream
    score. See ``max_shift_px``.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_MAX_SHIFT = 7   # px search radius — fallback only, when grid spacing is unknown
BLOCKS = 4               # BLOCKS x BLOCKS overlapping estimation windows
WET_THR = 0.05           # mm/h — cells that participate in matching
_EPS = 1e-6


def max_shift_px(km_per_px: float, step_min: float, *, max_kmh: float = 100.0,
                 min_px: int = 2, max_px: int = 25) -> int:
    """Search radius (px) that comfortably covers ``max_kmh`` motion over one
    ``step_min``-minute step on a grid with ``km_per_px`` spacing, clamped to
    a sane range so a degenerate store attr can't blow up the search cost."""
    if not km_per_px or km_per_px <= 0:
        return DEFAULT_MAX_SHIFT
    shift = math.ceil(max_kmh * (step_min / 60.0) / km_per_px)
    return int(min(max_px, max(min_px, shift)))


def _ncc_score(ref: np.ndarray, cand: np.ndarray) -> float:
    """Mean-subtracted, std-normalised cross correlation over the whole
    block — invariant to the overall mass/offset of either block (unlike a
    raw dot product, which is biased toward whichever candidate offset
    happens to overlap the most accumulated rain), so the best-matching
    displacement is chosen on *shape*, not magnitude.

    Deliberately NOT restricted to "wet" pixels only: on a flat-topped rain
    cell the interior has ~zero variance (every wet pixel is the same
    intensity), so a wet-only mask would throw away exactly the edge
    contrast that locates the cell. Blocks with too little rain to say
    anything are skipped by the caller (a block-level wet-fraction gate),
    but a block that clears that gate is scored over its full extent.
    """
    r = ref.astype("float64")
    c = cand.astype("float64")
    r = r - r.mean()
    c = c - c.mean()
    denom = math.sqrt(float((r * r).sum())) * math.sqrt(float((c * c).sum()))
    if denom < _EPS:
        return -np.inf
    return float((r * c).sum()) / denom


def flow_for_pair(a: np.ndarray, b: np.ndarray, *, max_shift: int = DEFAULT_MAX_SHIFT) -> np.ndarray:
    """Coarse displacement field (2, H, W): b's features sit at a's position
    plus this flow (i.e. this is the a→b motion, in pixels).

    Callers must pass finite (NaN-free) ``a``/``b`` — fill any store NaN
    (e.g. outside the radar domain) with 0 first; NaN reaching the block
    matcher silently zeroes every block it touches.
    """
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    h, w = a.shape
    la, lb = np.log1p(np.maximum(a, 0.0)), np.log1p(np.maximum(b, 0.0))
    wet_a = la > np.log1p(WET_THR)
    ys = np.linspace(0, h, BLOCKS + 1).astype(int)
    xs = np.linspace(0, w, BLOCKS + 1).astype(int)
    cys = [(ys[i] + ys[i + 1]) // 2 for i in range(BLOCKS)]
    cxs = [(xs[i] + xs[i + 1]) // 2 for i in range(BLOCKS)]
    vy = np.zeros((BLOCKS, BLOCKS), dtype="float32")
    vx = np.zeros((BLOCKS, BLOCKS), dtype="float32")
    m = int(max_shift)
    for bi in range(BLOCKS):
        for bj in range(BLOCKS):
            y0, y1 = ys[bi], ys[bi + 1]
            x0, x1 = xs[bj], xs[bj + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            ref = la[y0:y1, x0:x1]
            wet = wet_a[y0:y1, x0:x1]
            if float(wet.mean()) < 0.005:
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
    fy = np.zeros((h, w), dtype="float32")
    fx = np.zeros((h, w), dtype="float32")
    col_pos = np.arange(w, dtype="float32")
    row_pos = np.arange(h, dtype="float32")
    fy_rows = np.stack([np.interp(col_pos, cxs, vy[i]) for i in range(BLOCKS)])
    fx_rows = np.stack([np.interp(col_pos, cxs, vx[i]) for i in range(BLOCKS)])
    for c in range(w):
        fy[:, c] = np.interp(row_pos, cys, fy_rows[:, c])
        fx[:, c] = np.interp(row_pos, cys, fx_rows[:, c])
    return np.stack([fy, fx])


def warp(field: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Sample ``field`` at (y - dy, x - dx) with bilinear interpolation —
    i.e. advect ``field`` forward by displacement (dy, dx)."""
    field = np.asarray(field, dtype="float32")
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
