"""Canonical block-matching optical-flow estimator.

Two callers need the same coarse a→b displacement field on a rain-rate grid:
the serving-side temporal morph (``backend/src/pluvio_backend/morph.py``,
which densifies the nowcast band between forecast leads) and the benchmark's
advection baseline (``research/tools/_advection.py``). This module is the one
implementation; ``research/tools/_advection.py`` imports it directly.
``backend/morph.py`` cannot (the runtime image ships only ``backend/src`` —
see its Dockerfile — so it carries a self-contained copy of the same
algorithm instead; keep the two in sync by hand).

Each block is scored by mean-subtracted, std-normalised cross correlation
(NCC) over the block's full extent — not a raw dot product, which is biased
toward whichever candidate offset has the most accumulated mass. That bias
both overshoots the true displacement and creates spurious cross-axis drift
(see test_motion.py for a measured case). Blocks are gated by a wet-cell
fraction check first (a block with no rain has nothing to align and would
just chase noise); blocks that don't clear the gate are flagged invalid and
contribute zero flow rather than a spurious estimate.

Pure numpy on purpose: the backend copy runs in a python3.14-slim image
where shipping OpenCV is a wheel lottery, and this module mirrors that
constraint so a port never needs anything more.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_MAX_SHIFT = 7   # px search radius — fallback only, when grid spacing is unknown
BLOCKS = 4               # BLOCKS x BLOCKS overlapping estimation windows
WET_THR = 0.05           # mm/h — cells that count toward a block's wet fraction
MIN_WET_FRAC = 0.005     # below this wet fraction, a block is flagged invalid
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


def _parabolic_offset(s_minus: float, s_zero: float, s_plus: float) -> float:
    """Sub-pixel offset (in [-1, 1]) of a parabola's peak fit through three
    equally-spaced samples, the middle one the discrete-search maximum."""
    denom = s_minus - 2.0 * s_zero + s_plus
    if not np.isfinite(denom) or abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (s_minus - s_plus) / denom, -1.0, 1.0))


def block_flow(a: np.ndarray, b: np.ndarray, *, max_shift: int = DEFAULT_MAX_SHIFT,
               blocks: int = BLOCKS, wet_thr: float = WET_THR,
               min_wet_frac: float = MIN_WET_FRAC,
               subpixel: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block a→b displacement, scored by NCC on wet blocks only.

    Returns ``(vy, vx, valid)``, three ``(blocks, blocks)`` arrays: the
    per-block displacement (sub-pixel when ``subpixel=True``, via a
    parabolic fit of the score surface around the integer best offset) and a
    bool mask of blocks that had enough wet signal to estimate motion — a
    block below ``min_wet_frac`` is flagged invalid rather than guessed at,
    and left at ``(0, 0)``.

    Raises ``ValueError`` on non-finite input; NaN (e.g. outside the radar
    domain) must be filled by the caller before calling this.
    """
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if a.shape != b.shape:
        raise ValueError(f"block_flow: shape mismatch {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("block_flow: non-finite input — fill NaN (e.g. outside the "
                         "radar domain) before calling")
    h, w = a.shape
    la, lb = np.log1p(np.maximum(a, 0.0)), np.log1p(np.maximum(b, 0.0))
    wet_a = la > np.log1p(wet_thr)
    ys = np.linspace(0, h, blocks + 1).astype(int)
    xs = np.linspace(0, w, blocks + 1).astype(int)
    vy = np.zeros((blocks, blocks), dtype="float32")
    vx = np.zeros((blocks, blocks), dtype="float32")
    valid = np.zeros((blocks, blocks), dtype=bool)
    m = int(max_shift)
    for bi in range(blocks):
        for bj in range(blocks):
            y0, y1 = ys[bi], ys[bi + 1]
            x0, x1 = xs[bj], xs[bj + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            ref = la[y0:y1, x0:x1]
            if float(wet_a[y0:y1, x0:x1].mean()) < min_wet_frac:
                continue  # flagged invalid: no measurable motion, leave (0, 0)
            best, bdy, bdx = -np.inf, 0, 0
            scores: dict[tuple[int, int], float] = {}
            for dy in range(-m, m + 1):
                for dx in range(-m, m + 1):
                    yy0, yy1 = y0 + dy, y1 + dy
                    xx0, xx1 = x0 + dx, x1 + dx
                    if yy0 < 0 or xx0 < 0 or yy1 > h or xx1 > w:
                        continue
                    score = _ncc_score(ref, lb[yy0:yy1, xx0:xx1])
                    scores[(dy, dx)] = score
                    if score > best:
                        best, bdy, bdx = score, dy, dx
            fdy, fdx = float(bdy), float(bdx)
            if subpixel:
                sym, syp = scores.get((bdy - 1, bdx)), scores.get((bdy + 1, bdx))
                if sym is not None and syp is not None and np.isfinite(sym) and np.isfinite(syp):
                    fdy += _parabolic_offset(sym, best, syp)
                sxm, sxp = scores.get((bdy, bdx - 1)), scores.get((bdy, bdx + 1))
                if sxm is not None and sxp is not None and np.isfinite(sxm) and np.isfinite(sxp):
                    fdx += _parabolic_offset(sxm, best, sxp)
            vy[bi, bj], vx[bi, bj] = fdy, fdx
            valid[bi, bj] = True
    return vy, vx, valid


def upsample_flow(vy: np.ndarray, vx: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly interpolate a ``(blocks, blocks)`` flow field onto a full
    ``out_hw`` grid — separable: along x within each block-row, then along y
    between block-rows' block-centres."""
    blocks_y, blocks_x = vy.shape
    h, w = out_hw
    ys = np.linspace(0, h, blocks_y + 1).astype(int)
    xs = np.linspace(0, w, blocks_x + 1).astype(int)
    cys = [(ys[i] + ys[i + 1]) // 2 for i in range(blocks_y)]
    cxs = [(xs[i] + xs[i + 1]) // 2 for i in range(blocks_x)]
    fy = np.zeros((h, w), dtype="float32")
    fx = np.zeros((h, w), dtype="float32")
    col_pos = np.arange(w, dtype="float32")
    row_pos = np.arange(h, dtype="float32")
    fy_rows = np.stack([np.interp(col_pos, cxs, vy[i]) for i in range(blocks_y)])
    fx_rows = np.stack([np.interp(col_pos, cxs, vx[i]) for i in range(blocks_y)])
    for c in range(w):
        fy[:, c] = np.interp(row_pos, cys, fy_rows[:, c])
        fx[:, c] = np.interp(row_pos, cys, fx_rows[:, c])
    return np.stack([fy, fx])


def flow_field(a: np.ndarray, b: np.ndarray, *, max_shift: int = DEFAULT_MAX_SHIFT,
              blocks: int = BLOCKS, wet_thr: float = WET_THR,
              min_wet_frac: float = MIN_WET_FRAC, subpixel: bool = False) -> np.ndarray:
    """Full-resolution ``(2, H, W)`` a→b displacement field: block flow,
    smoothed onto the grid by bilinear interpolation between block centres."""
    vy, vx, _valid = block_flow(a, b, max_shift=max_shift, blocks=blocks, wet_thr=wet_thr,
                                min_wet_frac=min_wet_frac, subpixel=subpixel)
    return upsample_flow(vy, vx, a.shape)


def warp(field: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Sample ``field`` at (y - dy, x - dx) with bilinear interpolation —
    i.e. advect ``field`` forward by displacement (dy, dx): content moves by
    +D."""
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
