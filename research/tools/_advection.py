"""Pure-numpy block-matching advection, used for the benchmark's "advection"
baseline (persistence advected by the motion estimated from the two most
recent history frames, extrapolated to the target lead).

This is a standalone port of the block-matching flow estimator in
``backend/src/pluvio_backend/morph.py`` (``_block_flow`` / ``_warp``) — copied
rather than imported so this tool never depends on the backend package.
"""

from __future__ import annotations

import numpy as np

MAX_SHIFT = 7          # px search radius per block
BLOCKS = 4              # BLOCKS x BLOCKS overlapping estimation windows
WET_THR = 0.05          # mm/h — cells that participate in matching


def flow_for_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coarse displacement field (2, H, W): b's features sit at a's position
    plus this flow (i.e. this is the a→b motion, in pixels)."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
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
                continue
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
                    score = float((ref * cand).sum())
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
                     step_min: float) -> np.ndarray:
    """Advection baseline: extrapolate ``curr`` forward to ``lead_min`` minutes
    ahead, using the flow estimated between ``prev`` and ``curr`` (``step_min``
    minutes apart) scaled linearly to the target lead."""
    flow = flow_for_pair(prev, curr)
    scale = lead_min / step_min if step_min else 0.0
    fy, fx = flow
    out = warp(curr, scale * fy, scale * fx)
    return np.clip(out, 0.0, None).astype("float32")
