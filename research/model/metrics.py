"""Precipitation verification metrics that the community actually accepts.

MAE/RMSE on a ~98%-dry domain are dominated by zeros and reward blurry, hedged
fields (the double-penalty problem) — they are the *wrong* headline for precip.
This module adds the scale-aware and probabilistic scores that aren't fooled by
that (recommendation #1):

  * **FSS** — Fractions Skill Score: compares the *fraction* of wet pixels in a
    neighbourhood, so a near-miss isn't double-penalised and a forecast is
    credited at the spatial scale it's actually skilful at. The honest answer to
    "is our CSI just a coarse-grid artefact?" — report FSS across scales.
  * **CRPS** — the proper score for the probabilistic outlook (rec #3). For a
    point forecast CRPS reduces to MAE, so deterministic and probabilistic
    models compare on one axis. For quantiles, CRPS ≈ 2·mean pinball loss.
  * neighbourhood-aware categorical CSI/POD/FAR/bias for reference.

Everything is pure numpy (+ scipy.ndimage for the neighbourhood filter) and
operates on (H, W) or stacked (N, H, W) arrays, so it's cheap to unit-test and
shared by every eval harness.
"""

from __future__ import annotations

import numpy as np


def _as_stack(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    return a[None] if a.ndim == 2 else a


# ───────────────────────────────────────────────────────────── categorical


def categorical_scores(pred: np.ndarray, obs: np.ndarray, threshold: float) -> dict:
    """Pointwise CSI/POD/FAR/frequency-bias at a rain threshold (mm/h)."""
    p = np.asarray(pred) >= threshold
    o = np.asarray(obs) >= threshold
    hits = int((p & o).sum())
    misses = int((~p & o).sum())
    fa = int((p & ~o).sum())
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = fa / (hits + fa) if (hits + fa) else float("nan")
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else float("nan")
    bias = (hits + fa) / (hits + misses) if (hits + misses) else float("nan")
    return {"csi": csi, "pod": pod, "far": far, "freq_bias": bias,
            "hits": hits, "misses": misses, "false_alarms": fa}


# ─────────────────────────────────────────────────────────────────── FSS


def _fraction_field(mask: np.ndarray, scale_px: int) -> np.ndarray:
    """Fraction of wet pixels in a (scale_px × scale_px) neighbourhood."""
    from scipy.ndimage import uniform_filter

    return uniform_filter(mask.astype("float64"), size=scale_px, mode="constant", cval=0.0)


def fractions_skill_score(pred: np.ndarray, obs: np.ndarray, *, threshold: float,
                          scale_px: int) -> float:
    """FSS at one intensity threshold and one neighbourhood scale (in pixels).

    FSS = 1 − MSE(Pf, Po) / (⟨Pf²⟩ + ⟨Po²⟩), where Pf/Po are the wet-fraction
    fields. 1 = perfect, 0 = no skill. FSS rises with scale; the scale where it
    first exceeds 0.5 + f/2 (f = domain wet fraction) is the smallest skilful
    scale. Averaged over the sample stack.
    """
    P, O = _as_stack(pred), _as_stack(obs)
    num = den = 0.0
    for i in range(P.shape[0]):
        pf = _fraction_field(P[i] >= threshold, scale_px)
        po = _fraction_field(O[i] >= threshold, scale_px)
        num += float(np.mean((pf - po) ** 2))
        den += float(np.mean(pf ** 2) + np.mean(po ** 2))
    if den == 0.0:
        return float("nan")  # no wet pixels anywhere at this threshold
    return 1.0 - num / den


def fss_curve(pred: np.ndarray, obs: np.ndarray, *, threshold: float,
              scales_px) -> dict[int, float]:
    """FSS at several neighbourhood scales — the scale-sensitivity curve that
    reveals whether a CSI win is real or a coarse-grid artefact."""
    return {int(s): fractions_skill_score(pred, obs, threshold=threshold, scale_px=int(s))
            for s in scales_px}


# ────────────────────────────────────────────────────────────────── CRPS


def crps_deterministic(pred: np.ndarray, obs: np.ndarray) -> float:
    """CRPS of a point forecast == MAE. Lets a deterministic model sit on the
    same axis as a probabilistic one."""
    return float(np.mean(np.abs(np.asarray(pred, "float64") - np.asarray(obs, "float64"))))


def crps_from_quantiles(quantile_preds: np.ndarray, obs: np.ndarray, quantiles) -> float:
    """Approximate CRPS from a set of predictive quantiles.

    CRPS = 2 ∫₀¹ pinball_τ dτ ≈ 2 · mean_τ pinball_τ for quantile levels τ. Lower
    is better; for a degenerate (zero-spread) forecast this equals MAE, matching
    ``crps_deterministic`` so the two are directly comparable.

        quantile_preds : (Q, …) predictions, one leading entry per level
        obs            : (…) truth, broadcast against each quantile
        quantiles      : the Q levels (same order as the leading axis)
    """
    q = np.asarray(quantiles, dtype="float64")
    qp = np.asarray(quantile_preds, dtype="float64")
    o = np.asarray(obs, dtype="float64")
    if qp.shape[0] != q.shape[0]:
        raise ValueError(f"quantile_preds leading axis {qp.shape[0]} != {q.shape[0]} levels")
    total = 0.0
    for k, level in enumerate(q):
        err = o - qp[k]
        total += float(np.mean(np.maximum(level * err, (level - 1.0) * err)))
    return 2.0 * total / len(q)


def continuous_scores(pred: np.ndarray, obs: np.ndarray) -> dict:
    """MAE / RMSE / bias — kept for continuity, never the headline for precip."""
    e = np.asarray(pred, "float64") - np.asarray(obs, "float64")
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e))}
