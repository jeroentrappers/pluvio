"""Classical seamless forecast — the product baseline that ships *now*.

This is the deliberately non-learned producer behind recommendation #4
(decouple the product from the research): a credible 0 → 240 h precipitation
cube built from classical methods only, so the PWA never has to wait on the
research model. The learned `SeamlessNet` becomes a drop-in *upgrade* that
swaps in only once it beats this baseline on the champion/challenger gate
(docs/plan_overview.md §5) — never before.

Three regimes, the same split as docs/seamless_model_plan.md §1, but every
piece is a method with a known skill curve:

  * 0–2 h  **nowcast**  — optical-flow (Lucas–Kanade) advection extrapolation of
                          the latest OPERA radar analyses. This is the real
                          operational-grade nowcast baseline (pysteps), *not*
                          persistence.
  * 2–6 h  **blend**    — a smooth crossover that hands off from the decaying
                          radar extrapolation to NWP as radar skill crosses zero.
  * 6–240 h **outlook** — raw AIFS precip, regridded onto the analysis grid.

Every lead carries a **source tag** and a **confidence** that widens with lead
(docs/24h_extension.md's rule: never pretend the day-5 number came from radar).

`optical_flow_nowcast` uses pysteps when importable (the production / verification
path) and falls back to a self-contained FFT phase-correlation + semi-Lagrangian
advection so the module — and the eval baselines that depend on it — run even in
a slim env without pysteps. `engine` in the returned dict records which ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger("pluvio.classical")

# Regime boundaries (minutes). Radar extrapolation has real skill to ~2 h and
# crosses persistence between 1–2 h for convective regimes (docs/verification.md).
NOWCAST_END_MIN = 120
BLEND_END_MIN = 360  # 6 h: radar fully handed off to NWP

# Skill-based confidence anchors (0–1), interpolated across leads. These are the
# product's honesty knob, not a calibrated probability — calibrated spread is
# the learned model's job (rec #3, the quantile outlook head).
_CONFIDENCE_ANCHORS = {
    0: 0.95,
    30: 0.85,
    60: 0.70,
    120: 0.50,
    360: 0.38,
    1440: 0.30,
    4320: 0.20,
    14400: 0.12,
}


@dataclass
class ClassicalForecast:
    """The seamless cube produced from classical methods."""

    leads_min: np.ndarray  # (n_lead,) int
    rates: np.ndarray      # (n_lead, H, W) mm/h
    source: list[str]      # (n_lead,) one of {"nowcast", "blend", "nwp"}
    confidence: np.ndarray  # (n_lead,) in [0, 1]
    engine: str            # "pysteps" | "fallback-phasecorr"


# ───────────────────────────────────────────────────────── optical flow nowcast


def _estimate_motion_pysteps(frames: np.ndarray):
    """Dense Lucas–Kanade motion field via pysteps. frames: (T, H, W).

    Returns the (2, H, W) advection field pysteps' extrapolator expects, or
    raises ImportError if pysteps is unavailable.
    """
    from pysteps import motion

    # pysteps works in dBR / dBZ-like space; a mild log transform keeps the
    # flow estimate from being dominated by a few heavy-rain pixels.
    f = np.log1p(np.maximum(frames, 0.0)).astype("float64")
    oflow = motion.get_method("LK")
    return oflow(f)


def _extrapolate_pysteps(last_frame: np.ndarray, motion_field, n_steps: int) -> np.ndarray:
    from pysteps import nowcasts

    extrap = nowcasts.get_method("extrapolation")
    out = extrap(last_frame.astype("float64"), motion_field, n_steps)
    return np.asarray(out, dtype="float32")


def _global_motion_phasecorr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate a single (dy, dx) pixel shift from frame ``a`` to ``b`` by FFT
    phase correlation. Robust, cheap, and a legitimate (if coarse) advection
    estimate when pysteps' dense flow isn't available."""
    a = np.log1p(np.maximum(a, 0.0))
    b = np.log1p(np.maximum(b, 0.0))
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0, 0.0
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-8
    corr = np.fft.ifft2(r).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    # shift from a→b; motion per frame is the negative (where rain came *from*).
    return float(-dy), float(-dx)


def _advect_semilagrangian(field: np.ndarray, dy: float, dx: float, step: int) -> np.ndarray:
    """Backward semi-Lagrangian advection of ``field`` by ``step`` frames of a
    uniform (dy, dx) per-frame motion. Sample where each cell came *from*."""
    from scipy.ndimage import map_coordinates

    h, w = field.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    src_r = rr - dy * step
    src_c = cc - dx * step
    out = map_coordinates(field, [src_r, src_c], order=1, mode="constant", cval=0.0)
    return np.clip(out, 0.0, None).astype("float32")


def optical_flow_nowcast(
    history: np.ndarray,
    leads_min,
    *,
    dt_min: float,
    prefer_pysteps: bool = True,
) -> tuple[np.ndarray, str]:
    """Optical-flow advection nowcast.

    Args:
        history: (T, H, W) most-recent OPERA RATE analyses, oldest→newest, mm/h.
        leads_min: iterable of forecast lead minutes (≥0).
        dt_min: spacing between the history frames (minutes).
        prefer_pysteps: use pysteps' dense LK when importable.

    Returns:
        (rates, engine) where rates is (n_lead, H, W) mm/h and engine names the
        method that ran.
    """
    history = np.nan_to_num(np.asarray(history, dtype="float32"), nan=0.0)
    if history.ndim != 3 or history.shape[0] < 2:
        raise ValueError("history must be (T>=2, H, W)")
    leads = [int(x) for x in leads_min]
    last = history[-1]

    if prefer_pysteps:
        try:
            motion_field = _estimate_motion_pysteps(history)
            max_lead = max(leads)
            n_steps = max(1, int(round(max_lead / dt_min)))
            seq = _extrapolate_pysteps(last, motion_field, n_steps)  # (n_steps, H, W)
            rates = np.empty((len(leads), *last.shape), dtype="float32")
            for i, lead in enumerate(leads):
                if lead <= 0:
                    rates[i] = last
                    continue
                k = min(n_steps, int(round(lead / dt_min)))
                rates[i] = np.clip(np.nan_to_num(seq[k - 1], nan=0.0), 0.0, None)
            return rates, "pysteps"
        except ImportError:
            LOG.info("pysteps unavailable — using FFT phase-correlation fallback")
        except Exception as exc:  # numerical edge cases must not kill the product
            LOG.warning("pysteps nowcast failed (%s) — falling back", exc)

    # Fallback: global per-frame motion from the two latest frames + advection.
    dy, dx = _global_motion_phasecorr(history[-2], history[-1])
    rates = np.empty((len(leads), *last.shape), dtype="float32")
    for i, lead in enumerate(leads):
        step = lead / dt_min
        rates[i] = last if lead <= 0 else _advect_semilagrangian(last, dy, dx, step)
    return rates, "fallback-phasecorr"


# ──────────────────────────────────────────────────────────────── confidence


def confidence_for_leads(leads_min) -> np.ndarray:
    """Lead-widening confidence in [0, 1] by interpolating the skill anchors."""
    xs = np.array(sorted(_CONFIDENCE_ANCHORS))
    ys = np.array([_CONFIDENCE_ANCHORS[x] for x in xs])
    leads = np.asarray([int(x) for x in leads_min], dtype="float64")
    return np.interp(leads, xs, ys).astype("float32")


def source_for_lead(lead_min: int) -> str:
    if lead_min <= NOWCAST_END_MIN:
        return "nowcast"
    if lead_min <= BLEND_END_MIN:
        return "blend"
    return "nwp"


def _blend_weight(lead_min: float) -> float:
    """Radar weight in the 2–6 h handoff: 1.0 at the nowcast horizon, smoothly
    to 0.0 at the blend horizon (cosine taper)."""
    if lead_min <= NOWCAST_END_MIN:
        return 1.0
    if lead_min >= BLEND_END_MIN:
        return 0.0
    frac = (lead_min - NOWCAST_END_MIN) / (BLEND_END_MIN - NOWCAST_END_MIN)
    return float(0.5 * (1.0 + np.cos(np.pi * frac)))  # 1→0


# ──────────────────────────────────────────────────────────── seamless cube


def seamless_cube(
    history: np.ndarray,
    leads_min,
    *,
    dt_min: float,
    aifs_rates=None,
    prefer_pysteps: bool = True,
) -> ClassicalForecast:
    """Assemble the full 0–240 h classical cube.

    Args:
        history: (T, H, W) recent OPERA RATE analyses (oldest→newest), mm/h.
        leads_min: forecast lead minutes.
        dt_min: spacing of history frames (minutes).
        aifs_rates: optional (n_lead, H, W) raw AIFS precip already regridded
            onto the analysis grid, aligned to ``leads_min``. Drives the outlook
            and the blend tail. If None, the cube is nowcast-only (radar carried
            forward past its horizon — clearly lower confidence).
        prefer_pysteps: pass through to the nowcast.
    """
    leads = [int(x) for x in leads_min]
    nowcast, engine = optical_flow_nowcast(
        history, leads, dt_min=dt_min, prefer_pysteps=prefer_pysteps
    )

    if aifs_rates is not None:
        aifs = np.nan_to_num(np.asarray(aifs_rates, dtype="float32"), nan=0.0)
        if aifs.shape != nowcast.shape:
            raise ValueError(f"aifs_rates {aifs.shape} != nowcast {nowcast.shape}")
    else:
        aifs = None

    rates = np.empty_like(nowcast)
    source: list[str] = []
    for i, lead in enumerate(leads):
        w = _blend_weight(lead)
        if aifs is None:
            # No NWP anchor → every lead is radar extrapolation carried forward.
            # Label it truthfully as "nowcast" (never "nwp") — the whole point of
            # the source tag is to not pretend a day-5 number came from somewhere
            # it didn't. Confidence already decays with lead to flag the stretch.
            rates[i] = nowcast[i]
            source.append("nowcast")
        else:
            rates[i] = w * nowcast[i] + (1.0 - w) * aifs[i]
            source.append(source_for_lead(lead))

    return ClassicalForecast(
        leads_min=np.asarray(leads, dtype="int32"),
        rates=np.clip(rates, 0.0, None).astype("float32"),
        source=source,
        confidence=confidence_for_leads(leads),
        engine=engine,
    )
