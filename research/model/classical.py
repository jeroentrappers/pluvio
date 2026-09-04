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
    # (dy, dx) px by which the NWP field was shifted onto the radar frame at
    # the nowcast horizon (None when the blend had no NWP or was not corrected).
    phase_offset_px: tuple[float, float] | None = None



def global_motion_robust(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Per-frame (dy, dx) content motion prev→curr, from the same bounded
    smoothed cross-correlation the NWP phase offset uses (the whitened FFT
    estimator is fragile on sparse fields)."""
    dy, dx = nwp_phase_offset(curr, prev)  # shift that puts prev's rain where curr has it
    return dy, dx


def anchored_blend(
    rates: np.ndarray,
    source: list[str],
    leads_min,
    *,
    anchor_field: np.ndarray,
    anchor_lead_min: int,
    motion_per_frame: tuple[float, float],
    dt_min: float,
    aifs_rates: np.ndarray | None,
    phase_correct: bool = True,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Rebuild every lead past ``anchor_lead_min`` up to the blend horizon so
    the radar arm CONTINUES ``anchor_field`` (the last field the served
    nowcast actually shows) instead of a separate extrapolation from t0.

    The hybrid cube splices a learned nowcast (0–120 min) onto a classical
    cube whose 2–6 h radar arm was pysteps' own extrapolation of the OPERA
    history: two different nowcasts of the same rain, 20+ cells apart on
    2026-09-04, so the timeline jumped at 120→180 min. Here the radar arm is
    the anchor advected by ``motion_per_frame`` per ``dt_min``, the NWP is
    phase-corrected against the anchor, and the blend weights are unchanged.
    Returns (rates, phase_offset_px); leads at or before the anchor and past
    the blend horizon are left untouched.
    """
    leads = [int(x) for x in leads_min]
    out = np.array(rates, dtype="float32", copy=True)
    anchor = np.nan_to_num(np.asarray(anchor_field, dtype="float32"), nan=0.0)
    dy, dx = motion_per_frame
    aifs = None if aifs_rates is None else np.nan_to_num(np.asarray(aifs_rates, dtype="float32"), nan=0.0)
    offset: tuple[float, float] | None = None
    if aifs is not None and phase_correct and anchor_lead_min in leads:
        offset = nwp_phase_offset(anchor, aifs[leads.index(anchor_lead_min)])
    for i, lead in enumerate(leads):
        if lead <= anchor_lead_min or lead > BLEND_END_MIN + PHASE_RELAX_MIN:
            continue
        w = _blend_weight(lead)
        step = (lead - anchor_lead_min) / dt_min
        radar_arm = _advect_semilagrangian(anchor, dy, dx, step) if w > 0.0 else None
        if aifs is None:
            out[i] = radar_arm
            continue
        nwp = aifs[i]
        r = _phase_relax(lead)
        if offset is not None and offset != (0.0, 0.0) and r > 0.0:
            nwp = _advect_semilagrangian(nwp, offset[0] * r, offset[1] * r, 1)
        out[i] = nwp if radar_arm is None else w * radar_arm + (1.0 - w) * nwp
    return np.clip(out, 0.0, None).astype("float32"), offset


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


MIN_WET_FRAC_FOR_PHASE = 0.005   # both fields need rain to estimate an offset
PHASE_RELAX_MIN = 720            # relax the NWP phase correction over 12 h past the blend
MAX_PHASE_SHIFT_FRAC = 0.25      # never shift NWP by more than a quarter grid


MIN_PHASE_CORR = 0.2             # peak normalised correlation needed to trust an offset
PHASE_SMOOTH_SIGMA_PX = 3.0      # smooth both fields before matching (cells → blobs)


def nwp_phase_offset(radar_field: np.ndarray, nwp_field: np.ndarray,
                     wet_thr: float = 0.1) -> tuple[float, float]:
    """(dy, dx) px to advect ``nwp_field`` by so its rain sits where
    ``radar_field`` has it.

    Bounded cross-correlation of the two log-rate fields after Gaussian
    smoothing (a radar cell and the NWP's broad rain area are matched as
    blobs, not pixel patterns), peak searched only within a quarter grid.
    Zero when either field is essentially dry or the peak's normalised
    correlation is below ``MIN_PHASE_CORR`` — then the two fields describe
    different weather and moving the NWP would fake agreement rather than
    remove a phase error. Whitened FFT phase correlation was tried first and
    returned a spurious (113, -102) px on the first live field (2026-09-04).
    """
    from scipy.ndimage import gaussian_filter

    a = np.nan_to_num(np.asarray(radar_field, dtype="float32"), nan=0.0)
    b = np.nan_to_num(np.asarray(nwp_field, dtype="float32"), nan=0.0)
    if (a > wet_thr).mean() < MIN_WET_FRAC_FOR_PHASE or (b > wet_thr).mean() < MIN_WET_FRAC_FOR_PHASE:
        return 0.0, 0.0
    h, w = a.shape
    sa = gaussian_filter(np.log1p(np.maximum(a, 0.0)), PHASE_SMOOTH_SIGMA_PX)
    sb = gaussian_filter(np.log1p(np.maximum(b, 0.0)), PHASE_SMOOTH_SIGMA_PX)
    sa -= sa.mean()
    sb -= sb.mean()
    na, nb = float(np.sqrt((sa * sa).sum())), float(np.sqrt((sb * sb).sum()))
    if na < 1e-6 or nb < 1e-6:
        return 0.0, 0.0
    # Zero-padded (non-circular) cross-correlation: corr[dy, dx] = sum sa(y, x) * sb(y - dy, x - dx),
    # i.e. how well sb SHIFTED BY (dy, dx) matches sa.
    fa = np.fft.rfft2(sa, s=(2 * h, 2 * w))
    fb = np.fft.rfft2(sb, s=(2 * h, 2 * w))
    corr = np.fft.irfft2(fa * np.conj(fb), s=(2 * h, 2 * w)) / (na * nb)
    my, mx = int(MAX_PHASE_SHIFT_FRAC * h), int(MAX_PHASE_SHIFT_FRAC * w)
    dys = np.arange(-my, my + 1)
    dxs = np.arange(-mx, mx + 1)
    window = corr[np.ix_(dys % (2 * h), dxs % (2 * w))]
    k = np.unravel_index(int(np.argmax(window)), window.shape)
    peak = float(window[k])
    if peak < MIN_PHASE_CORR:
        LOG.info("nwp phase offset: peak corr %.2f < %.2f — fields too dissimilar, not applied",
                 peak, MIN_PHASE_CORR)
        return 0.0, 0.0
    return float(dys[k[0]]), float(dxs[k[1]])


def _phase_relax(lead_min: float) -> float:
    """Fraction of the NWP phase correction still applied at ``lead_min``:
    1.0 through the blend window, then linearly to 0.0 over PHASE_RELAX_MIN."""
    if lead_min <= BLEND_END_MIN:
        return 1.0
    frac = (lead_min - BLEND_END_MIN) / PHASE_RELAX_MIN
    return float(max(0.0, 1.0 - frac))


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
    phase_correct: bool = True,
) -> ClassicalForecast:
    """Assemble the full 0–240 h classical cube.

    The 2–6 h handoff is NOT a pointwise cross-fade any more (2.8): with
    ``phase_correct`` the NWP field is shifted onto the radar frame by the
    phase offset measured at the nowcast horizon and kept there through the
    whole blend window, so the composition stays co-located with the radar
    cells and moves with them; the shift then relaxes linearly to zero over
    the next ``PHASE_RELAX_MIN`` minutes of the pure-NWP outlook, where the
    hourly cadence carries no expectation of cell continuity. A pointwise
    fade between two fields that place rain in different spots moves the
    visible centre of mass from one to the other as the weight decays —
    cells appeared to travel backwards against their own motion (measured
    2026-09-03: eastward radar cells, NWP rain 10–20 cells west). Relaxing
    the shift inside the blend window would reproduce that drift, only
    smoother, so it is deferred past it.

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

    offset: tuple[float, float] | None = None
    if aifs is not None and phase_correct:
        # Offset at the nowcast horizon: the last lead the radar arm owns
        # outright (w == 1), or the first blend lead when 120 is not served.
        anchor = max([i for i, l in enumerate(leads) if l <= NOWCAST_END_MIN] or [0])
        offset = nwp_phase_offset(nowcast[anchor], aifs[anchor])
        if offset != (0.0, 0.0):
            LOG.info("nwp phase offset at %d min: dy=%.1f dx=%.1f px (held through %d min, "
                     "relaxed to 0 by %d min)", leads[anchor], offset[0], offset[1],
                     BLEND_END_MIN, BLEND_END_MIN + PHASE_RELAX_MIN)

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
            nwp = aifs[i]
            r = _phase_relax(lead)
            if offset is not None and offset != (0.0, 0.0) and r > 0.0:
                nwp = _advect_semilagrangian(nwp, offset[0] * r, offset[1] * r, 1)
            rates[i] = w * nowcast[i] + (1.0 - w) * nwp
            source.append(source_for_lead(lead))

    return ClassicalForecast(
        leads_min=np.asarray(leads, dtype="int32"),
        rates=np.clip(rates, 0.0, None).astype("float32"),
        source=source,
        confidence=confidence_for_leads(leads),
        engine=engine,
        phase_offset_px=offset,
    )
