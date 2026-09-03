"""Serve the precomputed forecast cube — classical baseline or learned model.

A hetz1 producer (research/model/produce_forecast.py) runs on a schedule and
writes `model_forecast.npz` — a seamless 0–240 h cube already reprojected onto
the Belgium grid, with per-lead **source** tags and **confidence**. This module
reads it and serves **every** band (nowcast → long) from it, interpolating the
cube to each band's lead steps.

Crucially, the served artifact is producer-agnostic (recommendation #4): it is
the *classical* pysteps⊕AIFS baseline by default, and the learned `SeamlessNet`
only once it has been promoted through the champion/challenger gate. The backend
neither knows nor cares which produced the file — it just serves it and surfaces
the `source`/`confidence` provenance so the product is honest about where each
lead's number came from.

Fallback chain (the API never goes dark):
    model_forecast.npz  (full horizon, any band)
      → model_nowcast.npz  (legacy nowcast-only UNet, nowcast band)
        → KMI stub  (any band)

`model_band` keeps the `BandInference` signature so it drops into
`inference_worker.run_tick`; `band_provenance` exposes the source/confidence for
the worker to fold into the snapshot's grid.json.
"""

from __future__ import annotations

import logging
import os
import pathlib
from datetime import UTC, datetime

import httpx
import numpy as np

from . import schedules
from .cache import GridSpec
from .stubs import stub_band

LOG = logging.getLogger("pluvio.model")

# Full-horizon cube (classical baseline or promoted model) — preferred.
FORECAST_NPZ_PATH = pathlib.Path(
    os.environ.get("PLUVIO_MODEL_FORECAST_NPZ", "/opt/pluvio/serve/model_forecast.npz")
)
# Legacy nowcast-only artifact (the original UNet) — back-compat fallback.
NPZ_PATH = pathlib.Path(
    os.environ.get("PLUVIO_MODEL_NOWCAST_NPZ", "/opt/pluvio/serve/model_nowcast.npz")
)
MAX_AGE_S = int(os.environ.get("PLUVIO_MODEL_MAX_AGE_S", "5400"))  # 90 min


def _interp_lead(src: np.ndarray, src_leads: list[int], lead: int) -> np.ndarray:
    """Linear interpolation of the (n_src, H, W) field to one target lead."""
    if lead <= src_leads[0]:
        return src[0]
    if lead >= src_leads[-1]:
        return src[-1]
    for j in range(len(src_leads) - 1):
        a, b = src_leads[j], src_leads[j + 1]
        if a <= lead <= b:
            w = (lead - a) / (b - a)
            return src[j] * (1 - w) + src[j + 1] * w
    return src[-1]


def _load_fresh(path: pathlib.Path):
    """Load an npz if it exists and is fresh; else None. Never raises."""
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as exc:
        LOG.warning("forecast field %s unreadable (%s)", path, exc)
        return None
    issued_at = datetime.fromtimestamp(int(d["issue_epoch"]), tz=UTC)
    age = (datetime.now(UTC) - issued_at).total_seconds()
    if age > MAX_AGE_S:
        LOG.warning("forecast field %s stale (%.0f s > %d)", path, age, MAX_AGE_S)
        return None
    return d, issued_at


def _band_from_cube(d, issued_at, band_name: schedules.BandName):
    """Interpolate the full cube onto one band's lead steps.

    Nowcast band: MOTION-morphed intermediates (cells move between the
    model's native leads instead of cross-fading in place) — this is what
    keeps the seamless timeline temporally consistent across t=0. Other
    bands keep linear interpolation (their steps match the source spacing).
    """
    src_leads = [int(x) for x in d["leads"]]
    # produce_forecast.py already nan_to_num's before writing the cube, but
    # morph.flow_for_pair/morph_pair raise on non-finite input (2.7) — belt
    # and suspenders so a stale/malformed artifact can't take serving down.
    src = np.nan_to_num(d["rates"].astype("float32"))
    band = schedules.band(band_name)
    if band_name != "nowcast":
        out = np.stack([_interp_lead(src, src_leads, L) for L in band.leads_min]).astype("float32")
        return out, issued_at

    from .morph import flow_for_pair, morph_pair

    flows: dict[tuple[int, int], np.ndarray] = {}
    frames = []
    for L in band.leads_min:
        if L <= src_leads[0]:
            frames.append(src[0])
            continue
        if L >= src_leads[-1]:
            frames.append(src[-1])
            continue
        j = max(i for i in range(len(src_leads)) if src_leads[i] <= L)
        a_lead, b_lead = src_leads[j], src_leads[j + 1]
        if L == a_lead:
            frames.append(src[j])
            continue
        key = (a_lead, b_lead)
        if key not in flows:
            flows[key] = flow_for_pair(src[j], src[j + 1])
        w = (L - a_lead) / (b_lead - a_lead)
        frames.append(morph_pair(src[j], src[j + 1], w, flow=flows[key]))
    out = np.stack(frames).astype("float32")
    return out, issued_at


def _area_resample(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Block-mean resample (downsampling) without cv2."""
    H, W = a.shape
    oh, ow = out_hw
    ri = np.clip(np.linspace(0, H, oh + 1).astype(int), 0, H)
    ci = np.clip(np.linspace(0, W, ow + 1).astype(int), 0, W)
    rows = np.add.reduceat(a, ri[:-1], axis=0) / np.maximum(np.diff(ri), 1)[:, None]
    return (np.add.reduceat(rows, ci[:-1], axis=1) / np.maximum(np.diff(ci), 1)[None, :]).astype("float32")


def _lagrangian_blend(out: np.ndarray, leads_min: list[int], issued_at: datetime,
                      grid: GridSpec) -> np.ndarray:
    """Anchor the seam: continue the OBSERVED composite into the forecast.

    The v2 artifact's issue time lags wall clock by 30-70 min (store-append
    latency), so the first frames the timeline shows after t=0 are 60-90-min
    leads — smooth, and blind to the last hour of real cell drift: cells
    visibly "stopped" or vanished at the seam. Fix, per nowcast lead:
      valid time in the observed past  → the observed frame itself;
      first future hour               → advected latest observation
                                         cross-faded into the model field
                                         (w: cos², 1 → 0 over 60 min);
      beyond                          → pure model field.
    Advection uses the same block-matching flow as the temporal morph,
    estimated from the last ~17 min of observed frames.
    """
    import os

    path = pathlib.Path(os.environ.get("PLUVIO_OBSERVED_NPZ", "/opt/pluvio/serve/observed.npz"))
    if not path.exists():
        return out
    try:
        z = np.load(path, allow_pickle=False)
        times = z["times"].astype("int64")
        rates = z["rates"]
        W0, S0, E0, N0 = (float(x) for x in z["bounds"])
    except Exception as exc:  # never break serving over the blend
        LOG.warning("lagrangian blend: observed cube unreadable (%s)", exc)
        return out
    if len(times) < 12:
        return out
    b = grid.bounds
    gh, gw = rates.shape[1:]
    c0 = int((b["west"] - W0) / (E0 - W0) * gw)
    c1 = int((b["east"] - W0) / (E0 - W0) * gw)
    r0 = int((N0 - b["north"]) / (N0 - S0) * gh)
    r1 = int((N0 - b["south"]) / (N0 - S0) * gh)
    if not (0 <= c0 < c1 <= gw and 0 <= r0 < r1 <= gh):
        return out

    from .morph import _warp, flow_for_pair

    def obs_at(i: int) -> np.ndarray:
        return _area_resample(np.nan_to_num(np.asarray(rates[i, r0:r1, c0:c1], dtype="float32")),
                              grid.shape)

    newest = obs_at(len(times) - 1)
    older = obs_at(len(times) - 11)
    span_min = max(1.0, (int(times[-1]) - int(times[-11])) / 60.0)
    fy, fx = flow_for_pair(older, newest)  # displacement over span_min
    t_obs = int(times[-1])
    issue_e = int(issued_at.timestamp())

    blended = out.copy()
    for k, lead in enumerate(leads_min):
        dt_min = (issue_e + lead * 60 - t_obs) / 60.0
        if dt_min <= 0:
            # valid time lies in the observed record: use the observation itself
            j = int(np.argmin(np.abs(times - (issue_e + lead * 60))))
            blended[k] = obs_at(j)
            continue
        w = float(np.cos(np.pi / 2 * min(dt_min, 60.0) / 60.0) ** 2)
        if w <= 0.0:
            continue
        scale = dt_min / span_min
        # _warp(f, D) translates content BY +D, so forward extrapolation is
        # +scale*flow. (A negative sign here advected cells BACKWARDS along
        # their own motion — cells visibly reversed direction at the seam.)
        adv = _warp(newest, scale * fy, scale * fx)
        blended[k] = np.clip(w * adv + (1 - w) * out[k], 0.0, None)
    LOG.info("nowcast lagrangian blend: obs_age=%.0f min issue_age=%.0f min flow_span=%.0f min",
             (datetime.now(UTC).timestamp() - t_obs) / 60,
             (datetime.now(UTC).timestamp() - issue_e) / 60, span_min)
    return blended


def model_band(
    client: httpx.Client,
    base_url: str,
    grid: GridSpec,
    band_name: schedules.BandName,
) -> tuple[np.ndarray, datetime]:
    # 1) The nowcast band prefers the dedicated nowcast artifact (the v2
    #    correction UNet, refreshed every 15 min by the infer_latest cron).
    #    Measured 2026-09-02: it tracks observed light-rain coverage almost
    #    exactly (wet>0.1: 4.5% vs 4.6% observed) where the full-horizon cube
    #    smears drizzle away (1.7% at lead 0, 0.4-0.6% by 30-60 min) — the
    #    cube used to win here purely by load order.
    if band_name == "nowcast":
        loaded = _load_fresh(NPZ_PATH)
        if loaded is not None:
            d, issued_at = loaded
            out, _ = _band_from_cube(d, issued_at, band_name)
            out = _lagrangian_blend(out, schedules.band(band_name).leads_min,
                                    issued_at, grid)
            LOG.info("nowcast served from v2 npz + observed blend: issued=%s max=%.2f mm/h",
                     issued_at.isoformat(), float(out.max()))
            return out, issued_at

    # 2) full-horizon cube serves every band (and the nowcast as fallback).
    loaded = _load_fresh(FORECAST_NPZ_PATH)
    if loaded is not None:
        d, issued_at = loaded
        out, _ = _band_from_cube(d, issued_at, band_name)
        LOG.info("forecast(%s) served from cube: producer=%s issued=%s max=%.2f mm/h",
                 band_name, str(d["producer"]) if "producer" in d else "?",
                 issued_at.isoformat(), float(out.max()))
        return out, issued_at

    # 3) stub keeps the API alive.
    LOG.warning("no fresh forecast artifact for band=%s — falling back to stub", band_name)
    return stub_band(client, base_url, grid, band_name)


def band_provenance(band_name: schedules.BandName) -> dict | None:
    """Source/confidence provenance for a band, read from the forecast cube.

    Returned to the worker for grid.json so the product can honestly label each
    horizon ("radar nowcast" vs "NWP outlook") and widen its uncertainty band.
    Reports the source at the band's representative (mid) lead and the mean
    confidence across the band. None if the cube is absent/stale (stub serving).
    """
    loaded = _load_fresh(FORECAST_NPZ_PATH)
    if loaded is None:
        return None
    d, _ = loaded
    if "source" not in d or "confidence" not in d:
        return None
    src_leads = np.asarray([int(x) for x in d["leads"]])
    sources = [str(s) for s in d["source"]]
    conf = np.asarray(d["confidence"], dtype="float32")
    band = schedules.band(band_name)
    lo, hi = band.lead_min_start, band.lead_min_end
    in_band = (src_leads >= lo) & (src_leads < hi)
    if not in_band.any():
        # nearest lead to the band centre
        mid = (lo + hi) // 2
        j = int(np.argmin(np.abs(src_leads - mid)))
        return {"source": sources[j], "confidence": float(conf[j]),
                "producer": str(d["producer"]) if "producer" in d else "unknown"}
    mid_lead = (lo + min(hi, int(src_leads.max()) + 1)) // 2
    j = int(np.argmin(np.abs(src_leads - mid_lead)))
    return {
        "source": sources[j],
        "confidence": float(conf[in_band].mean()),
        "producer": str(d["producer"]) if "producer" in d else "unknown",
    }
