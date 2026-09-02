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
    src = d["rates"].astype("float32")
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
            LOG.info("nowcast served from v2 npz: issued=%s max=%.2f mm/h",
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
