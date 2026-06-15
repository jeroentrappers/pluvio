"""Serve the trained UNet's nowcast.

The hetz1 inference step (research/model/infer_latest.py) runs the model on the
latest data every ~15 min and writes `model_nowcast.npz` — corrected rain fields
at leads [0, 30, 60, 90, 120] already reprojected onto our Belgium grid. Here we
read it, interpolate to the nowcast band's 10-min steps, and serve it. If the
file is missing or stale we fall back to the KMI stub so the API never goes dark.

Same signature as `stubs.stub_band` (a `BandInference`), so it drops into
`inference_worker.run_tick`. Only the nowcast band is model-backed; the longer
bands stay on the stub (the model only covers 0–120 min).
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


def model_band(
    client: httpx.Client,
    base_url: str,
    grid: GridSpec,
    band_name: schedules.BandName,
) -> tuple[np.ndarray, datetime]:
    # Model only covers the nowcast band; everything else stays on the stub.
    if band_name != "nowcast":
        return stub_band(client, base_url, grid, band_name)

    if not NPZ_PATH.exists():
        LOG.warning("model field %s missing — falling back to stub", NPZ_PATH)
        return stub_band(client, base_url, grid, band_name)

    # Any read/parse failure (permissions, half-written, corrupt) must not take
    # down the nowcast band — degrade to the stub instead.
    try:
        d = np.load(NPZ_PATH)
    except Exception as exc:
        LOG.warning("model field %s unreadable (%s) — falling back to stub", NPZ_PATH, exc)
        return stub_band(client, base_url, grid, band_name)
    issued_at = datetime.fromtimestamp(int(d["issue_epoch"]), tz=UTC)
    age = (datetime.now(UTC) - issued_at).total_seconds()
    if age > MAX_AGE_S:
        LOG.warning("model field stale (%.0f s > %d) — falling back to stub", age, MAX_AGE_S)
        return stub_band(client, base_url, grid, band_name)

    src_leads = [int(x) for x in d["leads"]]  # [0, 30, 60, 90, 120]
    src = d["rates"].astype("float32")  # (5, H, W) already on the Belgium grid
    band = schedules.band("nowcast")
    dst_leads = list(band.leads_min)  # [0, 10, …, 120]
    out = np.stack([_interp_lead(src, src_leads, L) for L in dst_leads]).astype("float32")
    LOG.info(
        "model nowcast served: issued=%s leads=%d max=%.2f mm/h",
        issued_at.isoformat(),
        len(dst_leads),
        float(out.max()),
    )
    return out, issued_at
