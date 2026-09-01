"""Forecast-vs-composite verification: replay archived nowcast runs against the
observed composite, with a signed difference view.

Data: forecast runs archived by tools/forecast_archive.py (one npz per issue,
leads x 100x100, f16) and the observed QPE daily zarrs. Observed frames are
area-resampled from the research grid onto the forecast grid/bounds; the
difference is forecast - observed in mm/h (positive = we over-forecast).
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
from datetime import UTC, datetime

import numpy as np

from .colormap import diff_rgba, rgba_for_array

LOG = logging.getLogger("pluvio.verify")

ARCHIVE_ROOT = pathlib.Path(os.environ.get("PLUVIO_FORECAST_ARCHIVE",
                                           "/storagebox/forecast_archive"))
QPE_ROOT = pathlib.Path(os.environ.get("PLUVIO_QPE_ROOT", "/storagebox/qpe"))
# research-grid bounds (W,S,E,N) — the day zarrs carry no attrs
QPE_BOUNDS = (1.5, 48.9, 7.5, 52.5)


def list_issues(limit: int = 96) -> list[dict]:
    out: list[dict] = []
    days = sorted(ARCHIVE_ROOT.glob("*/*/*"), reverse=True)
    for d in days:
        for f in sorted(d.glob("forecast_*.npz"), reverse=True):
            try:
                ts = datetime.strptime(
                    d.as_posix()[-10:] + f.stem.split("_")[1], "%Y/%m/%d%H%M"
                ).replace(tzinfo=UTC)
            except ValueError:
                continue
            out.append({"issue": int(ts.timestamp()),
                        "issued_at": ts.isoformat()})
            if len(out) >= limit:
                return out
    return out


def _forecast_path(issue: int) -> pathlib.Path | None:
    ts = datetime.fromtimestamp(issue, UTC)
    p = ARCHIVE_ROOT / f"{ts:%Y/%m/%d}" / f"forecast_{ts:%H%M}.npz"
    return p if p.exists() else None


def load_forecast(issue: int):
    p = _forecast_path(issue)
    if p is None:
        return None
    z = np.load(p, allow_pickle=False)
    return {"leads": z["leads"].astype(int), "rates": z["rates"].astype("float32"),
            "bounds": [float(x) for x in z["bounds"]]}


def _area_resample(a: np.ndarray, out_hw) -> np.ndarray:
    """Block-mean resample (downsampling only) without cv2."""
    H, W = a.shape
    oh, ow = out_hw
    ri = np.clip(np.linspace(0, H, oh + 1).astype(int), 0, H)
    ci = np.clip(np.linspace(0, W, ow + 1).astype(int), 0, W)
    rows = np.add.reduceat(a, ri[:-1], axis=0) / np.maximum(np.diff(ri), 1)[:, None]
    return np.add.reduceat(rows, ci[:-1], axis=1) / np.maximum(np.diff(ci), 1)[None, :]


def observed_on(valid_epoch: int, bounds, shape):
    """Observed composite at valid time, resampled onto the forecast grid.

    QPE day zarrs are slot-indexed: rate is (288, 768, 768) f16, slot k =
    (epoch % 86400) // 300, no time array and no attrs (bounds = QPE_BOUNDS).
    """
    import zarr

    ts = datetime.fromtimestamp(valid_epoch, UTC)
    zp = QPE_ROOT / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    if not zp.exists():
        return None
    root = zarr.open_group(str(zp), mode="r")
    slot = int(round((valid_epoch % 86400) / 300))
    if not 0 <= slot < root["rate"].shape[0]:
        return None
    rate = np.asarray(root["rate"][slot], dtype="float32")
    if not np.isfinite(rate).any():
        return None
    qb = QPE_BOUNDS
    H, W = rate.shape
    w, s, e, n = bounds
    c0 = int((w - qb[0]) / (qb[2] - qb[0]) * W)
    c1 = int((e - qb[0]) / (qb[2] - qb[0]) * W)
    r0 = int((qb[3] - n) / (qb[3] - qb[1]) * H)
    r1 = int((qb[3] - s) / (qb[3] - qb[1]) * H)
    if not (0 <= c0 < c1 <= W and 0 <= r0 < r1 <= H):
        return None
    return _area_resample(np.nan_to_num(rate[r0:r1, c0:c1]), shape)


def frame_png(issue: int, lead: int, kind: str) -> bytes | None:
    from PIL import Image

    fc = load_forecast(issue)
    if fc is None:
        return None
    leads = list(fc["leads"])
    if lead not in leads:
        return None
    li = leads.index(lead)
    f_rate = fc["rates"][li]
    if kind == "forecast":
        rgba = rgba_for_array(f_rate)
    else:
        obs = observed_on(issue + lead * 60, fc["bounds"], f_rate.shape)
        if obs is None:
            return None
        rgba = rgba_for_array(obs) if kind == "observed" else diff_rgba(f_rate - obs)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def scores(issue: int, lead: int) -> dict | None:
    fc = load_forecast(issue)
    if fc is None:
        return None
    leads = list(fc["leads"])
    if lead not in leads:
        return None
    f_rate = fc["rates"][leads.index(lead)]
    obs = observed_on(issue + lead * 60, fc["bounds"], f_rate.shape)
    if obs is None:
        return None
    out = {"bounds": fc["bounds"], "lead_min": lead,
           "bias_mm_h": round(float(np.mean(f_rate - obs)), 3),
           "mae_mm_h": round(float(np.mean(np.abs(f_rate - obs))), 3)}
    for thr in (0.1, 0.5, 1.0):
        p, o = f_rate > thr, obs > thr
        hit = int((p & o).sum()); miss = int((~p & o).sum()); fa = int((p & ~o).sum())
        out[f"csi_{thr}"] = round(hit / max(1, hit + miss + fa), 3)
    return out
