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

from .cache import edge_bounds
from .colormap import diff_rgba, rgba_for_array, upsample_field

LOG = logging.getLogger("pluvio.verify")

ARCHIVE_ROOT = pathlib.Path(os.environ.get("PLUVIO_FORECAST_ARCHIVE",
                                           "/storagebox/forecast_archive"))
QPE_ROOT = pathlib.Path(os.environ.get("PLUVIO_QPE_ROOT", "/storagebox/qpe"))
# research-grid bounds (W,S,E,N) — the day zarrs carry no attrs. CELL-CENTRE
# bounds, like every other `bounds` in this codebase (see cache.GridSpec):
# converted to pixel edges with `edge_bounds()` before any cropping/painting.
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
    H, W = rate.shape
    # Both boxes are CELL-CENTRE bounds, so the crop window is computed
    # between the two EDGE frames — the one backend pixel convention (1.13).
    # Mixing them (centre bounds against a whole-pixel-count index) shifted
    # the window by half a QPE cell on each side.
    qw, qs, qe, qn = edge_bounds(QPE_BOUNDS, (H, W))
    fw, fs, fe, fn = (float(x) for x in bounds)
    w, s, e, n = edge_bounds((fw, fs, fe, fn), (int(shape[0]), int(shape[1])))
    c0 = round((w - qw) / (qe - qw) * W)
    c1 = round((e - qw) / (qe - qw) * W)
    r0 = round((qn - n) / (qn - qs) * H)
    r1 = round((qn - s) / (qn - qs) * H)
    if not (0 <= c0 < c1 <= W and 0 <= r0 < r1 <= H):
        return None
    return _area_resample(np.nan_to_num(rate[r0:r1, c0:c1]), shape)


def issue_meta(issue: int) -> dict | None:
    """Leads available for one archived run (the UI's scrubber range)."""
    fc = load_forecast(issue)
    if fc is None:
        return None
    return {"issue": issue, "leads": [int(x) for x in fc["leads"]],
            "bounds": fc["bounds"]}


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
    w, s_, e, n = fc["bounds"]
    up = (max(round((n - s_) * 74.0), f_rate.shape[0]),
          max(round((e - w) * 46.51), f_rate.shape[1]))
    if kind == "forecast":
        rgba = rgba_for_array(upsample_field(f_rate, up))
    else:
        obs = observed_on(issue + lead * 60, fc["bounds"], f_rate.shape)
        if obs is None:
            return None
        rgba = (rgba_for_array(upsample_field(obs, up)) if kind == "observed"
                else diff_rgba(upsample_field(f_rate, up) - upsample_field(obs, up)))
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
