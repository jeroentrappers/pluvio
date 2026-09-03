"""Forecast-vs-composite verification: replay archived nowcast runs against the
observed composite, with a signed difference view.

Data: forecast runs archived by tools/forecast_archive.py (one npz per issue,
leads x 100x100, f16) and the observed QPE daily zarrs. Observed frames are
area-averaged from the research grid onto the forecast grid/bounds; the
difference is forecast - observed in mm/h (positive = we over-forecast).

⚠️ The QPE day zarrs are NOT on the forecast serving box, and their bounds are
not derivable here. tools/qpe_archive.py composites onto its own analysis grid
at PLUVIO_GRID_N (768 in production) and states the result in the store's
`bounds` attr; that attr is the only source this module accepts. Treating the
serving box (1.5, 48.9, 7.5, 52.5) as the composite's bounds, as this module
did, squashed the whole 768² composite onto the 100² serving box and read
station truth ~237 km from the station. Re-deriving the bounds from the
research package instead would be ~60 km out at the south edge, because the
production archiver runs an older model/geo.py (untrimmed extent, zero bias) —
which is why there is no fallback: a store without the attr is an error.

The serving box is a small window on the composite, so the composite is
area-averaged over it rather than stretched onto it, and any target cell the
composite does not sufficiently cover comes back NaN — unobserved, not dry.
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
from datetime import UTC, datetime

import numpy as np

from .colormap import diff_rgba, rgba_for_array, upsample_field

LOG = logging.getLogger("pluvio.verify")

ARCHIVE_ROOT = pathlib.Path(os.environ.get("PLUVIO_FORECAST_ARCHIVE",
                                           "/storagebox/forecast_archive"))
QPE_ROOT = pathlib.Path(os.environ.get("PLUVIO_QPE_ROOT", "/storagebox/qpe"))

class QpeGeometryError(RuntimeError):
    """A QPE day zarr that does not state its own georeference."""


BACKFILL_HINT = ("re-run tools/qpe_archive.py (>= 1b6f023) for that day, or backfill the "
                 "attr from the archiver's own model.geo — never from the research "
                 "checkout's, which resolves a different extent and bias")


def _store_bounds(root, path: pathlib.Path) -> tuple[float, float, float, float]:
    """The composite's own `bounds` attr — [west, south, east, north], outer
    edges. Mandatory: there is no fallback, because every candidate is wrong.
    The forecast serving box is a different grid (that was the bug), and
    re-deriving the analysis grid from the research package gives an extent
    ~60 km off at the south edge, since the archiver runs an older
    model/geo.py than the repo's. Raises QpeGeometryError, naming the store.
    """
    raw = dict(getattr(root, "attrs", {}) or {}).get("bounds")
    if raw is None:
        raise QpeGeometryError(
            f"{path} carries no 'bounds' attr, so its georeference is unknown. "
            f"Fix: {BACKFILL_HINT}."
        )
    try:
        vals = tuple(float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise QpeGeometryError(f"{path}: unparseable 'bounds' attr {raw!r}") from exc
    if len(vals) != 4 or not (vals[0] < vals[2] and vals[1] < vals[3]):
        raise QpeGeometryError(
            f"{path}: implausible 'bounds' attr {raw!r} — expected "
            "[west, south, east, north] with west < east and south < north"
        )
    return vals[0], vals[1], vals[2], vals[3]


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


MIN_BLOCK_COVERAGE = 0.5


def _regrid_block_mean(src: np.ndarray, src_bounds, out_bounds, out_hw,
                       min_coverage: float = MIN_BLOCK_COVERAGE) -> np.ndarray:
    """Area-average `src` (regular lat/lon, row 0 = north, both boxes EDGE
    referenced) onto out_bounds/out_hw, without cv2.

    The two boxes need not be nested: a target cell whose source footprint is
    less than `min_coverage` finite — including one wholly outside the source
    domain — comes back NaN, so "uncovered" never reads as "measured dry".
    Mirrors research/tools/scoreboard.py's helper of the same name.
    """
    H, W = src.shape
    qw, qs, qe, qn = (float(x) for x in src_bounds)
    ow_, os_, oe_, on_ = (float(x) for x in out_bounds)
    oh, owd = int(out_hw[0]), int(out_hw[1])

    r_f = (qn - np.linspace(on_, os_, oh + 1)) / (qn - qs) * H  # increasing
    c_f = (np.linspace(ow_, oe_, owd + 1) - qw) / (qe - qw) * W
    r_e = np.rint(r_f).astype("int64")
    c_e = np.rint(c_f).astype("int64")
    r0, r1 = np.clip(r_e[:-1], 0, H), np.clip(r_e[1:], 0, H)
    c0, c1 = np.clip(c_e[:-1], 0, W), np.clip(c_e[1:], 0, W)
    # Target finer than the source: an empty-but-inside span takes the single
    # source cell containing it (a wholly-outside span stays empty -> NaN).
    for lo, hi, f_lo, f_hi, limit in (
        (r0, r1, r_f[:-1], r_f[1:], H),
        (c0, c1, c_f[:-1], c_f[1:], W),
    ):
        thin = (hi <= lo) & (f_lo < limit) & (f_hi > 0)
        lo[thin] = np.minimum(lo[thin], limit - 1)
        hi[thin] = lo[thin] + 1

    finite = np.isfinite(src)
    sums = np.zeros((H + 1, W + 1), "float64")
    counts = np.zeros((H + 1, W + 1), "float64")
    # float64 before the cumsum: a 768^2 composite runs the running total to
    # ~1e6, where float32 spacing is ~0.06 — up to ~1e-2 mm/h of error in a
    # block mean near the far corner.
    sums[1:, 1:] = np.where(finite, src.astype("float64"), 0.0).cumsum(0).cumsum(1)
    counts[1:, 1:] = finite.astype("float64").cumsum(0).cumsum(1)

    def _box(a: np.ndarray) -> np.ndarray:
        return (
            a[np.ix_(r1, c1)] - a[np.ix_(r0, c1)]
            - a[np.ix_(r1, c0)] + a[np.ix_(r0, c0)]
        )

    total, count = _box(sums), _box(counts)
    size = ((r1 - r0)[:, None] * (c1 - c0)[None, :]).astype("float64")
    covered = (size > 0) & (count > 0) & (count >= min_coverage * size)
    return np.where(covered, total / np.maximum(count, 1.0), np.nan).astype("float32")


def observed_on(valid_epoch: int, bounds, shape):
    """Observed composite at valid time, area-averaged onto the forecast grid.

    QPE day zarrs are slot-indexed: rate is (288, N, N) f16 on the archiver's
    analysis grid (N = PLUVIO_GRID_N, 768 in production), slot k = the 5-min
    slot nearest the valid time, no time array. Their georeference comes from
    the store's mandatory `bounds` attr — never from the forecast grid, and
    never re-derived here (see _store_bounds). Cells the composite does not
    cover come back NaN. Returns None when the day/slot is not archived;
    raises QpeGeometryError when the store is there but says nothing about
    where it is.
    """
    import zarr

    # Snap to the nearest 5-min slot first, then take the day from the SNAPPED
    # epoch: a valid time in the last 150 s of a day belongs to the next day's
    # slot 0, not off the end of this day's array.
    snapped = round(valid_epoch / 300) * 300
    ts = datetime.fromtimestamp(snapped, UTC)
    zp = QPE_ROOT / f"{ts:%Y/%m}" / f"{ts:%d}.zarr"
    if not zp.exists():
        return None
    root = zarr.open_group(str(zp), mode="r")
    rate_arr = root["rate"]
    if not isinstance(rate_arr, zarr.Array):
        return None
    slot = (snapped % 86400) // 300
    if not 0 <= slot < rate_arr.shape[0]:
        return None
    rate = np.asarray(rate_arr[slot], dtype="float32")
    if not np.isfinite(rate).any():
        return None
    out = _regrid_block_mean(rate, _store_bounds(root, zp), bounds, shape)
    return out if np.isfinite(out).any() else None


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
    # Only cells the composite actually observed. Radar coverage does not
    # reach the whole serving box, and scoring the uncovered part as dry would
    # invent skill (or invent misses) over a large, weather-dependent slice.
    valid = np.isfinite(f_rate) & np.isfinite(obs)
    if not valid.any():
        return None
    fv, ov = f_rate[valid], obs[valid]
    out = {"bounds": fc["bounds"], "lead_min": lead,
           "n_valid": int(valid.sum()),
           "bias_mm_h": round(float(np.mean(fv - ov)), 3),
           "mae_mm_h": round(float(np.mean(np.abs(fv - ov))), 3)}
    for thr in (0.1, 0.5, 1.0):
        p, o = fv > thr, ov > thr
        hit = int((p & o).sum())
        miss = int((~p & o).sum())
        fa = int((p & ~o).sum())
        out[f"csi_{thr}"] = round(hit / max(1, hit + miss + fa), 3)
    return out
