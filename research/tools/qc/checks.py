"""Pure, numpy-only QC check functions shared by qc_inputs and qc_watchdog.

Nothing here opens a store, a zarr group, or the network — every function
takes arrays (or small closures over arrays) and a `Thresholds` and returns
either a plain value/dict or a `verdict.Check`. The CLIs (`qc_inputs.py`,
`qc_watchdog.py`) do the I/O and call these.
"""

from __future__ import annotations

import glob
import pathlib
from datetime import UTC, datetime

import numpy as np

from .verdict import Check


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.ravel(), y.ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# qc_inputs: registration, aux alignment, channel health, staleness
# ---------------------------------------------------------------------------

def registration_offset(
    field: np.ndarray,
    sampler,
    *,
    search_deg: float = 0.14,
    step_deg: float = 0.02,
) -> tuple[float, float, float]:
    """Grid-search the (dlat, dlon) shift that best aligns `sampler(dlat,
    dlon)` with `field`, by cross-correlation.

    `sampler(dlat, dlon) -> np.ndarray` must return an array shaped like
    `field` (a resample of some reference grid displaced by the given
    offset). Returns (best_corr, best_dlat, best_dlon); best_corr is -9.0
    if every candidate correlation was NaN (e.g. a flat field).
    """
    best = (-9.0, 0.0, 0.0)
    for dlat in np.arange(-search_deg, search_deg + step_deg / 2, step_deg):
        for dlon in np.arange(-search_deg, search_deg + step_deg / 2, step_deg):
            c = _corr(field, sampler(float(dlat), float(dlon)))
            if c == c and c > best[0]:
                best = (c, float(dlat), float(dlon))
    return best


def aggregate_registration(fits: list[tuple[float, float, float]], thresholds) -> Check:
    """Median a list of (corr, dlat, dlon) fits (one per issue) into one
    Check, warning when the offset or correlation crosses threshold."""
    if not fits:
        return Check("registration", "ok",
                      {"n": 0, "note": "no wet overlapping issues to fit"})
    arr = np.array(fits, dtype=float)
    corr = round(float(np.median(arr[:, 0])), 3)
    dlat = round(float(np.median(arr[:, 1])), 3)
    dlon = round(float(np.median(arr[:, 2])), 3)
    value = {"corr": corr, "dlat": dlat, "dlon": dlon, "n": len(fits)}
    threshold = {"offset_deg": thresholds.reg_offset_warn_deg, "corr": thresholds.reg_corr_warn}
    details = []
    status = "ok"
    if abs(dlat) > thresholds.reg_offset_warn_deg or abs(dlon) > thresholds.reg_offset_warn_deg:
        status = "warn"
        details.append(f"offset (dlat={dlat}, dlon={dlon})")
    if corr < thresholds.reg_corr_warn:
        status = "warn"
        details.append(f"corr {corr} < {thresholds.reg_corr_warn}")
    return Check("registration", status, value, threshold, "; ".join(details))


def signed_corr(radar: np.ndarray, aux: np.ndarray, sign: int) -> float:
    """Correlation of `radar` against `aux`, sign-flipped by `sign` so a
    physically-coupled aux channel (e.g. cold IR tops <-> rain, sign=-1)
    reads positive when aligned. NaN if either field is ~constant."""
    c = _corr(radar, np.nan_to_num(aux))
    return float("nan") if c != c else sign * c


def aggregate_aux_alignment(name: str, corrs: list[float], thresholds) -> Check:
    """Median a list of signed correlations (one per issue) for one aux
    channel into a Check, warning below `thresholds.aux_corr_warn`."""
    if not corrs:
        return Check(f"aux_alignment:{name}", "ok", None, thresholds.aux_corr_warn,
                      "no wet overlapping issues")
    med = round(float(np.median(corrs)), 3)
    value = {"signed_corr": med, "n": len(corrs)}
    if med < thresholds.aux_corr_warn:
        return Check(f"aux_alignment:{name}", "warn", value, thresholds.aux_corr_warn,
                      f"{name} signed corr {med} < {thresholds.aux_corr_warn}")
    return Check(f"aux_alignment:{name}", "ok", value, thresholds.aux_corr_warn)


def channel_health(block: np.ndarray, name: str, thresholds) -> Check:
    """NaN fraction and value-range health of one channel's recent block
    of issues. `block` is already sliced to the window under test.

    The range check compares the block's [100-P, P] percentile (P =
    `thresholds.range_percentile`, default 99.9) to the configured band,
    not the hard min/max — min/max are still reported for visibility, but
    a single outlier cell (bad IDW sample, one noisy report) over 48 issues
    x a whole grid must not page anyone; a real regression moves the bulk
    of the distribution, which the percentile catches.
    """
    nanfrac = round(float(np.mean(~np.isfinite(block))), 3)
    fin = block[np.isfinite(block)]
    vmin = round(float(fin.min()), 2) if fin.size else None
    vmax = round(float(fin.max()), 2) if fin.size else None
    value = {"nan_frac": nanfrac, "min": vmin, "max": vmax}
    rng = thresholds.range_for(name)
    details = []
    status = "ok"
    if nanfrac > thresholds.nan_limit:
        status = "warn"
        details.append(f"{name} {int(nanfrac * 100)}% NaN over last {block.shape[0]} issues")
    if rng and fin.size:
        pct = getattr(thresholds, "range_percentile", 99.9)
        p_lo = round(float(np.percentile(fin, 100.0 - pct)), 2)
        p_hi = round(float(np.percentile(fin, pct)), 2)
        value["p_lo"], value["p_hi"] = p_lo, p_hi
        if p_lo < rng[0] - 1e-6 or p_hi > rng[1]:
            status = "warn"
            details.append(
                f"{name} out of range [{p_lo}, {p_hi}] (p{100 - pct:g}/p{pct:g}) vs {rng}"
            )
    return Check(name, status, value, rng, "; ".join(details))


def staleness(newest_epoch: float, now_epoch: float, warn_min: float) -> Check:
    """Age of the newest issue/frame vs wall clock, in whole minutes.

    Rounds to the nearest minute BEFORE comparing to `warn_min` — this
    matches the original qc_inputs boundary (it computed `round(age_min)`
    once and compared that integer), so a warn does not fire ~30 s earlier
    than it used to just because the comparison moved to unrounded minutes.
    """
    age_min = round((now_epoch - newest_epoch) / 60.0)
    if age_min > warn_min:
        return Check("staleness", "warn", age_min, warn_min,
                      f"newest issue {age_min} min old")
    return Check("staleness", "ok", age_min, warn_min)


# ---------------------------------------------------------------------------
# qc_watchdog: per-region temporal consistency + gauge bias
# ---------------------------------------------------------------------------

def region_metrics(rates, times, bounds, box):
    """Churn/parity/freeze metrics for one region box over a served
    rates cube. Unchanged behaviour from the original qc_watchdog."""
    W, S, E, N = bounds
    h, w = rates.shape[1:]
    c0, c1 = int((box[0] - W) / (E - W) * w), int((box[2] - W) / (E - W) * w)
    r0, r1 = int((N - box[3]) / (N - S) * h), int((N - box[1]) / (N - S) * h)
    sub = np.nan_to_num(rates[:, max(0, r0):r1, max(0, c0):c1])
    if sub.size == 0:
        return None
    wet = sub > 0.3
    sc = [i for i in range(len(times)) if times[i] % 300 == 0]

    flips_s, flips_i, frozen = [], [], 0
    for i in range(1, len(times)):
        u = (wet[i - 1] | wet[i]).sum()
        fl = 100.0 * np.logical_xor(wet[i - 1], wet[i]).sum() / max(u, 1)
        (flips_s if times[i] % 300 == 0 else flips_i).append(fl)
    for a, b in zip(sc[:-1], sc[1:]):
        u = (wet[a] | wet[b]).sum()
        if u > 30 and np.logical_xor(wet[a], wet[b]).sum() / u < 0.02:
            frozen += 1
    area = np.array([float(wet[i].mean()) for i in sc])
    d = np.diff(area)
    parity = 0.0
    if len(d) > 4 and d.std() > 1e-9:
        parity = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    wet_enough = float(np.mean(area)) > 0.001
    return {
        "wet_area_mean_pct": round(100 * float(np.mean(area)), 3),
        "churn_scan_pct": round(float(np.median(flips_s)), 1) if flips_s else None,
        "churn_interp_ratio": (round(float(np.median(flips_i)) / max(float(np.median(flips_s)), 1e-6), 2)
                               if flips_i and flips_s else None),
        "parity_lag1": round(parity, 2),
        "freeze_frac": round(frozen / max(len(sc) - 1, 1), 2),
        "assessable": wet_enough,
    }


def gauge_bias(rates, times, bounds, box, gauge_dir):
    """Served mean rate vs gauge mm over the newest fully-covered clock
    hour. Unchanged behaviour from the original qc_watchdog."""
    hours = sorted(glob.glob(str(pathlib.Path(gauge_dir) / "*.json")))
    if not hours:
        return None
    hour = pathlib.Path(hours[-1]).stem
    h0 = datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=UTC).timestamp()
    idx = [i for i, t in enumerate(times) if h0 < t <= h0 + 3600 and t % 300 == 0]
    if len(idx) < 8:
        return None
    W, S, E, N = bounds
    h, w = rates.shape[1:]
    import json

    rows = json.loads(pathlib.Path(hours[-1]).read_text())
    diffs = []
    for la, lo, mm, _src in rows:
        if not (box[0] <= lo <= box[2] and box[1] <= la <= box[3]) or mm <= 0.25:
            continue
        c = int((lo - W) / (E - W) * w)
        r = int((N - la) / (N - S) * h)
        if not (0 <= r < h and 0 <= c < w):
            continue
        ours_mm = float(np.nansum([rates[i, r, c] / 12.0 for i in idx]))
        diffs.append(ours_mm - mm)
    if len(diffs) < 5:
        return None
    return round(float(np.mean(diffs)), 2)


def evaluate_region(m: dict, gb: float | None, thresholds) -> list[str]:
    """Warning codes for one region's metrics + gauge bias. Unchanged
    behaviour from the original qc_watchdog.main() verdict logic."""
    verdicts = []
    if m["assessable"]:
        if m["churn_scan_pct"] and m["churn_scan_pct"] > thresholds.churn_scan_warn:
            verdicts.append("CHURN")
        if m["churn_interp_ratio"] and m["churn_interp_ratio"] > thresholds.interp_ratio_warn:
            verdicts.append("INTERP")
        if m["parity_lag1"] < thresholds.parity_warn:
            verdicts.append("PARITY-PULSE")
        if m["freeze_frac"] > thresholds.freeze_warn:
            verdicts.append("FREEZE")
    if gb is not None and abs(gb) > thresholds.gauge_bias_warn:
        verdicts.append("GAUGE-BIAS")
    return verdicts


def issue_time_order(issue_time, tail: int = 1000, warn_tail: bool = True) -> "Check":
    """Is ``issue_time`` strictly increasing? The whole array is reported
    (the live store carries one historic out-of-order backfill block); only
    disorder inside the newest ``tail`` issues is a live-health WARN, because
    that is what a broken append would produce."""
    t = np.asarray(issue_time, dtype="int64")
    d = np.diff(t)
    total_bad = int((d <= 0).sum())
    tail_bad = int((d[-tail:] <= 0).sum()) if len(d) else 0
    value = {"n": int(t.size), "non_increasing_steps": total_bad, "non_increasing_in_tail": tail_bad,
             "tail": tail}
    status = "warn" if (warn_tail and tail_bad) else "ok"
    detail = (f"{tail_bad} non-increasing issue_time step(s) in the newest {tail} issues"
              if tail_bad else f"strictly increasing in the newest {tail} issues"
              + (f"; {total_bad} historic step(s) out of order" if total_bad else ""))
    return Check(name="issue_time_order", status=status, value=value, detail=detail)
