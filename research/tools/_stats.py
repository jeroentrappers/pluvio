"""Block-bootstrap confidence intervals for tools/benchmark.py.

Design constraint: the scorer must not retain per-sample pointwise arrays or
FSS field stacks — that grows with sample count x model count x lead count.
Instead every scored sample contributes a fixed-size *sufficient statistic*
record (contingency counts, sum/sum-abs/sum-sq error, FSS numerator/
denominator per threshold/scale), tagged with the sample's issue-time. Every
metric benchmark.py reports (CSI/POD/FAR/FSS/RMSE/MAE/mean_error/CRPS) is a
pure function of a SUM of these records over some set of samples — pooling
arrays and summing sufficient statistics give bit-identical results — so the
block bootstrap resamples issue-time blocks of *records*, not raw fields,
and never re-scores anything.
"""

from __future__ import annotations

import math

import numpy as np

#: metrics carried by every threshold row that the bootstrap reports a CI for
METRIC_KEYS = ("csi", "pod", "far", "freq_bias", "rmse", "mae", "mean_error", "crps")


def issue_block(issue_epoch: int, blocks_h: float) -> int:
    """Which resampling block an issue-time falls in: a floor division into
    fixed ``blocks_h``-hour windows since the Unix epoch, so block boundaries
    are stable across runs/configs rather than depending on the sample set at
    hand."""
    span_s = max(1, round(blocks_h * 3600))
    return int(issue_epoch) // span_s


RELIABILITY_BINS = 10   # forecast-probability bins for the reliability diagram


class SampleStats:
    """Sufficient statistics for one (model, lead)'s scored samples — one
    fixed-size record per sample, no pointwise arrays retained."""

    def __init__(self, thresholds: list[float], fss_scales: list[int]):
        self.thresholds = list(thresholds)
        self.fss_scales = list(fss_scales)
        self._records: list[dict] = []
        self._stacked: dict[str, np.ndarray] | None = None

    def add(self, *, issue_epoch: int, n: int, sum_e: float, sum_abs_e: float,
           sum_sq_e: float, cat: dict[float, tuple[int, int, int]],
           fss: dict[float, dict[int, tuple[float, float]]],
           sum_crps: float | None = None,
           rel: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None) -> None:
        """``sum_crps``: summed per-cell CRPS (quantile models; defaults to the
        deterministic identity CRPS == |error|). ``rel``: per threshold the
        reliability sufficient statistics (bin counts, observed exceedances,
        summed forecast probability) over RELIABILITY_BINS probability bins."""
        if self._stacked is not None:
            raise RuntimeError("SampleStats.add() after aggregate() — stack already built")
        self._records.append({
            "issue_epoch": int(issue_epoch), "n": n, "sum_e": sum_e,
            "sum_abs_e": sum_abs_e, "sum_sq_e": sum_sq_e, "cat": cat, "fss": fss,
            "sum_crps": sum_abs_e if sum_crps is None else float(sum_crps),
            "rel": rel,
        })

    def __len__(self) -> int:
        return len(self._records)

    def issue_epochs(self) -> np.ndarray:
        return np.asarray([r["issue_epoch"] for r in self._records], dtype="int64")

    def _stack(self) -> dict[str, np.ndarray]:
        if self._stacked is not None:
            return self._stacked
        nrec, nthr, nsc = len(self._records), len(self.thresholds), len(self.fss_scales)
        n = np.empty(nrec)
        sum_e = np.empty(nrec)
        sum_abs_e = np.empty(nrec)
        sum_sq_e = np.empty(nrec)
        hits = np.empty((nrec, nthr))
        misses = np.empty((nrec, nthr))
        fa = np.empty((nrec, nthr))
        fss_num = np.empty((nrec, nthr, nsc))
        fss_den = np.empty((nrec, nthr, nsc))
        sum_crps = np.empty(nrec)
        rel_cnt = np.zeros((nrec, nthr, RELIABILITY_BINS))
        rel_obs = np.zeros((nrec, nthr, RELIABILITY_BINS))
        rel_psum = np.zeros((nrec, nthr, RELIABILITY_BINS))
        has_rel = False
        for i, r in enumerate(self._records):
            n[i], sum_e[i], sum_abs_e[i], sum_sq_e[i] = r["n"], r["sum_e"], r["sum_abs_e"], r["sum_sq_e"]
            sum_crps[i] = r.get("sum_crps", r["sum_abs_e"])
            if r.get("rel"):
                has_rel = True
                for j, thr in enumerate(self.thresholds):
                    c, o, ps = r["rel"][thr]
                    rel_cnt[i, j], rel_obs[i, j], rel_psum[i, j] = c, o, ps
            for j, thr in enumerate(self.thresholds):
                h, m, f = r["cat"][thr]
                hits[i, j], misses[i, j], fa[i, j] = h, m, f
                for k, sc in enumerate(self.fss_scales):
                    num, den = r["fss"][thr][sc]
                    fss_num[i, j, k], fss_den[i, j, k] = num, den
        self._stacked = {"n": n, "sum_e": sum_e, "sum_abs_e": sum_abs_e, "sum_sq_e": sum_sq_e,
                        "hits": hits, "misses": misses, "fa": fa,
                        "fss_num": fss_num, "fss_den": fss_den, "sum_crps": sum_crps,
                        "rel_cnt": rel_cnt, "rel_obs": rel_obs, "rel_psum": rel_psum,
                        "has_rel": has_rel}
        return self._stacked

    @staticmethod
    def _reliability(st: dict, idx: np.ndarray, j: int) -> dict | None:
        if not st.get("has_rel"):
            return None
        cnt = st["rel_cnt"][idx, j].sum(axis=0)
        obs = st["rel_obs"][idx, j].sum(axis=0)
        psum = st["rel_psum"][idx, j].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fprob = np.where(cnt > 0, psum / cnt, np.nan)
            ofreq = np.where(cnt > 0, obs / cnt, np.nan)
        edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
        return {"bin_edges": edges.tolist(), "forecast_prob": [None if np.isnan(v) else float(v) for v in fprob],
                "observed_freq": [None if np.isnan(v) else float(v) for v in ofreq],
                "count": [int(v) for v in cnt]}

    def aggregate(self, positions: np.ndarray | None = None) -> dict:
        """Per-threshold metric dict, summed over ``positions`` (repeats
        allowed — a bootstrap draw) or every record when ``positions`` is
        ``None``. Same shape/keys as benchmark.py's ``per_threshold`` rows."""
        st = self._stack()
        idx = np.arange(len(self._records)) if positions is None else np.asarray(positions, dtype="int64")

        n = float(st["n"][idx].sum())
        sum_e = float(st["sum_e"][idx].sum())
        sum_abs_e = float(st["sum_abs_e"][idx].sum())
        sum_sq_e = float(st["sum_sq_e"][idx].sum())
        mean_error = sum_e / n if n else float("nan")
        mae = sum_abs_e / n if n else float("nan")
        rmse = math.sqrt(sum_sq_e / n) if n else float("nan")
        # Deterministic models: CRPS == MAE (the identity model.metrics.
        # crps_deterministic uses). Quantile models (2.2) supply a per-cell
        # CRPS estimate (2 x mean pinball over the predicted levels).
        crps = float(st["sum_crps"][idx].sum()) / n if n else float("nan")
        n_samples = int(idx.shape[0])

        hits = st["hits"][idx].sum(axis=0)
        misses = st["misses"][idx].sum(axis=0)
        fa = st["fa"][idx].sum(axis=0)
        fss_num = st["fss_num"][idx].sum(axis=0)
        fss_den = st["fss_den"][idx].sum(axis=0)

        per_threshold = {}
        for j, thr in enumerate(self.thresholds):
            h, m, f = float(hits[j]), float(misses[j]), float(fa[j])
            pod = h / (h + m) if (h + m) else float("nan")
            far = f / (h + f) if (h + f) else float("nan")
            csi = h / (h + m + f) if (h + m + f) else float("nan")
            bias = (h + f) / (h + m) if (h + m) else float("nan")
            fss = {}
            for k, sc in enumerate(self.fss_scales):
                den = float(fss_den[j, k])
                fss[str(sc)] = (1.0 - float(fss_num[j, k]) / den) if den else float("nan")
            per_threshold[str(thr)] = {
                "csi": csi, "pod": pod, "far": far, "freq_bias": bias,
                "hits": int(h), "misses": int(m), "false_alarms": int(f),
                "mean_error": mean_error, "mae": mae, "rmse": rmse, "crps": crps,
                "reliability": self._reliability(st, idx, j),
                "fss": fss, "n_samples": n_samples, "n_valid_cells": int(n),
            }
        return per_threshold


def _quantile_ci(values: list[float], ci: float) -> dict:
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    # n_finite alongside the bounds: a dry threshold that leaves most
    # bootstrap replicates NaN (e.g. no wet pixels in most resampled blocks)
    # would otherwise report a CI that looks as trustworthy as one built
    # from every replicate.
    if finite.size == 0:
        return {"ci_lo": None, "ci_hi": None, "n_finite": 0}
    alpha = 1.0 - ci
    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    return {"ci_lo": float(np.percentile(finite, lo_q)), "ci_hi": float(np.percentile(finite, hi_q)),
           "n_finite": int(finite.size)}


def block_bootstrap(stats_by_model: dict[str, SampleStats], *, blocks_h: float,
                    n_boot: int, ci: float, seed: int,
                    ref_model: str | None = None) -> dict:
    """Block-bootstrap CIs for every metric/threshold/FSS-scale of every
    model, plus a paired-difference CI vs ``ref_model`` (same block draw used
    for every model in a given replicate, so the difference is genuinely
    paired rather than two independently-resampled intervals).

    Returns ``{"ci": {model: {threshold: {metric: {"ci_lo","ci_hi"}}}},
    "diff_vs_ref": {"ref_model": ..., model: {...same shape...}}}`` — the
    caller merges these back onto the point-estimate report. Deterministic
    for a fixed ``seed``; empty dict if there are no scored samples.
    """
    names = list(stats_by_model)
    if not names or len(stats_by_model[names[0]]) == 0:
        return {}

    epochs = stats_by_model[names[0]].issue_epochs()
    # Every model must have scored exactly the same (issue, lead) samples in
    # the same order — the whole point of a *paired* bootstrap draw is that
    # position i means the same sample for every model. A silent mismatch
    # here (a model dropped a sample, or the lists came from different
    # selections) would corrupt every CI without raising.
    for name in names[1:]:
        other = stats_by_model[name].issue_epochs()
        if other.shape != epochs.shape or not np.array_equal(other, epochs):
            raise ValueError(
                f"block_bootstrap: {name!r} has {other.shape[0]} records with a different "
                f"issue-time sequence than {names[0]!r} ({epochs.shape[0]} records) — "
                "every model must be scored on identical, identically-ordered samples")

    block_ids = np.asarray([issue_block(e, blocks_h) for e in epochs])
    unique_blocks = np.unique(block_ids)
    positions_by_block = [np.nonzero(block_ids == b)[0] for b in unique_blocks]
    n_blocks = len(unique_blocks)

    thresholds = stats_by_model[names[0]].thresholds
    fss_scales = stats_by_model[names[0]].fss_scales
    metric_names = list(METRIC_KEYS) + [f"fss_{sc}" for sc in fss_scales]

    def _empty_reps():
        return {m: {str(thr): {k: [] for k in metric_names} for thr in thresholds} for m in names}

    reps = _empty_reps()
    diff_reps = None
    if ref_model in stats_by_model:
        diff_reps = {m: {str(thr): {k: [] for k in metric_names} for thr in thresholds}
                    for m in names if m != ref_model}

    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.integers(0, n_blocks, size=n_blocks)
        positions = (np.concatenate([positions_by_block[d] for d in draw])
                    if n_blocks else np.empty(0, dtype="int64"))
        agg = {m: stats_by_model[m].aggregate(positions) for m in names}
        for m in names:
            for thr in thresholds:
                row = agg[m][str(thr)]
                for k in METRIC_KEYS:
                    reps[m][str(thr)][k].append(row[k])
                for sc in fss_scales:
                    reps[m][str(thr)][f"fss_{sc}"].append(row["fss"][str(sc)])
        if diff_reps is not None:
            ref_row_by_thr = agg[ref_model]
            for m in diff_reps:
                for thr in thresholds:
                    row, rrow = agg[m][str(thr)], ref_row_by_thr[str(thr)]
                    for k in METRIC_KEYS:
                        diff_reps[m][str(thr)][k].append(row[k] - rrow[k])
                    for sc in fss_scales:
                        diff_reps[m][str(thr)][f"fss_{sc}"].append(row["fss"][str(sc)] - rrow["fss"][str(sc)])

    def _finish(reps_dict):
        return {m: {thr: {k: _quantile_ci(vals, ci) for k, vals in by_metric.items()}
                    for thr, by_metric in by_thr.items()}
                for m, by_thr in reps_dict.items()}

    result = {"ci": _finish(reps)}
    if diff_reps is not None:
        result["diff_vs_ref"] = {"ref_model": ref_model, **_finish(diff_reps)}
    return result
