"""Frozen nowcast benchmark: score one or more correction models — plus the
built-in persistence / advection / operational baselines — against a fixed
validation protocol (research/benchmark/benchmark.yaml).

Always scores the three baselines so a model result is never reported in
isolation:

  * ``persistence``  — radar[issue, lead=0] held constant for every lead.
  * ``advection``    — persistence advected by a block-matching flow field
                        estimated from the two most recent history frames,
                        linearly extrapolated to the target lead.
  * ``operational``   — radar[issue, lead_idx], the store's own nowcast.

Runs on CPU with no GPU and no model checkpoints at all (baselines only).
torch / the model package are only imported when at least one ``--model`` is
given.

Every model/baseline is scored on identical support: a per-sample validity
mask (finite obs AND finite prediction from every model/baseline being
compared) is applied before any pointwise metric is computed, so a store with
NaN outside its radar domain (the v3 store, by construction) can't give one
entry a free pass others don't get.

Usage:

    python -m tools.benchmark --zarr /data/timeseries.zarr \\
        --config research/benchmark/benchmark.yaml \\
        --model unet_v3=checkpoints/unet_v3.pt \\
        --out results.json --markdown results.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pathlib
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.metrics import categorical_scores, continuous_scores, crps_deterministic, fss_curve  # noqa: E402
from model.zarr_dataset import ZarrCorrectionDataset, issue_time_split  # noqa: E402
from tools._advection import advect_forecast, flow_for_pair, max_shift_px  # noqa: E402

LOG = logging.getLogger("pluvio.benchmark")

BASELINE_NAMES = ("persistence", "advection", "operational")
DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "benchmark.yaml"
_FLOW_CACHE_MAX = 512  # bounded cache of per-issue flow fields


# ─────────────────────────────────────────────────────────────── config


def load_config(path: str | pathlib.Path) -> dict:
    raw = pathlib.Path(path).read_bytes()
    cfg = yaml.safe_load(raw)
    cfg["_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    cfg.setdefault("thresholds_mm_h", [0.1, 0.5, 1.0, 2.0, 5.0])
    cfg.setdefault("leads_min", [30, 60, 90, 120])
    cfg.setdefault("fss_scales_px", [1, 3, 5])
    cfg.setdefault("max_samples", 2000)
    cfg.setdefault("sample_cells", 0)
    cfg.setdefault("seed", 42)
    cfg.setdefault("case_days", [])
    cfg.setdefault("val_frac_split", 0.2)
    cfg.setdefault("allow_train_overlap", False)

    fss_scales = [int(sc) for sc in cfg["fss_scales_px"]]
    bad = [sc for sc in fss_scales if sc % 2 == 0]
    if bad:
        raise ValueError(
            f"fss_scales_px must be odd (FSS neighbourhoods are centred on each cell): {bad}")
    return cfg


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _sample_set_hash(dataset: ZarrCorrectionDataset, indices: list[int]) -> str:
    """Hash of the exact (issue_epoch, lead_min) pairs scored, sorted for
    determinism — proof two runs scored the same samples."""
    pairs = sorted((int(dataset.index[i].issue_epoch), int(dataset.index[i].lead_min))
                  for i in indices)
    blob = "\n".join(f"{e},{l}" for e, l in pairs).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ─────────────────────────────────────────────────────────── sample selection


def _parse_case_days(cfg: dict) -> set[date]:
    case_days: set[date] = set()
    for d in cfg.get("case_days") or []:
        try:
            case_days.add(date.fromisoformat(str(d)))
        except ValueError:
            LOG.warning("skipping unparsable case_day %r", d)
    return case_days


def _select_samples(dataset: ZarrCorrectionDataset, cfg: dict) -> tuple[list[int], set[int]]:
    """Indices into ``dataset.index`` eligible under the val window / case
    days. Case-day samples are ALWAYS included in full (they're curated, not
    a statistical sample); the remaining budget is spent on a deterministic,
    per-lead-stratified subsample of the plain window. Returns
    (selected_indices, case_day_indices) — the second is a subset of the
    first, used to report case days as their own stratum."""
    start = datetime.fromisoformat(cfg["val_window"]["start"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(cfg["val_window"]["end"])
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    case_days = _parse_case_days(cfg)

    case_idxs: list[int] = []
    window_by_lead: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(dataset.index):
        dt = datetime.fromtimestamp(s.issue_epoch, tz=timezone.utc)
        is_case = dt.date() in case_days
        in_window = start <= dt < end
        if is_case:
            case_idxs.append(i)
        elif in_window:
            window_by_lead[s.lead_min].append(i)

    if not case_idxs and not window_by_lead:
        raise RuntimeError("no samples fall inside the benchmark's val_window / case_days")

    max_samples = int(cfg["max_samples"])
    budget = max(0, max_samples - len(case_idxs)) if max_samples else None
    seed = int(cfg["seed"])

    window_idxs: list[int] = []
    leads = sorted(window_by_lead)
    if leads:
        per_lead = None if budget is None else max(1, budget // len(leads)) if budget else 0
        for lead in leads:
            pool = window_by_lead[lead]
            if per_lead is not None and len(pool) > per_lead:
                rng = np.random.default_rng(seed + lead)  # stable per lead, independent of others
                chosen = rng.choice(len(pool), size=per_lead, replace=False)
                chosen.sort()
                pool = [pool[i] for i in chosen]
            window_idxs.extend(pool)

    selected = sorted(set(case_idxs) | set(window_idxs))
    return selected, set(case_idxs)


def _pointwise_cell_mask(cfg: dict, h: int, w: int, seed_base: int, issue_idx: int) -> np.ndarray | None:
    """A per-issue random cell mask for the pointwise metrics (``None`` = use
    every cell). Reseeded per issue so a fixed-cells artefact never sits at
    the same physical location for every sample."""
    n = int(cfg.get("sample_cells", 0) or 0)
    if n <= 0:
        return None
    rng = np.random.default_rng(seed_base + issue_idx)
    flat = h * w
    pick = rng.choice(flat, size=min(n, flat), replace=False)
    mask = np.zeros(flat, dtype=bool)
    mask[pick] = True
    return mask.reshape(h, w)


# ─────────────────────────────────────────────────────────────── models


def _load_models(specs: list[str], device):
    """``NAME=path.pt`` specs → {name: model}. Only imports torch when at
    least one model is requested."""
    if not specs:
        return {}
    import torch

    from model.unet import PluvioUNet

    models = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--model expects NAME=path.pt, got {spec!r}")
        name, ckpt_path = spec.split("=", 1)
        ckpt = torch.load(ckpt_path, map_location=device)
        in_channels = ckpt.get("in_channels", 29)
        base_channels = ckpt.get("base_channels", 32)
        model = PluvioUNet(in_channels=in_channels, base_channels=base_channels).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        LOG.info("loaded model %r from %s (val_rmse=%.4f, epoch=%s)",
                 name, ckpt_path, ckpt.get("val_rmse", float("nan")), ckpt.get("epoch"))
        models[name] = model
    return models


# ───────────────────────────────────────────────────────── grid / geometry


def _km_per_px(root) -> float:
    """Approximate grid spacing from the store's ``bounds``/``grid_n``
    attrs (v3+ stores); falls back to the legacy ~6 km KNMI-stereo grid
    spacing when those attrs aren't present."""
    attrs = dict(root.attrs)
    grid_n = attrs.get("grid_n")
    bounds = attrs.get("bounds")
    if not grid_n or not bounds or len(bounds) != 4:
        return 6.0
    west, south, east, north = (float(x) for x in bounds)
    lat_mid = (south + north) / 2.0
    lat_km = (north - south) * 111.0
    lon_km = (east - west) * 111.0 * math.cos(math.radians(lat_mid))
    return float(((lat_km / grid_n) + (lon_km / grid_n)) / 2.0)


# ─────────────────────────────────────────────────────────────── scoring


def run_benchmark(zarr_path: str, cfg: dict, model_specs: list[str],
                  device: str = "cpu") -> dict:
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    has_truth = "truth" in set(root.array_keys())
    radar = root["radar"]
    truth = root["truth"] if has_truth else None
    h, w = radar.shape[-2], radar.shape[-1]

    # Refuse to silently score against training data: the val window must
    # start at or after the store's own train/val split.
    val_frac = float(cfg["val_frac_split"])
    split_dt = issue_time_split(zarr_path, val_frac)
    win_start = datetime.fromisoformat(cfg["val_window"]["start"])
    if win_start.tzinfo is None:
        win_start = win_start.replace(tzinfo=timezone.utc)
    if win_start < split_dt and not cfg.get("allow_train_overlap"):
        raise RuntimeError(
            f"val_window.start ({win_start.isoformat()}) precedes the store's own "
            f"train/val split ({split_dt.isoformat()}, val_frac={val_frac}) — this window "
            f"would score against training data. Move val_window.start to "
            f"{split_dt.isoformat()} or later, or set allow_train_overlap: true to override "
            f"deliberately (e.g. for a store with no temporal train/val split at all)."
        )

    leads_min = [int(x) for x in cfg["leads_min"]]
    dataset = ZarrCorrectionDataset(zarr_path, leads_min=tuple(leads_min), build_index=True)

    selected, case_idx_set = _select_samples(dataset, cfg)
    LOG.info("%d / %d indexed samples selected (%d curated case-day, %d window)",
             len(selected), len(dataset), len(case_idx_set), len(selected) - len(case_idx_set))

    km_per_px = _km_per_px(root)
    max_shift = max_shift_px(km_per_px, dataset.history_step_min)
    LOG.info("grid spacing ~%.2f km/px -> advection search radius %d px", km_per_px, max_shift)

    torch_device = None
    torch_mod = None
    models = {}
    if model_specs:
        import torch as torch_mod
        torch_device = torch_mod.device(device)
        models = _load_models(model_specs, torch_device)
    model_names = list(BASELINE_NAMES) + list(models.keys())

    thresholds = [float(t) for t in cfg["thresholds_mm_h"]]
    fss_scales = [int(sc) for sc in cfg["fss_scales_px"]]
    fss_fill = min(thresholds) - 1.0  # sentinel guaranteed below every threshold

    # Per (model, lead): concatenated valid pointwise pred/obs (1-D), and
    # stacked (N, H, W) fields for FSS. Kept separately for "all selected"
    # and the case-day-only stratum.
    def _new_accum():
        return {m: defaultdict(lambda: {"pred_pw": [], "obs_pw": [],
                                        "pred_fss": [], "obs_fss": []})
                for m in model_names}

    accum_all = _new_accum()
    accum_case = _new_accum()
    n_valid_by_lead_all: dict[int, int] = defaultdict(int)
    n_valid_by_lead_case: dict[int, int] = defaultdict(int)
    n_scored_by_lead: dict[int, int] = defaultdict(int)

    flow_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
    pw_seed = int(cfg["seed"]) + 1

    for si in selected:
        s = dataset.index[si]
        obs_raw = np.asarray(truth[s.target_idx] if has_truth else radar[s.target_idx, 0],
                             dtype="float32")

        issue_raw = np.asarray(radar[s.issue_idx, 0], dtype="float32")
        operational_raw = np.asarray(radar[s.issue_idx, s.lead_idx], dtype="float32")
        prev_idx = s.history_idx[-2] if len(s.history_idx) >= 2 else s.history_idx[-1]
        prev_raw = np.asarray(radar[prev_idx, 0], dtype="float32")

        if s.issue_idx not in flow_cache:
            issue_filled = np.nan_to_num(issue_raw)
            prev_filled = np.nan_to_num(prev_raw)
            flow_cache[s.issue_idx] = flow_for_pair(prev_filled, issue_filled, max_shift=max_shift)
            flow_cache.move_to_end(s.issue_idx)
            while len(flow_cache) > _FLOW_CACHE_MAX:
                flow_cache.popitem(last=False)
        else:
            flow_cache.move_to_end(s.issue_idx)
        advected = advect_forecast(prev_raw, np.nan_to_num(issue_raw), s.lead_min,
                                    dataset.history_step_min, max_shift=max_shift,
                                    flow=flow_cache[s.issue_idx])

        preds = {
            "persistence": issue_raw,
            "advection": advected,
            "operational": operational_raw,
        }
        if models:
            x = dataset.build_input(s.issue_idx, s.lead_min, s.history_idx)
            for name, model in models.items():
                with torch_mod.no_grad():
                    xt = torch_mod.from_numpy(x).unsqueeze(0).to(torch_device)
                    pred = model(xt).squeeze(0).squeeze(0).cpu().numpy().astype("float32")
                preds[name] = pred

        # One validity mask per sample, shared by every model/baseline so
        # every entry is scored on identical support.
        valid = np.isfinite(obs_raw)
        for pred in preds.values():
            valid &= np.isfinite(pred)

        cell_mask = _pointwise_cell_mask(cfg, h, w, pw_seed, s.issue_idx)
        selector = valid if cell_mask is None else (valid & cell_mask)

        obs_fss = np.where(valid, np.nan_to_num(obs_raw), fss_fill)
        n_valid_by_lead_all[s.lead_min] += int(valid.sum())
        n_scored_by_lead[s.lead_min] += 1
        is_case = si in case_idx_set
        if is_case:
            n_valid_by_lead_case[s.lead_min] += int(valid.sum())

        for name, pred in preds.items():
            pred_fss = np.where(valid, np.nan_to_num(pred), fss_fill)
            for accum in ((accum_all, True), (accum_case, is_case)):
                store, keep = accum
                if not keep:
                    continue
                bucket = store[name][s.lead_min]
                bucket["pred_pw"].append(pred[selector])
                bucket["obs_pw"].append(obs_raw[selector])
                bucket["pred_fss"].append(pred_fss)
                bucket["obs_fss"].append(obs_fss)

    def _score(accum, n_valid_by_lead) -> dict:
        results: dict = {}
        for name in model_names:
            results[name] = {}
            for lead, bucket in accum[name].items():
                if not bucket["pred_pw"]:
                    continue
                pred_pw = np.concatenate([a.ravel() for a in bucket["pred_pw"]])
                obs_pw = np.concatenate([a.ravel() for a in bucket["obs_pw"]])
                pred_stack = np.stack(bucket["pred_fss"])
                obs_stack = np.stack(bucket["obs_fss"])
                n_samples = len(bucket["pred_fss"])

                cont = continuous_scores(pred_pw, obs_pw)
                crps = crps_deterministic(pred_pw, obs_pw)

                per_threshold = {}
                for thr in thresholds:
                    cat = categorical_scores(pred_pw, obs_pw, thr)
                    fss = fss_curve(pred_stack, obs_stack, threshold=thr, scales_px=fss_scales)
                    per_threshold[str(thr)] = {
                        "csi": cat["csi"],
                        "pod": cat["pod"],
                        "far": cat["far"],
                        "freq_bias": cat["freq_bias"],
                        "hits": cat["hits"],
                        "misses": cat["misses"],
                        "false_alarms": cat["false_alarms"],
                        "mean_error": cont["bias"],
                        "mae": cont["mae"],
                        "rmse": cont["rmse"],
                        "crps": crps,
                        "reliability": None,  # Brier-decomposition slot, wired once a
                                              # probabilistic (quantile) model is scored
                        "fss": {str(k): v for k, v in fss.items()},
                        "n_samples": n_samples,
                        "n_valid_cells": n_valid_by_lead.get(lead, 0),
                    }
                results[name][str(lead)] = per_threshold
        return results

    results = _score(accum_all, n_valid_by_lead_all)
    results_case_days = _score(accum_case, n_valid_by_lead_case) if case_idx_set else {}

    n_scored_total = sum(n_scored_by_lead.values())  # counted once, not per model

    metadata = {
        "store": str(zarr_path),
        "config_name": cfg.get("name"),
        "config_version": cfg.get("version"),
        "config_hash": cfg["_hash"],
        "n_samples_indexed": len(dataset),
        "n_samples_selected": len(selected),
        "n_case_day_samples": len(case_idx_set),
        "n_scores_total": n_scored_total,
        "git_commit": _git_commit(),
        "models": list(models.keys()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_set_hash": _sample_set_hash(dataset, selected),
        "thresholds_mm_h": thresholds,
        "leads_min": leads_min,
        "fss_scales_px": fss_scales,
        "fss_boundary": "constant zero padding (scipy.ndimage.uniform_filter, cval=0.0)",
        "seed": int(cfg["seed"]),
        "max_samples": int(cfg["max_samples"]),
        "sample_cells": int(cfg.get("sample_cells", 0) or 0),
        "km_per_px": km_per_px,
        "advection_max_shift_px": max_shift,
        "train_val_split": split_dt.isoformat(),
    }
    return {"metadata": metadata, "results": results, "results_case_days": results_case_days}


# ─────────────────────────────────────────────────────────────── reporting


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and v == v else ("nan" if v is None else str(v))


def _markdown_table(results: dict) -> list[str]:
    lines = [
        "| model | lead (min) | threshold (mm/h) | CSI | POD | FAR | mean_error | RMSE | FSS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, by_lead in results.items():
        for lead, by_thr in sorted(by_lead.items(), key=lambda kv: int(kv[0])):
            for thr, row in sorted(by_thr.items(), key=lambda kv: float(kv[0])):
                fss_txt = ", ".join(f"{k}px={_fmt(v)}" for k, v in row["fss"].items())
                lines.append(
                    f"| {model} | {lead} | {thr} | {_fmt(row['csi'])} | {_fmt(row['pod'])} | "
                    f"{_fmt(row['far'])} | {_fmt(row['mean_error'])} | {_fmt(row['rmse'])} | {fss_txt} |"
                )
    return lines


def to_markdown(report: dict) -> str:
    meta = report["metadata"]
    lines = [
        f"# Benchmark: {meta.get('config_name')} (v{meta.get('config_version')})",
        "",
        f"store: `{meta['store']}`  |  config hash: `{meta['config_hash']}`  |  "
        f"sample set: `{meta.get('sample_set_hash')}`  |  "
        f"samples: {meta['n_samples_selected']} / {meta['n_samples_indexed']} "
        f"({meta.get('n_case_day_samples', 0)} case-day)  |  "
        f"commit: `{meta.get('git_commit')}`",
        "",
        *_markdown_table(report["results"]),
    ]
    if report.get("results_case_days"):
        lines += ["", "## Case-day stratum", "", *_markdown_table(report["results_case_days"])]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────── CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", required=True, help="path to the training store (zarr group)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", action="append", default=[],
                        help="NAME=path.pt, repeatable")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--markdown", default=None, help="optional output markdown path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-train-overlap", action="store_true",
                        help="override the val_window/train-split guard")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.allow_train_overlap:
        cfg["allow_train_overlap"] = True
    report = run_benchmark(args.zarr, cfg, args.model, device=args.device)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_nan_to_null(report), indent=2, allow_nan=False))
    LOG.info("wrote %s", out_path)

    if args.markdown:
        md_path = pathlib.Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(to_markdown(report))
        LOG.info("wrote %s", md_path)
    return 0


def _nan_to_null(obj):
    # NaN metrics (e.g. FSS with no wet pixels at a threshold, or a
    # zero-denominator CSI) are real, informative results we don't want to
    # drop — but bare NaN isn't valid JSON, so serialise it as null.
    if isinstance(obj, float):
        return None if obj != obj else obj  # obj != obj  ⇔  isnan(obj)
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


if __name__ == "__main__":
    sys.exit(main())
