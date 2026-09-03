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
import pathlib
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.metrics import categorical_scores, continuous_scores, fss_curve  # noqa: E402
from model.zarr_dataset import ZarrCorrectionDataset  # noqa: E402
from tools._advection import advect_forecast  # noqa: E402

LOG = logging.getLogger("pluvio.benchmark")

BASELINE_NAMES = ("persistence", "advection", "operational")


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
    return cfg


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────── sample selection


def _select_samples(dataset: ZarrCorrectionDataset, cfg: dict) -> list[int]:
    """Indices into ``dataset.index`` eligible under the val window / case
    days, deterministically subsampled to ``max_samples``."""
    start = datetime.fromisoformat(cfg["val_window"]["start"]).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(cfg["val_window"]["end"]).replace(tzinfo=timezone.utc)
    case_days: set[date] = set()
    for d in cfg.get("case_days") or []:
        try:
            case_days.add(date.fromisoformat(str(d)))
        except ValueError:
            LOG.warning("skipping unparsable case_day %r", d)

    eligible = []
    for i, s in enumerate(dataset.index):
        dt = datetime.fromtimestamp(s.issue_epoch, tz=timezone.utc)
        if (start <= dt < end) or (dt.date() in case_days):
            eligible.append(i)

    if not eligible:
        raise RuntimeError("no samples fall inside the benchmark's val_window / case_days")

    max_samples = int(cfg["max_samples"])
    if max_samples and len(eligible) > max_samples:
        rng = np.random.default_rng(int(cfg["seed"]))
        chosen = rng.choice(len(eligible), size=max_samples, replace=False)
        chosen.sort()
        eligible = [eligible[i] for i in chosen]
    return eligible


def _pointwise_mask(cfg: dict, h: int, w: int, seed: int):
    n = int(cfg.get("sample_cells", 0) or 0)
    if n <= 0:
        return None
    rng = np.random.default_rng(seed)
    flat = h * w
    pick = rng.choice(flat, size=min(n, flat), replace=False)
    return np.unravel_index(pick, (h, w))


def _select_pw(stack: np.ndarray, mask) -> np.ndarray:
    """stack: (N, H, W) → the pointwise subset (or the full field)."""
    if mask is None:
        return stack
    rs, cs = mask
    return stack[:, rs, cs]


# ─────────────────────────────────────────────────────────────── models


def _load_models(specs: list[str], device):
    """``NAME=path.pt`` specs → {name: (model, callable)}. Only imports torch
    when at least one model is requested."""
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


# ─────────────────────────────────────────────────────────────── scoring


def run_benchmark(zarr_path: str, cfg: dict, model_specs: list[str],
                  device: str = "cpu") -> dict:
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    has_truth = "truth" in set(root.array_keys())
    radar = root["radar"]
    truth = root["truth"] if has_truth else None
    _, h, w = radar.shape[0], radar.shape[-2], radar.shape[-1]

    leads_min = [int(x) for x in cfg["leads_min"]]
    dataset = ZarrCorrectionDataset(zarr_path, leads_min=tuple(leads_min), build_index=True)

    selected = _select_samples(dataset, cfg)
    LOG.info("%d / %d indexed samples selected by the benchmark window", len(selected), len(dataset))

    pw_mask = _pointwise_mask(cfg, h, w, seed=int(cfg["seed"]) + 1)

    torch_device = None
    models = {}
    if model_specs:
        import torch
        torch_device = torch.device(device)
        models = _load_models(model_specs, torch_device)

    model_names = list(BASELINE_NAMES) + list(models.keys())

    # fields[model_name][lead_min] -> (list of pred (H,W), list of obs (H,W))
    fields: dict[str, dict[int, tuple[list, list]]] = {
        m: defaultdict(lambda: ([], [])) for m in model_names
    }

    flow_cache: dict[int, np.ndarray] = {}
    _torch = __import__("torch") if models else None

    for si in selected:
        s = dataset.index[si]
        obs = np.asarray(truth[s.target_idx] if has_truth else radar[s.target_idx, 0],
                         dtype="float32")
        np.nan_to_num(obs, copy=False)

        issue_frame = np.asarray(radar[s.issue_idx, 0], dtype="float32")
        operational = np.asarray(radar[s.issue_idx, s.lead_idx], dtype="float32")

        if s.issue_idx not in flow_cache:
            prev_idx = s.history_idx[-2] if len(s.history_idx) >= 2 else s.history_idx[-1]
            prev_frame = np.asarray(radar[prev_idx, 0], dtype="float32")
            flow_cache[s.issue_idx] = np.stack([prev_frame, issue_frame])
        prev_frame, curr_frame = flow_cache[s.issue_idx]
        advected = advect_forecast(prev_frame, curr_frame, s.lead_min,
                                    dataset.history_step_min)

        preds = {
            "persistence": issue_frame,
            "advection": advected,
            "operational": operational,
        }
        for name, model in models.items():
            x = dataset.build_input(s.issue_idx, s.lead_min, s.history_idx)
            with _torch.no_grad():
                xt = _torch.from_numpy(x).unsqueeze(0).to(torch_device)
                pred = model(xt).squeeze(0).squeeze(0).cpu().numpy().astype("float32")
            preds[name] = pred

        for name, pred in preds.items():
            plist, olist = fields[name][s.lead_min]
            plist.append(pred)
            olist.append(obs)

    thresholds = [float(t) for t in cfg["thresholds_mm_h"]]
    fss_scales = [int(sc) for sc in cfg["fss_scales_px"]]

    results: dict = {}
    n_scored = 0
    for name in model_names:
        results[name] = {}
        for lead, (plist, olist) in fields[name].items():
            if not plist:
                continue
            pred_stack = np.stack(plist)
            obs_stack = np.stack(olist)
            n_scored += pred_stack.shape[0]
            pred_pw = _select_pw(pred_stack, pw_mask)
            obs_pw = _select_pw(obs_stack, pw_mask)
            cont = continuous_scores(pred_pw, obs_pw)

            per_threshold = {}
            for thr in thresholds:
                cat = categorical_scores(pred_pw, obs_pw, thr)
                fss = fss_curve(pred_stack, obs_stack, threshold=thr, scales_px=fss_scales)
                per_threshold[str(thr)] = {
                    "csi": cat["csi"],
                    "pod": cat["pod"],
                    "far": cat["far"],
                    "bias": cont["bias"],
                    "rmse": cont["rmse"],
                    "fss": {str(k): v for k, v in fss.items()},
                    "n": int(pred_stack.shape[0]),
                }
            results[name][str(lead)] = per_threshold

    metadata = {
        "store": str(zarr_path),
        "config_name": cfg.get("name"),
        "config_version": cfg.get("version"),
        "config_hash": cfg["_hash"],
        "n_samples_indexed": len(dataset),
        "n_samples_selected": len(selected),
        "n_scores_total": n_scored,
        "git_commit": _git_commit(),
        "models": list(models.keys()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"metadata": metadata, "results": results}


# ─────────────────────────────────────────────────────────────── reporting


def to_markdown(report: dict) -> str:
    meta = report["metadata"]
    lines = [
        f"# Benchmark: {meta.get('config_name')} (v{meta.get('config_version')})",
        "",
        f"store: `{meta['store']}`  |  config hash: `{meta['config_hash']}`  |  "
        f"samples: {meta['n_samples_selected']} / {meta['n_samples_indexed']}  |  "
        f"commit: `{meta.get('git_commit')}`",
        "",
        "| model | lead (min) | threshold (mm/h) | CSI | POD | FAR | bias | RMSE | FSS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, by_lead in report["results"].items():
        for lead, by_thr in sorted(by_lead.items(), key=lambda kv: int(kv[0])):
            for thr, row in sorted(by_thr.items(), key=lambda kv: float(kv[0])):
                fss_txt = ", ".join(f"{k}px={v:.3f}" if v == v else f"{k}px=nan"
                                    for k, v in row["fss"].items())
                def fmt(v):
                    return f"{v:.3f}" if v == v else "nan"
                lines.append(
                    f"| {model} | {lead} | {thr} | {fmt(row['csi'])} | {fmt(row['pod'])} | "
                    f"{fmt(row['far'])} | {row['bias']:+.3f} | {row['rmse']:.3f} | {fss_txt} |"
                )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────── CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", required=True, help="path to the training store (zarr group)")
    parser.add_argument("--config", default="research/benchmark/benchmark.yaml")
    parser.add_argument("--model", action="append", default=[],
                        help="NAME=path.pt, repeatable")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--markdown", default=None, help="optional output markdown path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
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
