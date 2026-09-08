"""Training loop for the Pluvio correction UNet.

Designed to be run as:

    python -m model.train --data ../data --epochs 20 --batch-size 16

Key design choices:
- **Weighted Huber loss** with sample weights ∝ (1 + obs_mm_per_h)² so the
  optimizer cares disproportionately about heavy-rain cells. Otherwise the
  95%-dry data distribution drives the model to "always predict 0".
- **Auxiliary BCE head** would normally be added here — keeping it out of
  the v0 loop until we've confirmed the regression head trains stably.
- Mixed precision (`torch.amp`) for ~1.7× speedup on consumer GPUs.
- Early stopping on validation RMSE plateau.

The radar-only dataset (v1) trains on CPU. Use ``--max-minutes`` to cap the
wall-clock budget on a laptop. ONNX export is a follow-up.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
import signal
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.dataset import PluvioCorrectionDataset  # noqa: E402
from model.losses import CombinedLoss  # noqa: E402
from model.metrics import fss_components  # noqa: E402
from model.shard_dataset import (  # noqa: E402
    ShardDataset,
    ShardRecipeMismatch,
    compare_recipes,
    recipe_from_dataset,
    source_store_hash,
)
from model.zarr_dataset import ZarrCorrectionDataset, issue_time_split  # noqa: E402
from model.unet import PluvioUNet, num_params  # noqa: E402

LOG = logging.getLogger("pluvio.train")

_DT_MIN = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DT_MAX = datetime(2100, 1, 1, tzinfo=timezone.utc)


def _time_split(data_root: pathlib.Path, val_frac: float) -> datetime:
    """Pick the issue-time boundary so the most-recent ``val_frac`` of the
    forecast window becomes validation. Splitting by time (not random)
    prevents leakage between near-identical adjacent frames."""
    fc_dir = data_root / "radar_forecast" / "2.0"
    stamps = sorted(
        datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        for p in fc_dir.glob("RAD_NL25_RAC_FM_*.h5")
        if (m := re.search(r"(\d{12})", p.name))
    )
    if not stamps:
        raise FileNotFoundError(f"no forecast files under {fc_dir}")
    cut = int(len(stamps) * (1.0 - val_frac))
    cut = min(max(cut, 1), len(stamps) - 1)
    return stamps[cut]


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_fn: CombinedLoss,
    should_stop=None,
) -> tuple[float, dict[str, float], bool]:
    """Runs one epoch. Returns the mean batch loss, the mean of each loss
    component (``loss_fn.last_terms``) over the epoch — components with zero
    weight are simply absent from every batch's ``last_terms`` dict — and
    whether it stopped early because ``should_stop()`` went true (a pause
    signal), in which case the epoch is INCOMPLETE and its metrics are not
    comparable to a full one."""
    model.train()
    stopped = False
    losses: list[float] = []
    term_sums: dict[str, float] = {}
    term_counts: dict[str, int] = {}
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for k, v in loss_fn.last_terms.items():
            term_sums[k] = term_sums.get(k, 0.0) + v
            term_counts[k] = term_counts.get(k, 0) + 1
        if should_stop is not None and should_stop():
            stopped = True
            break
    mean_terms = {k: term_sums[k] / term_counts[k] for k in term_sums}
    return float(sum(losses) / max(len(losses), 1)), mean_terms, stopped


VAL_FSS_THRESHOLD_MM_H = 0.1   # mm/h — radar and targets are in mm/h (zarr_dataset)
VAL_FSS_SCALE_PX = 3


def _state_path(args) -> pathlib.Path:
    if args.state:
        return pathlib.Path(args.state)
    ck = pathlib.Path(args.checkpoint)
    return ck.with_name(f"{ck.stem}.state{ck.suffix}")


def _resume_path(args) -> pathlib.Path | None:
    """Where to resume from, or None to start fresh."""
    mode = (args.resume or "auto").strip()
    if mode == "never":
        return None
    path = _state_path(args) if mode == "auto" else pathlib.Path(mode)
    return path if path.exists() else None


def _companion_metrics(args, select_on: str) -> list[str]:
    """Alternative selection metrics to keep a checkpoint for.

    ``auto``: for a shaped loss, the skill metric the product actually cares
    about (val_fss3) plus RMSE, minus whichever one already drives selection.
    """
    raw = ("auto" if args.save_best_of is None else args.save_best_of).strip()
    if raw != "auto":
        return [m for m in (x.strip() for x in raw.split(",")) if m and m != select_on]
    shaped = select_on != "val_rmse"
    return [m for m in (["val_fss3", "val_rmse"] if shaped else []) if m != select_on]


def resolve_select_on(args) -> str:
    """Which validation metric picks the checkpoint.

    ``auto`` means "the objective": any loss that is not plain RMSE — FSS,
    sharpness, or a quantile head — trades RMSE for something else on purpose,
    so RMSE stops the run at epoch 1 and discards everything after it.
    """
    if args.select_on != "auto":
        return args.select_on
    shaped = (float(getattr(args, "fss_weight", 0) or 0) > 0
              or float(getattr(args, "sharpness_weight", 0) or 0) > 0
              or bool(getattr(args, "quantiles", None)))
    return "val_loss" if shaped else "val_rmse"


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, median_index: int = 0,
    loss_fn: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Validation metrics of the deterministic output — the median channel for
    a quantile head (2.2), the only channel otherwise.

    ``val_rmse`` alone is the wrong model-selection signal for any loss that
    deliberately trades RMSE for structure: under FSS+sharpness (2.1b) both v3
    arms had their best val RMSE at epoch 1 and then spent 30 epochs "getting
    worse" while the objective and the wet-cell skill kept improving. So this
    also reports the validation value of the objective itself, an FSS at the
    lowest scored threshold, and the wet-area frequency bias that the sharpness
    term inflates.
    """
    model.eval()
    rmses: list[float] = []
    losses: list[float] = []
    fss_num = fss_den = 0.0
    pred_wet = obs_wet = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw = model(x)
        if loss_fn is not None:
            losses.append(float(loss_fn(raw, y).detach().cpu()))
        pred = raw[:, median_index : median_index + 1] if raw.shape[1] > 1 else raw
        rmses.append(float(rmse(pred, y).cpu()))
        p_np = pred[:, 0].float().cpu().numpy()
        o_np = y[:, 0].float().cpu().numpy()
        for pf, of in zip(p_np, o_np, strict=True):
            num, den = fss_components(pf, of, threshold=VAL_FSS_THRESHOLD_MM_H,
                                      scale_px=VAL_FSS_SCALE_PX)
            fss_num += num
            fss_den += den
        pred_wet += float((p_np >= VAL_FSS_THRESHOLD_MM_H).mean())
        obs_wet += float((o_np >= VAL_FSS_THRESHOLD_MM_H).mean())
    n = max(len(rmses), 1)
    out = {"val_rmse": sum(rmses) / n,
           "val_fss3": 1.0 - fss_num / fss_den if fss_den > 0 else float("nan"),
           "val_wet_bias": (pred_wet / obs_wet) if obs_wet > 0 else float("nan")}
    if losses:
        out["val_loss"] = sum(losses) / len(losses)
    return out


def _load_shard_sets(args) -> tuple[ShardDataset, ShardDataset]:
    """Open the train/val halves of a pre-rendered shard store (WP 2.6).

    The point of the shard path is that the loss curve is unchanged, so every
    way the shards could silently *not* be the samples this invocation asked
    for is checked here rather than discovered as a mystery divergence:

    * train and val must carry the same recipe apart from the ``split`` label —
      in particular the same ``split_boundary_epoch``, so the two halves come
      from one chronological split with no overlap;
    * ``--require-rain-fraction`` must match what the train shards were
      rendered with (the filter is baked into the rendered sample set);
    * a bumped ``zarr_dataset.NORMALISE_VERSION`` is refused by
      ``ShardDataset`` itself, with or without ``--zarr`` — the version is a
      property of this build, not of the store;
    * with ``--zarr`` also given, the recipe is re-derived from the store and
      compared field by field — that catches a changed lead set, an added aux
      channel, or a ``--lagrangian-channels`` count the shards were not
      rendered with — and the store's own
      fingerprint (``source_store_hash``: structure plus sampled content) is
      compared against the one the render recorded, which is the only thing
      that catches a store REBUILT IN PLACE with the same shapes, attrs and
      ``issue_time`` but different values.

    Without ``--zarr`` there is no store to re-derive from, so the shards'
    own recipe is authoritative: ``--lagrangian-channels`` is then checked
    against what they were rendered with rather than silently ignored.
    """
    root = pathlib.Path(args.shards)
    train_dir, val_dir = root / "train", root / "val"
    missing = [str(d) for d in (train_dir, val_dir) if not (d / "manifest.json").exists()]
    if missing:
        raise SystemExit(
            f"--shards {root}: expected rendered train and val splits, missing "
            f"{', '.join(missing)} — run "
            f"`python -m tools.render_shards --zarr <store> --out {root} --split train,val`"
        )

    expected_train = expected_val = expected_source = None
    if args.zarr:
        boundary = issue_time_split(args.zarr, args.val_frac)
        probe = ZarrCorrectionDataset(
            args.zarr, build_index=False,
            require_rain_fraction=args.require_rain_fraction,
            # so `--shards --lagrangian-channels 2` against shards rendered
            # without the planes is a named recipe mismatch, not a run that
            # silently trains on the wrong channel set (2.3/2.6).
            lagrangian_channels=args.lagrangian_channels,
        )
        common = {
            "split_boundary_epoch": int(boundary.timestamp()),
            "val_frac": args.val_frac,
        }
        expected_train = recipe_from_dataset(probe, split="train", dtype=None, **common)
        expected_val = recipe_from_dataset(probe, split="val", dtype=None, **common)
        expected_val["require_rain_fraction"] = None
        expected_source = source_store_hash(args.zarr)

    # dtype and max_samples are render-time choices, not sample semantics we can
    # derive from the store — exclude them from the store-derived comparison.
    ignore = frozenset({"dtype", "max_samples"})
    train_set = ShardDataset(train_dir, expected_recipe=expected_train,
                             expected_source_store=expected_source, ignore_recipe_keys=ignore)
    val_set = ShardDataset(val_dir, expected_recipe=expected_val,
                           expected_source_store=expected_source, ignore_recipe_keys=ignore)

    diffs = compare_recipes(train_set.recipe, val_set.recipe, ignore=frozenset({"split"}))
    diffs = [d for d in diffs if not d.startswith("require_rain_fraction")]
    if diffs:
        raise ShardRecipeMismatch(
            f"--shards {root}: train and val shards disagree ({len(diffs)} field(s)):\n  "
            + "\n  ".join(diffs) + "\nRe-render both splits from the same store."
        )
    rendered_lagrangian = train_set.recipe.get("lagrangian_channels", 0)
    if int(rendered_lagrangian or 0) != int(args.lagrangian_channels):
        raise SystemExit(
            f"--shards {root}: train shards were rendered with "
            f"--lagrangian-channels {rendered_lagrangian!r} but this run asks for "
            f"{args.lagrangian_channels!r}. The planes are baked into the rendered "
            "input; re-render the shards with the count you want, or drop the flag."
        )
    rendered_filter = train_set.recipe.get("require_rain_fraction")
    if rendered_filter != args.require_rain_fraction:
        raise SystemExit(
            f"--shards {root}: train shards were rendered with "
            f"--require-rain-fraction {rendered_filter!r} but this run asks for "
            f"{args.require_rain_fraction!r}. The filter is baked into the rendered sample "
            "set; re-render, or drop the flag to train on the shards as rendered."
        )
    if val_set.recipe.get("require_rain_fraction") is not None:
        raise ShardRecipeMismatch(
            f"--shards {root}: val shards carry a require_rain_fraction filter "
            f"({val_set.recipe['require_rain_fraction']!r}); validation must never be filtered "
            "or the per-epoch metric is not comparable to a zarr run."
        )
    LOG.info("Shards: %s (%d train / %d val, %d channels, dtype=%s, layout=%s, boundary=%s)",
             root, len(train_set), len(val_set), train_set.n_channels, train_set.dtype,
             train_set.layout, train_set.recipe.get("split_boundary_epoch"))
    return train_set, val_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=pathlib.Path, default=None,
                        help="Path to timeseries.zarr. If set, trains on the full "
                             "multi-source store (ZarrCorrectionDataset) instead of "
                             "the legacy radar-HDF5 dataset under --data.")
    parser.add_argument("--shards", type=pathlib.Path, default=None,
                        help="Pre-rendered shard root from tools/render_shards.py (expects "
                             "<dir>/train and <dir>/val). Swaps ZarrCorrectionDataset for "
                             "ShardDataset: identical samples/targets/order, no per-sample "
                             "assembly. Pass --zarr as well to re-derive the recipe from the "
                             "store and refuse mismatched shards.")
    parser.add_argument("--data", type=pathlib.Path, required=False,
                        help="KNMI data root (holds radar_forecast/ and nl_rdr_data_rtcor_5m/).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/pluvio_unet.pt")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of the time window (most recent) held out for validation.")
    parser.add_argument("--require-rain-fraction", type=float, default=None,
                        help="Drop training samples whose target wet-cell fraction is below this.")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="UNet width. 16 is ~4x faster on CPU than 32.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Randomly subsample the training index to this many per run (CPU speed).")
    parser.add_argument("--max-val-samples", type=int, default=2500,
                        help="Subsample validation for the per-epoch metric (CPU speed).")
    parser.add_argument("--bias-penalty", type=float, default=0.5,
                        help="Weight on the mean-bias penalty term (fixes wet over-prediction).")
    parser.add_argument("--fss-weight", type=float, default=0.0,
                        help="Weight on the differentiable exceedance-FSS structure loss "
                             "(0 = disabled, matching the pre-existing Huber-only objective).")
    parser.add_argument("--fss-thresholds", default="0.5,1.0,2.0",
                        help="Comma-separated rain-rate thresholds (mm/h) for --fss-weight.")
    parser.add_argument("--fss-scales", default="1,3,5",
                        help="Comma-separated neighbourhood pooling scales (px) for --fss-weight.")
    parser.add_argument("--fss-tau", type=float, default=0.05,
                        help="Softness (mm/h) of the sigmoid exceedance indicator for --fss-weight.")
    parser.add_argument("--sharpness-weight", type=float, default=0.0,
                        help="Weight on the gradient-energy sharpness loss "
                             "(0 = disabled, matching the pre-existing Huber-only objective).")
    parser.add_argument("--quantiles", default=None,
                        help="Comma-separated quantile levels for a probabilistic head (2.2), e.g. "
                             "0.1,0.5,0.9; must include 0.5. Default: single deterministic output.")
    parser.add_argument("--quantile-weight", type=float, default=1.0,
                        help="Weight on the pinball + crossing terms when --quantiles is set.")
    parser.add_argument("--lagrangian-channels", type=int, default=0,
                        choices=(0, 1, 2),
                        help="Append Lagrangian-persistence input channels (2.3): 1 = the "
                             "latest radar analysis advected to the sample's lead by the "
                             "flow between the two newest history frames, 2 = that plus the "
                             "per-step flow magnitude. 0 (default) keeps the input identical "
                             "to every existing checkpoint. Recorded in the checkpoint's "
                             "channel recipe so infer_latest rebuilds the same input.")
    parser.add_argument("--patience", type=int, default=30,
                        help="early-stopping patience in epochs (plateau of --select-on)")
    parser.add_argument("--state", default=None,
                        help="path of the resumable training state (model + optimizer + "
                             "scheduler + epoch counters), written every epoch. "
                             "Default: <checkpoint stem>.state.pt")
    parser.add_argument("--resume", default="auto",
                        help="'auto' (default) resumes from --state when that file exists, "
                             "'never' always starts fresh, or a path to resume from. A run "
                             "can therefore be stopped (SIGTERM/SIGINT, or a reboot) and "
                             "picked up where it left off")
    parser.add_argument("--save-best-of", default="auto",
                        help="comma-separated extra metrics to keep a companion "
                             "checkpoint for (val_loss,val_rmse,val_fss3), 'auto' for "
                             "the sensible set, '' for none. The frozen benchmark then "
                             "decides which selection metric ships")
    parser.add_argument("--select-on", default="auto",
                        choices=("auto", "val_loss", "val_rmse", "val_fss3"),
                        help="metric that picks the checkpoint and drives early stopping. "
                             "auto = the objective (val_loss) whenever the loss is not plain "
                             "RMSE, else val_rmse — selecting on RMSE under an FSS+sharpness "
                             "loss keeps the epoch-1 model and throws the run away")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop training after this many wall-clock minutes (CPU budget guard).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.fss_weight < 0:
        raise SystemExit(f"--fss-weight must be >= 0, got {args.fss_weight}")
    if args.sharpness_weight < 0:
        raise SystemExit(f"--sharpness-weight must be >= 0, got {args.sharpness_weight}")
    if args.sharpness_weight > 0.3:
        raise SystemExit(
            f"--sharpness-weight must be <= 0.3, got {args.sharpness_weight}. The review "
            "measured that the sharpness hinge locally rewards noise injection up to parity "
            "with the target's gradient energy, and that the noise-vs-structure crossover "
            "moves past this weight above ~0.3 (see model/losses.py:sharpness_loss docstring, "
            "'Residual gaming risk'). Re-measure the crossover before raising this cap."
        )
    if args.fss_tau <= 0:
        raise SystemExit(f"--fss-tau must be > 0, got {args.fss_tau}")
    fss_thresholds = tuple(float(v) for v in args.fss_thresholds.split(",") if v.strip())
    fss_scales = tuple(int(v) for v in args.fss_scales.split(",") if v.strip())
    if not fss_thresholds:
        raise SystemExit(f"--fss-thresholds must list at least one threshold, got {args.fss_thresholds!r}")
    if not fss_scales or any(s < 1 for s in fss_scales):
        raise SystemExit(f"--fss-scales must list integers >= 1, got {args.fss_scales!r}")

    device = torch.device(args.device)
    LOG.info("Training on %s", device)

    if not args.zarr and not args.data and not args.shards:
        raise SystemExit("provide --shards <dir> (pre-rendered), --zarr <timeseries.zarr> "
                         "(multi-source) or --data <radar dir> (legacy)")

    if args.shards:
        train_set, val_set = _load_shard_sets(args)
    elif args.zarr:
        split = issue_time_split(args.zarr, args.val_frac)
        LOG.info("Time split (zarr): train < %s ≤ val", split.isoformat())
        train_set = ZarrCorrectionDataset(
            args.zarr, time_range=(_DT_MIN, split),
            require_rain_fraction=args.require_rain_fraction,
            lagrangian_channels=args.lagrangian_channels,
        )
        val_set = ZarrCorrectionDataset(args.zarr, time_range=(split, _DT_MAX),
                                        lagrangian_channels=args.lagrangian_channels)
    else:
        if args.lagrangian_channels:
            raise SystemExit(
                "--lagrangian-channels needs the zarr store (--zarr); the legacy "
                "radar-HDF5 dataset does not assemble channels through "
                "ZarrCorrectionDataset.build_input")
        split = _time_split(args.data, args.val_frac)
        LOG.info("Time split: train < %s ≤ val", split.isoformat())
        train_set = PluvioCorrectionDataset(
            args.data,
            time_range=(_DT_MIN, split),
            require_rain_fraction=args.require_rain_fraction,
        )
        val_set = PluvioCorrectionDataset(args.data, time_range=(split, _DT_MAX))

    import torch.utils.data as tud

    def _subsample(ds, n, seed):
        if n is None or len(ds) <= n:
            return ds
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(ds), generator=g)[:n].tolist()
        return tud.Subset(ds, pick)

    train_for_loader = _subsample(train_set, args.max_train_samples, 0)
    val_for_loader = _subsample(val_set, args.max_val_samples, 1)
    LOG.info("Train: %d (using %d) | Val: %d (using %d)",
             len(train_set), len(train_for_loader), len(val_set), len(val_for_loader))

    train_loader = DataLoader(
        train_for_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    quantiles = tuple(float(q) for q in args.quantiles.split(",")) if args.quantiles else None
    out_channels = len(quantiles) if quantiles else 1
    model = PluvioUNet(in_channels=train_set.n_channels, base_channels=args.base_channels,
                       out_channels=out_channels).to(device)
    LOG.info(
        "Model: PluvioUNet (%d channels, base=%d, %d parameters)",
        train_set.n_channels,
        args.base_channels,
        num_params(model),
    )

    loss_fn = CombinedLoss(
        bias_penalty=args.bias_penalty,
        fss_weight=args.fss_weight,
        fss_thresholds=fss_thresholds,
        fss_scales=fss_scales,
        fss_tau=args.fss_tau,
        sharpness_weight=args.sharpness_weight,
        quantiles=quantiles,
        quantile_weight=args.quantile_weight,
    )
    loss_config = {
        "bias_penalty": args.bias_penalty,
        "fss_weight": args.fss_weight,
        "fss_thresholds": fss_thresholds,
        "fss_scales": fss_scales,
        "fss_tau": args.fss_tau,
        "sharpness_weight": args.sharpness_weight,
        "quantiles": quantiles,
        "quantile_weight": args.quantile_weight,
    }
    median_index = loss_fn.median_index

    # The exact channel layout this run trained on, so infer_latest/benchmark
    # rebuild it rather than re-deriving it from a store that may have gained
    # channels since (2.3). None for the legacy HDF5 dataset, which has no
    # build_input recipe.
    if isinstance(train_set, ZarrCorrectionDataset):
        channel_recipe = train_set.channel_recipe()
    elif isinstance(train_set, ShardDataset):
        # The manifest already carries ``ds.channel_recipe()`` verbatim, as
        # written by the renderer from the very dataset these shards came out
        # of — so use it rather than re-deriving a subset here. Re-deriving is
        # how a shard-trained checkpoint ended up with no ``channel_names``
        # while a zarr-trained one had them: same run, two different recipes
        # depending on the dataset source.
        channel_recipe = train_set.manifest.get("channel_recipe")
        if not channel_recipe:
            raise ShardRecipeMismatch(
                f"--shards {args.shards}: the train manifest carries no channel_recipe, so "
                "the checkpoint would not record the channel layout it trained on (and "
                "infer_latest could not rebuild the input). Re-render the shards."
            )
        channel_recipe = dict(channel_recipe)
    else:
        channel_recipe = None   # legacy HDF5 dataset: no build_input recipe

    # Which rendered store this run actually read, so "was this checkpoint
    # trained on the shards I think it was" is answerable from the checkpoint
    # alone — the recipe hash identifies the sample semantics and the source
    # fingerprint identifies the zarr they were rendered from.
    shard_provenance = None
    if isinstance(train_set, ShardDataset):
        shard_provenance = {
            "root": str(args.shards),
            "recipe_hash": train_set.manifest.get("recipe_hash"),
            "source_store": train_set.manifest.get("source_store"),
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # Val RMSE at a fixed LR oscillates around an early best without ever
    # beating it (measured on the first full v2 runs: best at epoch 1-2, then
    # 30 epochs of 0.69-0.80 bounce while train loss keeps falling). Halve the
    # LR whenever val plateaus so the optimizer can settle into the minimum.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    # An FSS/sharpness objective deliberately gives up RMSE, so selecting on
    # RMSE stops the run at epoch 1. Default to the objective the model is
    # actually minimising, and say so in the log.
    select_on = resolve_select_on(args)
    LOG.info("model selection on %s (--select-on=%s)", select_on, args.select_on)
    best_val = float("inf")
    best_score = float("inf")
    patience = args.patience
    no_improve = 0
    checkpoint_path = pathlib.Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # metric -> "is a better than the best so far"
    ORDER = {"val_loss": lambda a, b: a < b, "val_rmse": lambda a, b: a < b,
             "val_fss3": lambda a, b: a > b}
    wanted = [m for m in _companion_metrics(args, select_on) if m in ORDER]
    companions = {m: ORDER[m] for m in wanted}
    companion_best = {m: (float("-inf") if m == "val_fss3" else float("inf")) for m in wanted}
    companion_paths = {
        m: checkpoint_path.with_name(f"{checkpoint_path.stem}.{m}{checkpoint_path.suffix}")
        for m in wanted
    }
    if wanted:
        LOG.info("also keeping one checkpoint per companion metric: %s",
                 ", ".join(f"{m} → {companion_paths[m].name}" for m in wanted))

    # Resumable state, written every epoch and on a stop signal: weights,
    # optimizer moments, LR-schedule position and every counter that decides
    # early stopping. Without it a reboot (or a machine wanted for something
    # else for an evening) costs the whole run, since only the best weights
    # were ever on disk and those carry no optimizer state.
    state_path = _state_path(args)
    first_epoch = 1
    resume_from = _resume_path(args)
    if resume_from is not None:
        st = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if st.get("scaler") is not None:
            scaler.load_state_dict(st["scaler"])
        first_epoch = int(st["epoch"]) + 1
        best_score = float(st["best_score"])
        best_val = float(st["best_val"])
        no_improve = int(st["no_improve"])
        companion_best.update({k: float(v) for k, v in (st.get("companion_best") or {}).items()
                               if k in companion_best})
        if st.get("select_on") and st["select_on"] != select_on:
            LOG.warning("resumed state selected on %r, this run selects on %r — "
                        "the early-stopping counter carries over anyway",
                        st["select_on"], select_on)
        LOG.info("resumed from %s at epoch %d (best %s=%.4f, %d epochs without improvement)",
                 resume_from, st["epoch"], select_on, abs(best_score), no_improve)

    # A stop signal saves state and exits cleanly, so pausing costs at most the
    # unfinished epoch's compute — never the run.
    stopping = {"signal": None}

    def _on_stop(signum, _frame):
        stopping["signal"] = signum
        LOG.warning("signal %d — saving state and stopping after this step", signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_stop)

    def _save_state(epoch: int) -> None:
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch,
                "best_score": best_score,
                "best_val": best_val,
                "no_improve": no_improve,
                "companion_best": companion_best,
                "select_on": select_on,
                "in_channels": train_set.n_channels,
                "base_channels": args.base_channels,
                "out_channels": out_channels,
                "quantiles": list(quantiles) if quantiles else None,
                "channel_recipe": channel_recipe,
            },
            tmp,
        )
        tmp.replace(state_path)   # atomic: a kill mid-write can't leave a torn state

    started = time.monotonic()

    for epoch in range(first_epoch, args.epochs + 1):
        train_loss, term_means, interrupted = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn,
            should_stop=lambda: stopping["signal"] is not None,
        )
        if interrupted:
            # Partial epoch: its metrics mean nothing, so skip validation and
            # selection entirely and record the state as "epoch-1 done" so the
            # resumed run repeats this epoch from these weights.
            _save_state(epoch - 1)
            LOG.info("paused inside epoch %d — resume with --resume auto (state: %s)",
                     epoch, state_path)
            return 0
        metrics = validate(model, val_loader, device, median_index, loss_fn=loss_fn)
        score = metrics[select_on] if select_on != "val_fss3" else -metrics["val_fss3"]
        scheduler.step(score)
        elapsed_min = (time.monotonic() - started) / 60
        LOG.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_rmse=%.4f val_fss3=%.4f "
            "val_wet_bias=%.2f lr=%.1e (%.1f min elapsed) [select_on=%s]",
            epoch, train_loss, metrics.get("val_loss", float("nan")), metrics["val_rmse"],
            metrics["val_fss3"], metrics["val_wet_bias"],
            optimizer.param_groups[0]["lr"], elapsed_min, select_on,
        )
        if args.fss_weight > 0 or args.sharpness_weight > 0:
            LOG.info(
                "  ↳ loss terms: %s",
                ", ".join(f"{k}={v:.4f}" for k, v in term_means.items()),
            )
        def _state(epoch=epoch, metrics=metrics, chosen_on=select_on) -> dict:
            return {
                "model": model.state_dict(),
                "val_rmse": metrics["val_rmse"],
                "val_loss": metrics.get("val_loss"),
                "val_fss3": metrics["val_fss3"],
                "val_wet_bias": metrics["val_wet_bias"],
                "select_on": chosen_on,
                "in_channels": train_set.n_channels,
                "base_channels": args.base_channels,
                "out_channels": out_channels,
                "quantiles": list(quantiles) if quantiles else None,
                "arch": "PluvioUNet",
                "epoch": epoch,
                "loss_config": loss_config,
                "channel_recipe": channel_recipe,
                "shards": shard_provenance,
            }

        # Companion checkpoints, one per alternative selection metric. The
        # loss is a weighted sum, so its minimum need not be the epoch with
        # the best forecast: on the first run under val_loss selection the
        # chosen epoch had val_fss3=0.597/bias 2.16 while a later epoch
        # reached 0.629/1.88 at a val_loss 0.003 worse. Keeping one checkpoint
        # per metric lets the frozen benchmark decide which weights ship,
        # instead of the argument being settled by whichever metric the run
        # happened to select on.
        for metric, better in companions.items():
            value = metrics.get(metric)
            if value is None:
                continue
            if better(value, companion_best[metric]):
                companion_best[metric] = value
                path = companion_paths[metric]
                torch.save(_state(chosen_on=metric), path)
                LOG.info("  ↳ companion (best %s=%.4f) → %s", metric, value, path)

        if score < best_score:
            best_score = score
            best_val = metrics["val_rmse"]
            no_improve = 0
            torch.save(_state(), checkpoint_path)
            LOG.info("  ↳ checkpoint saved → %s", checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                LOG.info("Early stopping at epoch %d (best %s=%.4f, val_rmse=%.4f)",
                         epoch, select_on, abs(best_score), best_val)
                break
        _save_state(epoch)
        if stopping["signal"] is not None:
            LOG.info("stopped after epoch %d — resume with --resume auto (state: %s)",
                     epoch, state_path)
            return 0
        if args.max_minutes is not None and elapsed_min >= args.max_minutes:
            LOG.info("Hit --max-minutes=%.1f budget; stopping.", args.max_minutes)
            break

    LOG.info("Training done. Best %s=%.4f (val_rmse=%.4f) → %s",
             select_on, abs(best_score), best_val, checkpoint_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
