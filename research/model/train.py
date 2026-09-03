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
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.dataset import PluvioCorrectionDataset  # noqa: E402
from model.losses import CombinedLoss  # noqa: E402
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


def weighted_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss weighted by ``(1 + obs)`` so heavy rain matters.

    Softened from the original ``(1 + obs)²``: the squared weight made the
    optimizer hedge precipitation upward everywhere, producing a persistent
    wet bias. Linear weighting keeps the heavy-rain emphasis without the
    systematic over-prediction.
    """
    delta = 1.0
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    lin = abs_diff - quad
    per_pixel = 0.5 * quad**2 + delta * lin
    weight = 1.0 + target
    return (per_pixel * weight).mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor, bias_penalty: float) -> torch.Tensor:
    """Weighted Huber + a penalty on the systematic (batch-mean) bias.

    The bias term directly punishes ``mean(pred) - mean(target)``, which is
    the exact quantity we saw drift to +0.14 mm/h. Keeps the model honest
    about *how much* rain, not just *where*.
    """
    base = weighted_huber(pred, target)
    bias = (pred.mean() - target.mean()).pow(2)
    return base + bias_penalty * bias


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    loss_fn: CombinedLoss,
) -> tuple[float, dict[str, float]]:
    """Runs one epoch. Returns the mean batch loss and the mean of each
    loss component (``loss_fn.last_terms``) over the epoch — components with
    zero weight are simply absent from every batch's ``last_terms`` dict."""
    model.train()
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
    mean_terms = {k: term_sums[k] / term_counts[k] for k in term_sums}
    return float(sum(losses) / max(len(losses), 1)), mean_terms


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    rmses: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        rmses.append(float(rmse(pred, y).cpu()))
    return {"val_rmse": sum(rmses) / max(len(rmses), 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=pathlib.Path, default=None,
                        help="Path to timeseries.zarr. If set, trains on the full "
                             "multi-source store (ZarrCorrectionDataset) instead of "
                             "the legacy radar-HDF5 dataset under --data.")
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
    parser.add_argument("--sharpness-weight", type=float, default=0.0,
                        help="Weight on the gradient-energy sharpness loss "
                             "(0 = disabled, matching the pre-existing Huber-only objective).")
    parser.add_argument("--patience", type=int, default=30,
                        help="early-stopping patience in epochs (val RMSE plateau)")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop training after this many wall-clock minutes (CPU budget guard).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    device = torch.device(args.device)
    LOG.info("Training on %s", device)

    if not args.zarr and not args.data:
        raise SystemExit("provide --zarr <timeseries.zarr> (multi-source) or --data <radar dir> (legacy)")

    if args.zarr:
        split = issue_time_split(args.zarr, args.val_frac)
        LOG.info("Time split (zarr): train < %s ≤ val", split.isoformat())
        train_set = ZarrCorrectionDataset(
            args.zarr, time_range=(_DT_MIN, split),
            require_rain_fraction=args.require_rain_fraction,
        )
        val_set = ZarrCorrectionDataset(args.zarr, time_range=(split, _DT_MAX))
    else:
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

    model = PluvioUNet(in_channels=train_set.n_channels, base_channels=args.base_channels).to(device)
    LOG.info(
        "Model: PluvioUNet (%d channels, base=%d, %d parameters)",
        train_set.n_channels,
        args.base_channels,
        num_params(model),
    )

    fss_thresholds = tuple(float(v) for v in args.fss_thresholds.split(",") if v.strip())
    fss_scales = tuple(int(v) for v in args.fss_scales.split(",") if v.strip())
    loss_fn = CombinedLoss(
        bias_penalty=args.bias_penalty,
        fss_weight=args.fss_weight,
        fss_thresholds=fss_thresholds,
        fss_scales=fss_scales,
        sharpness_weight=args.sharpness_weight,
    )
    loss_config = {
        "bias_penalty": args.bias_penalty,
        "fss_weight": args.fss_weight,
        "fss_thresholds": fss_thresholds,
        "fss_scales": fss_scales,
        "sharpness_weight": args.sharpness_weight,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # Val RMSE at a fixed LR oscillates around an early best without ever
    # beating it (measured on the first full v2 runs: best at epoch 1-2, then
    # 30 epochs of 0.69-0.80 bounce while train loss keeps falling). Halve the
    # LR whenever val plateaus so the optimizer can settle into the minimum.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4)

    best_val = float("inf")
    patience = args.patience
    no_improve = 0
    checkpoint_path = pathlib.Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss, term_means = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn
        )
        metrics = validate(model, val_loader, device)
        scheduler.step(metrics["val_rmse"])
        elapsed_min = (time.monotonic() - started) / 60
        LOG.info(
            "Epoch %d: train_loss=%.4f val_rmse=%.4f lr=%.1e (%.1f min elapsed)",
            epoch, train_loss, metrics["val_rmse"],
            optimizer.param_groups[0]["lr"], elapsed_min,
        )
        if args.fss_weight > 0 or args.sharpness_weight > 0:
            LOG.info(
                "  ↳ loss terms: %s",
                ", ".join(f"{k}={v:.4f}" for k, v in term_means.items()),
            )
        if metrics["val_rmse"] < best_val:
            best_val = metrics["val_rmse"]
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_rmse": best_val,
                    "in_channels": train_set.n_channels,
                    "base_channels": args.base_channels,
                    "arch": "PluvioUNet",
                    "epoch": epoch,
                    "loss_config": loss_config,
                },
                checkpoint_path,
            )
            LOG.info("  ↳ checkpoint saved → %s", checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                LOG.info("Early stopping at epoch %d (best val_rmse=%.4f)", epoch, best_val)
                break
        if args.max_minutes is not None and elapsed_min >= args.max_minutes:
            LOG.info("Hit --max-minutes=%.1f budget; stopping.", args.max_minutes)
            break

    LOG.info("Training done. Best val_rmse=%.4f → %s", best_val, checkpoint_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
