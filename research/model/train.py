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
    """Huber loss weighted by ``(1 + obs)²`` so heavy rain matters."""
    delta = 1.0
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    lin = abs_diff - quad
    per_pixel = 0.5 * quad**2 + delta * lin
    weight = (1.0 + target) ** 2
    return (per_pixel * weight).mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = weighted_huber(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = weighted_huber(pred, y)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))


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
    parser.add_argument("--data", type=pathlib.Path, required=True,
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
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop training after this many wall-clock minutes (CPU budget guard).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    device = torch.device(args.device)
    LOG.info("Training on %s", device)

    split = _time_split(args.data, args.val_frac)
    LOG.info("Time split: train < %s ≤ val", split.isoformat())
    train_set = PluvioCorrectionDataset(
        args.data,
        time_range=(_DT_MIN, split),
        require_rain_fraction=args.require_rain_fraction,
    )
    val_set = PluvioCorrectionDataset(args.data, time_range=(split, _DT_MAX))

    train_for_loader: torch.utils.data.Dataset = train_set
    if args.max_train_samples is not None and len(train_set) > args.max_train_samples:
        import torch.utils.data as tud

        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(len(train_set), generator=g)[: args.max_train_samples].tolist()
        train_for_loader = tud.Subset(train_set, pick)
    LOG.info("Train: %d samples (using %d) | Val: %d samples",
             len(train_set), len(train_for_loader), len(val_set))

    train_loader = DataLoader(
        train_for_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    patience = 5
    no_improve = 0
    checkpoint_path = pathlib.Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        metrics = validate(model, val_loader, device)
        elapsed_min = (time.monotonic() - started) / 60
        LOG.info(
            "Epoch %d: train_loss=%.4f val_rmse=%.4f (%.1f min elapsed)",
            epoch, train_loss, metrics["val_rmse"], elapsed_min,
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
