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

Not implemented yet (deliberate — needs the dataset class to be wired):
- the actual call to PluvioCorrectionDataset (raises NotImplementedError
  until ``_build_index`` is filled in against the on-disk layout).
- ONNX / TFLite export at the end of training.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.dataset import PluvioCorrectionDataset  # noqa: E402
from model.unet import PluvioUNet, num_params  # noqa: E402

LOG = logging.getLogger("pluvio.train")


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
    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            pred = model(x)
            loss = weighted_huber(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    rmses: list[float] = []
    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        rmses.append(float(rmse(pred, y).cpu()))
    return {"val_rmse": sum(rmses) / max(len(rmses), 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/pluvio_unet.pt")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    device = torch.device(args.device)
    LOG.info("Training on %s", device)

    train_set = PluvioCorrectionDataset(args.data / "train")
    val_set = PluvioCorrectionDataset(args.data / "val")
    LOG.info("Train: %d samples | Val: %d samples", len(train_set), len(val_set))

    train_loader = DataLoader(
        train_set,
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

    model = PluvioUNet().to(device)
    LOG.info("Model: PluvioUNet (%d parameters)", num_params(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val = float("inf")
    patience = 5
    no_improve = 0
    checkpoint_path = pathlib.Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        metrics = validate(model, val_loader, device)
        LOG.info(
            "Epoch %d: train_loss=%.4f val_rmse=%.4f", epoch, train_loss, metrics["val_rmse"]
        )
        if metrics["val_rmse"] < best_val:
            best_val = metrics["val_rmse"]
            no_improve = 0
            torch.save({"model": model.state_dict(), "val_rmse": best_val}, checkpoint_path)
            LOG.info("  ↳ checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                LOG.info("Early stopping at epoch %d (best val_rmse=%.4f)", epoch, best_val)
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
