"""Training entrypoint for the seamless 0–240 h model (P3).

    python -m model.train_seamless --zarr /opt/pluvio/zarr/seamless.zarr \
        --epochs 30 --batch-size 16 --out checkpoints/pluvio_seamless.pt

Pairs `SeamlessDataset` (multimodal + lead/time cond + OPERA truth) with
`SeamlessNet` (dual head + seam). Weighted Huber so heavy rain isn't drowned by
the 95 %-dry distribution; AMP on CUDA; time-based early stop. Verification is
the separate head-to-head harness (extend `model/evaluate.py`).

⚠️ Requires the built seamless zarr (tools/build_seamless_zarr.py) and a GPU for
a real run — it is NOT auto-run anywhere. The CPU path (tiny `--max-minutes`)
exists only to smoke-test the loop.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.seamless import SeamlessNet, num_params  # noqa: E402
from model.seamless_dataset import SeamlessDataset, issue_time_split  # noqa: E402  (added below)

LOG = logging.getLogger("pluvio.train_seamless")


def precip_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0,
                weight_cap: float = 5.0) -> torch.Tensor:
    """Huber loss in **log1p space** with a mild, capped rain weight.

    The first GPU run used ``w = (1 + obs)²`` in *linear* mm/h space. On a ~98%-dry
    domain that makes a handful of heavy-rain pixels dominate the objective by a
    factor of hundreds, so the optimizer over-predicts/blurs rain everywhere to
    hedge — which beat the weighted loss but lost to persistence on plain MAE
    (skill −189%). Computing the Huber in log1p space compresses the heavy tail
    (so extremes no longer swamp the gradient) and a capped linear weight keeps
    *some* heavy-rain emphasis without sacrificing the dry field. Dry pixels
    (target 0 → log1p 0) are cheap to get right, so the over-prediction bias goes
    away. Output stays mm/h (Softplus), so inference/eval are unchanged.
    """
    lp, lt = torch.log1p(pred), torch.log1p(target)
    w = torch.clamp(1.0 + target, max=weight_cap)
    loss = F.huber_loss(lp, lt, delta=delta, reduction="none")
    return (w * loss).mean()


def quantile_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles,
                  weight_cap: float = 5.0) -> torch.Tensor:
    """Pinball (quantile) loss averaged over quantile levels — the CRPS-consistent
    objective for the probabilistic outlook head (rec #3).

    The mean pinball loss over a set of quantile levels is a discrete
    approximation to the CRPS, so minimising it trains *calibrated spread*, not
    just a point. Computed in log1p space with the same capped rain weight as
    ``precip_loss`` so the dry field and the heavy tail stay balanced.

        pred_q : (B, Q, H, W) monotone non-crossing quantiles (mm/h)
        target : (B, 1, H, W) truth (mm/h)
        quantiles : the Q levels, e.g. (0.1, 0.5, 0.9)
    """
    lt = torch.log1p(target)                       # (B,1,H,W)
    lp = torch.log1p(pred_q)                       # (B,Q,H,W)
    w = torch.clamp(1.0 + target, max=weight_cap)  # broadcasts over Q
    err = lt - lp
    q = torch.tensor(quantiles, device=pred_q.device, dtype=err.dtype)[None, :, None, None]
    pinball = torch.maximum(q * err, (q - 1.0) * err)
    return (w * pinball).mean()


def run_epoch(model, loader, optim, device, scaler, train: bool, loss_fn=None):
    loss_fn = loss_fn or precip_loss
    model.train(train)
    total, n = 0.0, 0
    for x, cond, y in loader:
        x, cond, y = x.to(device), cond.to(device), y.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=(device.type == "cuda")):
            pred = model(x, cond)
            loss = loss_fn(pred, y)
        if train:
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
            else:
                loss.backward(); optim.step()
        total += float(loss) * x.size(0); n += x.size(0)
    return total / max(1, n)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--history-steps", type=int, default=6,
                   help="radar-history input frames; 0 = pure downscaling (ERA5 pretraining)")
    p.add_argument("--aux-at-valid-time", action="store_true",
                   help="Stage B: read the ERA5/aux anchor at the valid-time (issue+lead) "
                        "as a perfect-forecast proxy the outlook head downscales")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-minutes", type=float, default=None, help="wall-clock cap (laptop/CPU smoke test)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints/pluvio_seamless.pt")
    p.add_argument("--quantiles", default="",
                   help="comma quantile levels for a probabilistic outlook head, "
                        "e.g. '0.1,0.5,0.9' (rec #3). Empty = deterministic.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut = issue_time_split(args.zarr, args.val_frac)
    from datetime import datetime, timezone
    lo, hi = datetime(1970, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc)
    train_ds = SeamlessDataset(args.zarr, time_range=(lo, cut), require_rain_fraction=0.0,
                               history_steps=args.history_steps, aux_at_valid_time=args.aux_at_valid_time)
    val_ds = SeamlessDataset(args.zarr, time_range=(cut, hi), history_steps=args.history_steps,
                             aux_at_valid_time=args.aux_at_valid_time)
    LOG.info("train=%d val=%d | %d channels | device=%s", len(train_ds), len(val_ds), train_ds.n_channels, device)

    quantiles = tuple(float(q) for q in args.quantiles.split(",") if q.strip()) or None
    model = SeamlessNet(in_channels=train_ds.n_channels, base_channels=args.base_channels,
                        quantiles=quantiles).to(device)
    if quantiles:
        LOG.info("probabilistic outlook head: quantiles=%s (CRPS-consistent pinball loss)", quantiles)
        loss_fn = lambda pred, y: quantile_loss(pred, y, quantiles)  # noqa: E731
    else:
        loss_fn = precip_loss
    LOG.info("SeamlessNet params: %s", f"{num_params(model):,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    best = float("inf")
    start = time.time()
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        tr = run_epoch(model, tl, optim, device, scaler, train=True, loss_fn=loss_fn)
        va = run_epoch(model, vl, optim, device, scaler, train=False, loss_fn=loss_fn)
        LOG.info("epoch %d: train=%.4f val=%.4f", epoch, tr, va)
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "in_channels": train_ds.n_channels,
                        "base_channels": args.base_channels, "val_loss": best,
                        "quantiles": list(quantiles) if quantiles else None}, out)
        if args.max_minutes and (time.time() - start) / 60 > args.max_minutes:
            LOG.info("hit --max-minutes; stopping"); break
    LOG.info("done. best val=%.4f → %s", best, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
