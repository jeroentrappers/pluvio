"""Eval for the ERA5→OPERA downscaling pretraining (Stage A).

The right zero-skill baseline here is **the coarse ERA5 precip itself** (already
bilinearly on the analysis grid, channel `era5_tp`): does the net sharpen/correct
it toward the OPERA truth? Reports MAE/RMSE for model vs era5_tp vs dry.

skill% = 100·(1 − MAE_model / MAE_era5tp): positive = we beat raw ERA5 precip.

    python -m model.eval_downscale --zarr pretrain_v2.zarr --ckpt checkpoints/pretrain.pt
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.seamless import SeamlessNet  # noqa: E402
from model.seamless_dataset import SeamlessDataset, issue_time_split  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--baseline-channel", default="era5_tp")
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut = issue_time_split(args.zarr, args.val_frac)
    hi = datetime(2100, 1, 1, tzinfo=timezone.utc)
    ds = SeamlessDataset(args.zarr, time_range=(cut, hi), history_steps=0)
    try:
        base_idx = ds.aux_channels.index(args.baseline_channel)
    except ValueError:
        print(f"baseline channel {args.baseline_channel} not in {ds.aux_channels}")
        return 2

    ck = torch.load(args.ckpt, map_location=dev)
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt}: in_ch={ck['in_channels']} val={len(ds)} "
          f"baseline=ch[{base_idx}]={args.baseline_channel} dev={dev}")

    random.seed(0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    dl = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, num_workers=args.workers)
    sm = sp = sz = em = ep = 0.0
    nn = 0
    with torch.no_grad():
        for x, cond, y in dl:
            x, cond, y = x.to(dev), cond.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=(dev.type == "cuda")):
                pred = net(x, cond).float()
            base = x[:, base_idx:base_idx + 1]  # raw ERA5 coarse precip (mm/h)
            sm += float((pred - y).abs().mean(dim=(1, 2, 3)).sum())
            sp += float((base - y).abs().mean(dim=(1, 2, 3)).sum())
            sz += float(y.abs().mean(dim=(1, 2, 3)).sum())
            em += float(((pred - y) ** 2).mean(dim=(1, 2, 3)).sum())
            ep += float(((base - y) ** 2).mean(dim=(1, 2, 3)).sum())
            nn += x.size(0)

    mae_m, mae_b, mae_z = sm / nn, sp / nn, sz / nn
    rmse_m, rmse_b = (em / nn) ** 0.5, (ep / nn) ** 0.5
    skill = 100 * (1 - mae_m / mae_b) if mae_b > 0 else float("nan")
    print(f"\n{'':12}{'MAE':>10}{'RMSE':>10}")
    print(f"{'model':12}{mae_m:>10.4f}{rmse_m:>10.4f}")
    print(f"{'era5_tp':12}{mae_b:>10.4f}{rmse_b:>10.4f}")
    print(f"{'dry':12}{mae_z:>10.4f}{'':>10}")
    print(f"\nskill vs raw ERA5 precip: {skill:+.1f}%  (n={nn})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
