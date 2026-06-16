"""Per-lead-band eval for the unified Stage-B seamless model.

The model takes radar history (nowcast head) + the ERA5 anchor read at the
*valid* time (outlook head) and blends by lead. So each regime has its own
honest reference:
  * nowcast/short → persistence (last radar frame carried forward);
  * outlook        → the raw ERA5 anchor (era5_tp at the valid time).
Reports MAE/RMSE for model vs persistence vs era5_tp vs dry, per band.

    python -m model.eval_stageb --zarr stageb_v2.zarr --ckpt checkpoints/stageb.pt
"""

from __future__ import annotations

import argparse
import collections
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

BANDS = [("nowcast <=2h", 0, 121), ("short 2-6h", 121, 361),
         ("medium 6-24h", 361, 1441), ("long 1-10d", 1441, 14401)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut = issue_time_split(args.zarr, args.val_frac)
    hi = datetime(2100, 1, 1, tzinfo=timezone.utc)
    ds = SeamlessDataset(args.zarr, time_range=(cut, hi), history_steps=6, aux_at_valid_time=True)
    H = ds.history_steps
    tp_ch = H + (1 if ds.has_aifs else 0) + ds.aux_channels.index("era5_tp")  # era5 anchor channel in x

    ck = torch.load(args.ckpt, map_location=dev)
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt}: in_ch={ck['in_channels']} val={len(ds)} "
          f"persist=ch[{H-1}] era5=ch[{tp_ch}] dev={dev}")

    random.seed(0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    leads = np.array([ds.index[i].lead_min for i in idxs])
    dl = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, num_workers=args.workers)
    res = {b[0]: collections.defaultdict(float) for b in BANDS}
    ptr = 0
    with torch.no_grad():
        for x, cond, y in dl:
            bs = x.size(0); lead_b = leads[ptr:ptr + bs]; ptr += bs
            x, cond, y = x.to(dev), cond.to(dev), y.to(dev)
            with torch.autocast(dev.type, enabled=(dev.type == "cuda")):
                pred = net(x, cond).float()
            per = x[:, H - 1:H]          # persistence (last radar frame)
            era = x[:, tp_ch:tp_ch + 1]  # raw ERA5 anchor at valid time
            am = (pred - y).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            ap = (per - y).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            ae = (era - y).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            az = y.abs().mean(dim=(1, 2, 3)).cpu().numpy()
            sm = ((pred - y) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            for j in range(bs):
                for name, lo, hh in BANDS:
                    if lo <= lead_b[j] < hh:
                        r = res[name]
                        r["m"] += am[j]; r["p"] += ap[j]; r["e"] += ae[j]; r["z"] += az[j]
                        r["sm"] += sm[j]; r["n"] += 1
                        break

    hdr = f"{'band':<14}{'n':>7}{'MAE_mdl':>9}{'MAE_per':>9}{'MAE_era5':>9}{'MAE_dry':>9}{'RMSE_mdl':>10}"
    print(hdr); print("-" * len(hdr))
    for name, lo, hh in BANDS:
        r = res[name]
        if not r["n"]:
            continue
        k = r["n"]
        print(f"{name:<14}{int(k):>7}{r['m']/k:>9.4f}{r['p']/k:>9.4f}{r['e']/k:>9.4f}"
              f"{r['z']/k:>9.4f}{(r['sm']/k)**0.5:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
