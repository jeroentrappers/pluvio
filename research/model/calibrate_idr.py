"""Calibrate the nowcast model's rain probability (roadmap #3).

The model emits a *rate* (mm/h); the product needs a trustworthy **P(rain)**.
We fit, per lead, an isotonic map  model_rate → P(obs ≥ τ mm/h)  on the held-out
val split (isotonic = monotone, tuning-free, auto-calibrated — the IDR idea).
Reports per-lead reliability (predicted prob vs observed freq) + Brier + ECE, and
saves the calibrators. CPU-only, no GPU.

    python -m model.calibrate_idr --zarr seamless_0_6h_256_v2.zarr --ckpt checkpoints/nowcast_256.pt
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import pickle
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.seamless import SeamlessNet  # noqa: E402
from model.seamless_dataset import SeamlessDataset, issue_time_split  # noqa: E402

NOWCAST_LEADS = tuple(range(0, 121, 10))


def _ece(prob, label, n_bins=10):
    """Expected calibration error + per-bin (mean_pred, obs_freq, n)."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece, N = [], 0.0, len(prob)
    for i in range(n_bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < n_bins - 1 else prob <= 1.0)
        if not m.any():
            continue
        mp, of, k = float(prob[m].mean()), float(label[m].mean()), int(m.sum())
        rows.append((edges[i], edges[i + 1], mp, of, k))
        ece += (k / N) * abs(mp - of)
    return ece, rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=900)
    p.add_argument("--pixels-per-sample", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.1, help="rain threshold τ mm/h")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints/idr_calibrators.pkl")
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut = issue_time_split(args.zarr, args.val_frac)
    hi = datetime(2100, 1, 1, tzinfo=timezone.utc)
    ds = SeamlessDataset(args.zarr, time_range=(cut, hi), leads_min=NOWCAST_LEADS)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    quantiles = ck.get("quantiles")
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"],
                      quantiles=tuple(quantiles) if quantiles else None).to(dev).eval()
    net.load_state_dict(ck["model"])
    print(f"IDR calibration | {args.ckpt} | τ={args.threshold} mm/h | val={len(ds)} | dev={dev}")

    random.seed(0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    leads = np.array([ds.index[i].lead_min for i in idxs])
    dl = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, num_workers=args.workers)

    rng = np.random.default_rng(0)
    acc = collections.defaultdict(lambda: ([], []))  # lead → (rates, labels)
    ptr = 0
    with torch.no_grad():
        for x, cond, y in dl:
            bs = x.size(0); lead_b = leads[ptr:ptr + bs]; ptr += bs
            with torch.autocast(dev.type, enabled=(dev.type == "cuda")):
                out = net(x.to(dev), cond.to(dev))
            point = (net.median(out) if quantiles else out).squeeze(1).cpu().numpy()  # (bs,H,W)
            obs = y.squeeze(1).numpy()
            flatn = point.shape[1] * point.shape[2]
            for j in range(bs):
                sel = rng.integers(0, flatn, size=min(args.pixels_per_sample, flatn))
                r, l = acc[int(lead_b[j])]
                r.append(point[j].ravel()[sel])
                l.append((obs[j].ravel()[sel] >= args.threshold).astype("float32"))

    calibrators, summary = {}, []
    for lead in NOWCAST_LEADS:
        if lead not in acc:
            continue
        rates = np.concatenate(acc[lead][0]); labels = np.concatenate(acc[lead][1])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(rates, labels)
        calibrators[lead] = iso
        prob = iso.predict(rates)
        base = float(labels.mean())
        brier = float(np.mean((prob - labels) ** 2))
        brier_clim = base * (1 - base)  # climatology Brier = p(1-p)
        ece, _ = _ece(prob, labels)
        summary.append((lead, len(labels), base, brier, brier_clim, ece))

    print(f"\n{'lead':>5}{'n_pix':>9}{'baserate':>10}{'Brier_cal':>11}{'Brier_clim':>12}{'BSS':>7}{'ECE':>7}")
    print("-" * 61)
    for lead, npx, base, brier, briclim, ece in summary:
        bss = 1 - brier / briclim if briclim > 0 else float("nan")  # Brier skill score vs climatology
        print(f"{lead:>5}{npx:>9}{base:>10.4f}{brier:>11.4f}{briclim:>12.4f}{bss:>7.3f}{ece:>7.3f}")
    print("\nBSS>0 = better than climatology; ECE→0 = well-calibrated (post-isotonic should be low).")

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"threshold": args.threshold, "leads": list(calibrators),
                     "calibrators": calibrators}, f)
    print(f"\nsaved {len(calibrators)} per-lead calibrators → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
