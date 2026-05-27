"""Evaluate a trained correction model against the operational nowcast.

For every sample in the validation split we have three things on the same
grid: the **operational** KNMI nowcast (input channel 6), the **model's**
corrected field, and the **observed** truth (the target). We compute the
same metrics the verification notebook uses — MAE / RMSE / bias and
categorical CSI / POD / FAR — for both the operational baseline and the
model, side by side, stratified by lead time.

The headline question: does the model beat the operational nowcast,
especially on heavy rain at longer leads (the documented weakness)?

    python -m model.evaluate --data ../data/knmi --checkpoint checkpoints/pluvio_unet.pt
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))
import _lib as kpi  # noqa: E402
from model.dataset import PluvioCorrectionDataset  # noqa: E402
from model.train import _DT_MAX, _time_split  # noqa: E402
from model.unet import PluvioUNet  # noqa: E402

LOG = logging.getLogger("pluvio.evaluate")

# Channel 6 is the operational nowcast frame (see dataset.py docstring).
OPERATIONAL_CHANNEL = 6


def _tidy_from_samples(
    dataset: PluvioCorrectionDataset,
    model: torch.nn.Module,
    device: torch.device,
    sample_cells: int,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a tidy frame: one row per (sampled cell, lead) with the
    operational forecast, the model forecast, and the observation."""
    rng = np.random.default_rng(rng_seed)
    h, w = dataset[0][0].shape[-2:]
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            lead = dataset.index[i].lead_min
            operational = x[OPERATIONAL_CHANNEL].numpy()
            observed = y[0].numpy()
            pred = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()

            flat = h * w
            idx = rng.choice(flat, size=min(sample_cells, flat), replace=False)
            ys, xs = np.unravel_index(idx, (h, w))
            for r, c in zip(ys, xs, strict=True):
                rows.append(
                    {
                        "lead_min": lead,
                        "operational_mm_per_h": float(operational[r, c]),
                        "model_mm_per_h": float(pred[r, c]),
                        "observed_mm_per_h": float(observed[r, c]),
                    }
                )
    return pd.DataFrame(rows)


def _metrics_for(df: pd.DataFrame, forecast_col: str, threshold: float) -> pd.DataFrame:
    """Continuous + categorical metrics per lead for one forecast column."""
    renamed = df.rename(
        columns={forecast_col: "forecast_mm_per_h", "observed_mm_per_h": "observed_mm_per_h"}
    )[["lead_min", "forecast_mm_per_h", "observed_mm_per_h"]]
    cont = kpi.continuous_metrics(renamed)
    cat = kpi.categorical_metrics(renamed, threshold=threshold)
    return cont.merge(cat[["lead_min", "csi", "pod", "far"]], on="lead_min")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=1.0, help="rain/no-rain mm/h")
    parser.add_argument("--sample-cells", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    in_channels = ckpt.get("in_channels", 10)
    model = PluvioUNet(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model"])
    LOG.info("Loaded %s (val_rmse=%.4f, epoch=%s)",
             args.checkpoint, ckpt.get("val_rmse", float("nan")), ckpt.get("epoch"))

    split = _time_split(args.data, args.val_frac)
    val_set = PluvioCorrectionDataset(args.data, time_range=(split, _DT_MAX))
    LOG.info("Validation samples: %d", len(val_set))

    df = _tidy_from_samples(val_set, model, device, args.sample_cells)
    LOG.info("Built %d (cell, lead) evaluation rows", len(df))

    op = _metrics_for(df, "operational_mm_per_h", args.threshold)
    md = _metrics_for(df, "model_mm_per_h", args.threshold)
    merged = op.merge(md, on="lead_min", suffixes=("_op", "_model"))

    print(f"\n=== Operational nowcast vs. model (τ = {args.threshold} mm/h) ===")
    print("lead | RMSE op→model |  CSI op→model |  bias op→model")
    print("-----|---------------|---------------|----------------")
    for _, r in merged.iterrows():
        print(
            f"{int(r['lead_min']):>4} | "
            f"{r['rmse_op']:.3f} → {r['rmse_model']:.3f} | "
            f"{r['csi_op']:.3f} → {r['csi_model']:.3f} | "
            f"{r['bias_op']:+.3f} → {r['bias_model']:+.3f}"
        )

    # Aggregate verdict.
    rmse_win = (merged["rmse_model"] < merged["rmse_op"]).mean()
    csi_win = (merged["csi_model"] > merged["csi_op"]).mean()
    print(
        f"\nModel beats operational on RMSE at {rmse_win:.0%} of leads, "
        f"on CSI at {csi_win:.0%} of leads."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
