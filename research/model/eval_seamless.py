"""Head-to-head eval for the seamless model — against the baselines that matter.

The first version of this harness scored only vs **persistence** and **dry**.
Persistence is the floor: beating it past +30 min is nearly automatic and proves
nothing (rec #1). This version adds the references a meteorologist would demand
and the metrics that aren't fooled by a 98 %-dry domain:

Baselines, each scored with the *same* metrics as the model, per lead band:
  * **persistence** — last analysis carried forward (zero-skill floor);
  * **optical-flow** — pysteps-style advection extrapolation (model/classical.py):
    the real 0–2 h nowcast bar. Beating *this* is the nowcast claim;
  * **raw AIFS** — the NWP field already in the input stack (`aifs_tp`): the real
    outlook bar. The downscaler must beat *this* to justify itself.

Metrics: MAE/RMSE (reference only), **CSI** at τ=0.1/1, **FSS** across
neighbourhood scales (scale-aware — exposes whether a CSI win is just the coarse
grid), and **CRPS** (= MAE for a point forecast; from quantiles for a
probabilistic checkpoint). The grid resolution is printed alongside so CSI is
never read as if it were a 1-km score.

    python -m model.eval_seamless --zarr seamless.zarr --ckpt checkpoints/pluvio_seamless.pt
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
from model import metrics as M  # noqa: E402
from model.classical import optical_flow_nowcast  # noqa: E402
from model.geo import grid_resolution_km  # noqa: E402
from model.seamless import SeamlessNet  # noqa: E402
from model.seamless_dataset import SeamlessDataset, issue_time_split  # noqa: E402

BANDS = [
    ("nowcast <=2h", 0, 121, "optical-flow"),
    ("short 2-6h", 121, 361, "optical-flow"),
    ("medium 6-24h", 361, 1441, "raw-AIFS"),
    ("long 1-10d", 1441, 14401, "raw-AIFS"),
]
THRESHOLDS = (0.1, 1.0)
FSS_SCALES_PX = (1, 3, 5)


class _BandAcc:
    """Streaming metric accumulator for one band, over several methods."""

    def __init__(self, methods: list[str]):
        self.methods = methods
        self.n = 0
        self.ae = {m: 0.0 for m in methods}
        self.se = {m: 0.0 for m in methods}
        self.crps = {m: 0.0 for m in methods}
        self.cat = {(m, t): [0, 0, 0] for m in methods for t in THRESHOLDS}  # h,m,fa
        self.fss = {(m, s): [0.0, 0.0] for m in methods for s in FSS_SCALES_PX}  # num,den (τ=0.1)

    def update(self, obs: np.ndarray, fields: dict, quantile_fields: dict | None):
        self.n += 1
        for m, pred in fields.items():
            e = pred - obs
            self.ae[m] += float(np.mean(np.abs(e)))
            self.se[m] += float(np.mean(e ** 2))
            # CRPS: from quantiles where available, else point (== MAE).
            if quantile_fields and m in quantile_fields:
                self.crps[m] += M.crps_from_quantiles(quantile_fields[m], obs, quantile_fields["_levels"])
            else:
                self.crps[m] += M.crps_deterministic(pred, obs)
            for t in THRESHOLDS:
                sc = M.categorical_scores(pred, obs, t)
                acc = self.cat[(m, t)]
                acc[0] += sc["hits"]; acc[1] += sc["misses"]; acc[2] += sc["false_alarms"]
            for s in FSS_SCALES_PX:
                pf = M._fraction_field(pred >= 0.1, s)
                of = M._fraction_field(obs >= 0.1, s)
                f = self.fss[(m, s)]
                f[0] += float(np.mean((pf - of) ** 2))
                f[1] += float(np.mean(pf ** 2) + np.mean(of ** 2))

    def csi(self, m: str, t: float) -> float:
        h, mi, fa = self.cat[(m, t)]
        return h / (h + mi + fa) if (h + mi + fa) else float("nan")

    def fss_at(self, m: str, s: int) -> float:
        num, den = self.fss[(m, s)]
        return 1.0 - num / den if den else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-optflow", action="store_true", help="skip the optical-flow baseline (faster)")
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut = issue_time_split(args.zarr, args.val_frac)
    hi = datetime(2100, 1, 1, tzinfo=timezone.utc)
    ds = SeamlessDataset(args.zarr, time_range=(cut, hi))
    Hsteps = ds.history_steps
    dt_min = ds.history_step_min

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    quantiles = ck.get("quantiles")
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"],
                      quantiles=tuple(quantiles) if quantiles else None).to(dev).eval()
    net.load_state_dict(ck["model"])

    dy_km, dx_km = grid_resolution_km()
    print(f"loaded {args.ckpt}: in_ch={ck['in_channels']} base={ck['base_channels']} "
          f"quantiles={quantiles} | val={len(ds)} dev={dev}")
    print(f"GRID resolution ≈ {dy_km:.1f}×{dx_km:.1f} km/cell — CSI at fine τ is inflated by "
          f"coarse pixels; read FSS(scale) for the scale-aware picture.")
    print("per-band reference baseline = the bar the model must beat in that regime.\n")

    methods = ["model", "persistence"]
    if not args.no_optflow:
        methods.append("optical-flow")
    if ds.has_aifs:
        methods.append("raw-AIFS")

    random.seed(0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    leads = np.array([ds.index[i].lead_min for i in idxs])
    accs = {b[0]: _BandAcc(methods) for b in BANDS}

    sub = Subset(ds, idxs)
    dl = DataLoader(sub, batch_size=args.batch_size, num_workers=args.workers)  # sequential → aligns with leads
    ptr = 0
    with torch.no_grad():
        for x, cond, y in dl:
            bs = x.size(0)
            lead_b = leads[ptr:ptr + bs]; ptr += bs
            xg, condg = x.to(dev), cond.to(dev)
            with torch.autocast(dev.type, enabled=(dev.type == "cuda")):
                out = net(xg, condg).float()
            point = net.median(out) if quantiles else out
            point_np = point.squeeze(1).cpu().numpy()
            out_np = out.cpu().numpy()
            xn = x.numpy(); yn = y.squeeze(1).numpy()
            for j in range(bs):
                lm = int(lead_b[j])
                band = next((b for b in BANDS if b[1] <= lm < b[2]), None)
                if band is None:
                    continue
                obs = yn[j]
                fields = {"model": point_np[j], "persistence": xn[j, Hsteps - 1]}
                if "raw-AIFS" in methods:
                    fields["raw-AIFS"] = xn[j, Hsteps]  # AIFS channel @ this lead
                if "optical-flow" in methods:
                    history = xn[j, :Hsteps]
                    of, _ = optical_flow_nowcast(history, [lm], dt_min=dt_min)
                    fields["optical-flow"] = of[0]
                qf = None
                if quantiles:
                    qf = {"model": out_np[j], "_levels": quantiles}
                accs[band[0]].update(obs, fields, qf)

    # ─────────────────────────────────────────────────────────── report
    for name, _lo, _hh, ref in BANDS:
        a = accs[name]
        if not a.n:
            continue
        print(f"━━ {name}  (n={a.n}, reference = {ref}) ━━")
        hdr = f"{'method':<13}{'MAE':>8}{'RMSE':>8}{'CRPS':>8}{'CSI.1':>8}{'CSI1':>8}" + \
              "".join(f"{'FSS'+str(s):>8}" for s in FSS_SCALES_PX)
        print(hdr)
        for m in a.methods:
            row = (f"{m:<13}{a.ae[m]/a.n:>8.3f}{(a.se[m]/a.n)**0.5:>8.3f}{a.crps[m]/a.n:>8.3f}"
                   f"{a.csi(m,0.1):>8.3f}{a.csi(m,1.0):>8.3f}"
                   + "".join(f"{a.fss_at(m,s):>8.3f}" for s in FSS_SCALES_PX))
            print(row)
        # verdict against the regime's reference baseline
        if ref in a.methods:
            dmae = 100 * (1 - (a.ae["model"]) / a.ae[ref]) if a.ae[ref] else float("nan")
            dcrps = 100 * (1 - (a.crps["model"]) / a.crps[ref]) if a.crps[ref] else float("nan")
            dfss = a.fss_at("model", FSS_SCALES_PX[-1]) - a.fss_at(ref, FSS_SCALES_PX[-1])
            print(f"  → vs {ref}: MAE skill {dmae:+.1f}%, CRPS skill {dcrps:+.1f}%, "
                  f"ΔFSS@{FSS_SCALES_PX[-1]} {dfss:+.3f}  (positive = model wins)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
