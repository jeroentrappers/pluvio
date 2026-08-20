"""The honest near-term deliverable: the 0–2 h OPERA nowcast, head-to-head.

Recommendation #2 — scope what we *can* defend today. The multimodal/multi-day
thesis needs months of accumulated MTG/AIFS; the nowcast does not (22 months of
OPERA are already deep). So the first real, self-contained result is:

    does the model beat **optical-flow advection** (the operational-grade
    nowcast bar) on the 0–2 h OPERA truth — and if so, where (lead × scale ×
    intensity), driven by which signal?

This harness reports per-lead (10-min steps) MAE, CSI at τ=0.1/1, scale-aware
**FSS**, and **CRPS**, for the model vs **optical-flow** vs **persistence**, and
prints the lead at which the model overtakes optical-flow on each axis. That
crossover, if positive, is the claim — not "beats persistence".

    python -m model.eval_nowcast --zarr ./seamless.zarr --ckpt checkpoints/pluvio_seamless.pt
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
from model import metrics as M  # noqa: E402
from model.classical import optical_flow_nowcast  # noqa: E402
from model.geo import grid_resolution_km  # noqa: E402
from model.seamless import SeamlessNet  # noqa: E402
from model.seamless_dataset import SeamlessDataset, issue_time_split  # noqa: E402

NOWCAST_LEADS = tuple(range(0, 121, 10))
THRESHOLDS = (0.1, 1.0)
FSS_SCALES_PX = (1, 3, 5)
METHODS = ("model", "optical-flow", "persistence")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--aux-channels", default="",
                   help="match training: 'none' = radar-only ablation, empty = auto-discover")
    p.add_argument("--advection", action=argparse.BooleanOptionalAction, default=None,
                   help="match training: build inputs with the advection prior + tendency "
                        "channels so the channel count matches an --advection checkpoint. "
                        "Default: read from the checkpoint, else off.")
    p.add_argument("--static", action=argparse.BooleanOptionalAction, default=None,
                   help="include the static_* terrain channels. Default: read from the "
                        "checkpoint. Pass --no-static to evaluate a pre-static checkpoint "
                        "(e.g. c16) against a zarr that has since gained static channels.")
    args = p.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the checkpoint FIRST: since 2026-08 it records the full input recipe
    # (aux/static channels, history steps, advection), so the eval can rebuild the
    # exact assembly training used instead of re-deriving it and hoping it matches.
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cut = issue_time_split(args.zarr, args.val_frac)
    hi = datetime(2100, 1, 1, tzinfo=timezone.utc)
    sel = args.aux_channels.strip().lower()
    if sel:
        aux_channels = [] if sel == "none" else [s.strip() for s in args.aux_channels.split(",") if s.strip()]
    else:
        aux_channels = ck.get("aux_channels")  # None on a legacy ckpt → auto-discover
    use_advection = args.advection if args.advection is not None else bool(ck.get("advection", False))
    include_static = (args.static if args.static is not None
                      else bool(ck.get("static_channels")) if "static_channels" in ck else True)
    ds = SeamlessDataset(args.zarr, time_range=(cut, hi), leads_min=NOWCAST_LEADS,
                         aux_channels=aux_channels, use_advection=use_advection,
                         include_static=include_static)
    Hsteps, dt_min = ds.history_steps, ds.history_step_min

    # Fail loudly on a layout mismatch. load_state_dict would catch a channel-COUNT
    # change, but not a same-count-different-channels one — which verifies as noise.
    if ds.n_channels != ck["in_channels"]:
        raise SystemExit(
            f"channel mismatch: checkpoint expects {ck['in_channels']}, this zarr assembles "
            f"{ds.n_channels} (aux={ds.aux_channels}, static={ds.static_channels}, "
            f"history={ds.history_steps}, advection={use_advection}). "
            f"Checkpoint recipe: aux={ck.get('aux_channels')}, static={ck.get('static_channels')}, "
            f"history={ck.get('history_steps')}, advection={ck.get('advection')}. "
            f"Use --aux-channels/--advection/--no-static to match it.")
    quantiles = ck.get("quantiles")
    net = SeamlessNet(in_channels=ck["in_channels"], base_channels=ck["base_channels"],
                      quantiles=tuple(quantiles) if quantiles else None).to(dev).eval()
    net.load_state_dict(ck["model"])

    dy_km, dx_km = grid_resolution_km()
    print(f"NOWCAST deliverable | {args.ckpt} | grid ≈ {dy_km:.1f}×{dx_km:.1f} km | val={len(ds)} | dev={dev}")
    print(f"the bar is OPTICAL-FLOW, not persistence. FSS scales (px) → km: "
          f"{[f'{s}={s*dy_km:.0f}km' for s in FSS_SCALES_PX]}\n")

    # per (method, lead): running metric accumulators
    ae = collections.defaultdict(float)
    crps = collections.defaultdict(float)
    cat = collections.defaultdict(lambda: [0, 0, 0])      # (method,lead,τ) → h,m,fa
    fss = collections.defaultdict(lambda: [0.0, 0.0])      # (method,lead,scale) → num,den
    cnt = collections.defaultdict(int)

    random.seed(0)
    n = min(args.samples, len(ds))
    idxs = random.sample(range(len(ds)), n)
    leads = np.array([ds.index[i].lead_min for i in idxs])
    dl = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, num_workers=args.workers)
    ptr = 0
    with torch.no_grad():
        for x, cond, y in dl:
            bs = x.size(0)
            lead_b = leads[ptr:ptr + bs]; ptr += bs
            with torch.autocast(dev.type, enabled=(dev.type == "cuda")):
                out = net(x.to(dev), cond.to(dev)).float()
            point = (net.median(out) if quantiles else out).squeeze(1).cpu().numpy()
            qn = out.cpu().numpy() if quantiles else None
            xn, yn = x.numpy(), y.squeeze(1).numpy()
            for j in range(bs):
                lm = int(lead_b[j]); obs = yn[j]
                of, _ = optical_flow_nowcast(xn[j, :Hsteps], [lm], dt_min=dt_min)
                fields = {"model": point[j], "optical-flow": of[0], "persistence": xn[j, Hsteps - 1]}
                cnt[lm] += 1
                for m, pr in fields.items():
                    ae[(m, lm)] += float(np.mean(np.abs(pr - obs)))
                    if quantiles and m == "model":
                        crps[(m, lm)] += M.crps_from_quantiles(qn[j], obs, quantiles)
                    else:
                        crps[(m, lm)] += M.crps_deterministic(pr, obs)
                    for t in THRESHOLDS:
                        sc = M.categorical_scores(pr, obs, t); a = cat[(m, lm, t)]
                        a[0] += sc["hits"]; a[1] += sc["misses"]; a[2] += sc["false_alarms"]
                    for s in FSS_SCALES_PX:
                        pf = M._fraction_field(pr >= 0.1, s); ofr = M._fraction_field(obs >= 0.1, s)
                        f = fss[(m, lm, s)]
                        f[0] += float(np.mean((pf - ofr) ** 2)); f[1] += float(np.mean(pf**2)+np.mean(ofr**2))

    def csi(m, lm, t):
        h, mi, fa = cat[(m, lm, t)]; return h/(h+mi+fa) if (h+mi+fa) else float("nan")
    def fss_at(m, lm, s):
        num, den = fss[(m, lm, s)]; return 1-num/den if den else float("nan")

    print(f"{'lead':>5}{'method':>14}{'MAE':>8}{'CRPS':>8}{'CSI.1':>8}{'CSI1':>8}"
          + "".join(f"{'FSS'+str(s):>7}" for s in FSS_SCALES_PX))
    print("-" * 75)
    overtakes = {}
    for lm in NOWCAST_LEADS:
        if not cnt[lm]:
            continue
        for m in METHODS:
            c = cnt[lm]
            print(f"{lm:>5}{m:>14}{ae[(m,lm)]/c:>8.3f}{crps[(m,lm)]/c:>8.3f}"
                  f"{csi(m,lm,0.1):>8.3f}{csi(m,lm,1.0):>8.3f}"
                  + "".join(f"{fss_at(m,lm,s):>7.3f}" for s in FSS_SCALES_PX))
        # record where the model first beats optical-flow on CSI@0.1
        if lm not in overtakes and csi("model", lm, 0.1) > csi("optical-flow", lm, 0.1):
            overtakes[lm] = True
        print()

    beats = [lm for lm in NOWCAST_LEADS if cnt[lm]
             and csi("model", lm, 0.1) > csi("optical-flow", lm, 0.1)]
    print("HEADLINE (the honest claim):")
    if beats:
        print(f"  model beats optical-flow on CSI@0.1 at leads {beats} min "
              f"(of {[l for l in NOWCAST_LEADS if cnt[l]]}).")
    else:
        print("  model does NOT yet beat optical-flow on CSI@0.1 at any nowcast lead — "
              "no nowcast claim is defensible yet (this is the honest state).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
