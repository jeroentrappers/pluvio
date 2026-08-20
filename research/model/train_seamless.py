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
from model.geo import GRID  # noqa: E402
from model.seamless import SeamlessNet, num_params  # noqa: E402
from model.seamless_dataset import (  # noqa: E402
    DEFAULT_LEADS, RAIN_THRESHOLD, SeamlessDataset, issue_time_split)

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


def multiscale_loss(pred: torch.Tensor, target: torch.Tensor,
                    scales: tuple[int, ...] = (3, 5, 11), mode: str = "rate",
                    threshold: float = RAIN_THRESHOLD, tau: float = 0.05) -> torch.Tensor:
    """Neighbourhood-pooled structure term, aligned with the eval's scale-aware FSS.

    A per-pixel loss (``precip_loss``) is minimised by the conditional *mean*, so under
    positional uncertainty it hedges → a blurred field that wins MAE but loses FSS to
    optical-flow advection. This term compares **average-pooled** (stride-1, same-size)
    log1p-rate fields at several neighbourhood widths, so getting the rain *fraction* right
    over 3/9/15 km blobs is rewarded even when the exact pixel is off — i.e. it pays for
    sharpness/placement, not smoothing. Added to ``precip_loss`` with a small weight so the
    point accuracy the model already wins isn't sacrificed. Scales mirror eval FSS px (1,3,5),
    plus a wider one for the meso field. Differentiable and cheap (avg_pool2d).

    ``mode`` selects WHAT gets pooled, and it matters more than the weight did:

    * ``"rate"`` (c16) pools log1p **intensity**. Measured effect on the metric it
      was meant to fix: c16_full vs c16_nofss moved FSS@3km by one lead and
      FSS@9km/@15km by nothing at all (0/12 → 0/12 vs pysteps).
    * ``"exceedance"`` pools a soft **wet-mask indicator** instead, which is what
      the eval's FSS actually scores (``pr >= 0.1``, eval_nowcast.py). Matching
      pooled mean rate over a 9 km box is not the same objective as matching the
      pooled wet fraction — a field can agree on the former while misplacing the
      wet/dry boundary the latter measures. ``tau`` sets the softness in mm/h —
      small enough to approximate a step, large enough that gradients don't vanish
      for pixels far from the threshold.

      The **same** soft indicator is applied to prediction and target. Using a soft
      prediction against a hard target looks closer to the metric but is not a
      divergence: every dry pixel then sits at sigma(-threshold/tau) ≈ 0.12 against
      a target of 0, so the loss has an irreducible floor (measured 0.0139 on
      identical fields) and spends gradient pushing an already-clamped Softplus
      output further down. Soft-on-both is zero iff the fields agree, and still
      compares pooled wet fractions.
    """
    if mode == "exceedance":
        pf_src = torch.sigmoid((pred - threshold) / tau)
        tf_src = torch.sigmoid((target - threshold) / tau)
    elif mode == "rate":
        pf_src, tf_src = torch.log1p(pred), torch.log1p(target)
    else:
        raise ValueError(f"unknown multiscale_loss mode {mode!r}")
    terms = []
    for k in scales:
        pad = k // 2
        pf = F.avg_pool2d(pf_src, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        tf = F.avg_pool2d(tf_src, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
        terms.append(F.mse_loss(pf, tf))
    return torch.stack(terms).mean()


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


def _augment(x: torch.Tensor, y: torch.Tensor):
    """Random dihedral transform (h/v flips + k·90° rotation) applied *identically*
    to inputs and target. Precip-intensity fields are ~isotropic, so this 8× expands
    the effective training set and directly fights the fast overfit we measured
    (val bottomed at epoch 3 with no regularisation). The lead/time `cond` vector is
    non-spatial and left untouched. ⚠️ Only safe while inputs are direction-agnostic
    (radar intensity, lightning); if wind components (u10/v10) are added as channels,
    rotation would desynchronise them from the field — gate augmentation then."""
    if torch.rand(()) < 0.5:
        x, y = torch.flip(x, dims=[-1]), torch.flip(y, dims=[-1])
    if torch.rand(()) < 0.5:
        x, y = torch.flip(x, dims=[-2]), torch.flip(y, dims=[-2])
    if x.shape[-1] == x.shape[-2]:
        k = int(torch.randint(0, 4, ()).item())
        if k:
            x, y = torch.rot90(x, k, dims=[-2, -1]), torch.rot90(y, k, dims=[-2, -1])
    return x, y


def run_epoch(model, loader, optim, device, scaler, train: bool, loss_fn=None, augment=False):
    loss_fn = loss_fn or precip_loss
    model.train(train)
    total, n = 0.0, 0
    for x, cond, y in loader:
        x, cond, y = x.to(device), cond.to(device), y.to(device)
        if train and augment:
            x, y = _augment(x, y)
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
    p.add_argument("--leads", default="",
                   help="comma lead-mins to train on; empty = the full seamless set. "
                        "Restrict to '0,10,...,120' for a nowcast-only run.")
    p.add_argument("--advection", action="store_true",
                   help="feed the optical-flow advection prior + growth/decay tendency as input "
                        "channels (needs tools/add_nowcast_channels.py run on the zarr first)")
    p.add_argument("--fss-mode", choices=["rate", "exceedance"], default="rate",
                   help="what the multi-scale term pools: 'rate' = log1p intensity (c16; moved "
                        "FSS@9/15km by nothing), 'exceedance' = soft wet-mask, which is what the "
                        "eval's FSS actually scores. See multiscale_loss.")
    p.add_argument("--fss-weight", type=float, default=0.0,
                   help="weight on the multi-scale (FSS-aligned) structure loss added to the "
                        "per-pixel Huber; 0 disables. ~0.3 sharpens without wrecking MAE.")
    p.add_argument("--aux-at-valid-time", action="store_true",
                   help="Stage B: read the ERA5/aux anchor at the valid-time (issue+lead) "
                        "as a perfect-forecast proxy the outlook head downscales")
    p.add_argument("--aux-channels", default="",
                   help="comma aux-channel names to feed the model; 'none' = radar-only "
                        "ablation (ignore lightning/GII even if present in the zarr); "
                        "empty = auto-discover all aux arrays in the store")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW L2 regularisation (e.g. 1e-4); 0 keeps plain Adam behaviour")
    p.add_argument("--augment", action="store_true",
                   help="random dihedral (flip/rotate) augmentation — fights overfit on isotropic fields")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs with no val improvement; 0 disables")
    p.add_argument("--cosine", action="store_true", help="cosine-anneal the LR over --epochs")
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
    # aux-channel selection: '' → auto-discover; 'none' → radar-only ablation ([]);
    # else an explicit comma list. Lets mm vs radar-only share ONE zarr (controlled).
    sel = args.aux_channels.strip().lower()
    aux_channels = [] if sel == "none" else (
        None if not sel else [s.strip() for s in args.aux_channels.split(",") if s.strip()])
    leads = [int(x) for x in args.leads.split(",") if x.strip()] or DEFAULT_LEADS
    ds_kw = dict(history_steps=args.history_steps, aux_at_valid_time=args.aux_at_valid_time,
                 aux_channels=aux_channels, leads_min=leads, use_advection=args.advection)
    # NB: require_rain_fraction left at None (not 0.0). A 0.0 threshold filters nothing yet
    # forces _build_index to read every candidate's truth frame — ~405k disk reads, ~2 h once
    # opera_rate is evicted from page cache. None yields the identical index in seconds.
    train_ds = SeamlessDataset(args.zarr, time_range=(lo, cut), **ds_kw)
    val_ds = SeamlessDataset(args.zarr, time_range=(cut, hi), **ds_kw)
    LOG.info("train=%d val=%d | %d channels | device=%s", len(train_ds), len(val_ds), train_ds.n_channels, device)

    quantiles = tuple(float(q) for q in args.quantiles.split(",") if q.strip()) or None
    model = SeamlessNet(in_channels=train_ds.n_channels, base_channels=args.base_channels,
                        quantiles=quantiles).to(device)
    if quantiles:
        LOG.info("probabilistic outlook head: quantiles=%s (CRPS-consistent pinball loss)", quantiles)
        loss_fn = lambda pred, y: quantile_loss(pred, y, quantiles)  # noqa: E731
    elif args.fss_weight > 0:
        fw = args.fss_weight
        fm = args.fss_mode
        loss_fn = lambda pred, y: precip_loss(pred, y) + fw * multiscale_loss(pred, y, mode=fm)  # noqa: E731
        LOG.info("loss = precip_loss + %.3g * multiscale_loss(mode=%s) (FSS-aligned structure term)",
                 fw, fm)
    else:
        loss_fn = precip_loss
    LOG.info("SeamlessNet params: %s", f"{num_params(model):,}")
    if args.weight_decay > 0:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs) if args.cosine else None
    LOG.info("optim=%s wd=%g augment=%s patience=%d cosine=%s",
             type(optim).__name__, args.weight_decay, args.augment, args.patience, bool(sched))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    best = float("inf")
    since_improve = 0
    start = time.time()
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        tr = run_epoch(model, tl, optim, device, scaler, train=True, loss_fn=loss_fn, augment=args.augment)
        va = run_epoch(model, vl, optim, device, scaler, train=False, loss_fn=loss_fn)
        if sched:
            sched.step()
        LOG.info("epoch %d: train=%.4f val=%.4f%s", epoch, tr, va,
                 f" lr={sched.get_last_lr()[0]:.2e}" if sched else "")
        if va < best:
            best = va; since_improve = 0
            # Record the full input recipe, not just the channel COUNT. Without this
            # the layout is unreproducible from the artifact: eval/serving rebuild
            # the dataset with auto-discovered aux + static, so adding any channel to
            # the zarr (or forgetting --advection) silently changes the assembly — and
            # a same-count-different-order mismatch trains and verifies as nonsense.
            torch.save({"model": model.state_dict(), "in_channels": train_ds.n_channels,
                        "base_channels": args.base_channels, "val_loss": best,
                        "quantiles": list(quantiles) if quantiles else None,
                        "aux_channels": list(train_ds.aux_channels),
                        "static_channels": list(train_ds.static_channels),
                        "history_steps": train_ds.history_steps,
                        "has_aifs": bool(train_ds.has_aifs),
                        "advection": bool(args.advection),
                        "leads_min": list(leads),
                        "grid": list(GRID),
                        "fss_weight": float(args.fss_weight),
                        "fss_mode": args.fss_mode}, out)
        else:
            since_improve += 1
            if args.patience and since_improve >= args.patience:
                LOG.info("early stop: no val improvement in %d epochs (best=%.4f)", args.patience, best)
                break
        if args.max_minutes and (time.time() - start) / 60 > args.max_minutes:
            LOG.info("hit --max-minutes; stopping"); break
    LOG.info("done. best val=%.4f → %s", best, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
