"""Structure-aware loss terms for the correction UNet (WP 2.1).

The deterministic Huber objective (``weighted_huber`` / ``total_loss``,
moved here from ``train.py`` so there is a single source of truth) is
minimised by the conditional mean, which under positional uncertainty
rewards a blurred field — it wins pixelwise error but loses cell detection
and Fractions Skill Score (FSS) to a naive optical-flow advection baseline.
This module adds two differentiable structure terms that can be blended in:

* ``fss_exceedance_loss`` — a soft, differentiable Fractions Skill Score
  computed on exceedance (wet-mask) indicators, ported from the
  ``mode="exceedance"`` branch of ``train_seamless.multiscale_loss``. Unlike
  that helper (which returns a raw pooled-MSE structure term), this returns
  ``1 - FSS`` using the actual FSS ratio
  ``mean((pf-tf)^2) / (mean(pf^2) + mean(tf^2))`` per sample, so the value is
  bounded in ``[0, 1]`` and directly comparable across thresholds/scales,
  matching the eval's per-case FSS metric. All pooling/ratio arithmetic runs
  in float32 regardless of the input dtype (GPU training runs fp16 autocast,
  where the raw sigmoid/mse/eps terms underflow to 0 and produce ``0/0`` on
  all-dry batches) — see ``fss_exceedance_loss`` for the specifics.
* ``sharpness_loss`` — a one-sided hinge on the *deficit* of gradient energy
  relative to the target: penalises a prediction that is blurrier than the
  target, but never rewards one that is sharper (see its docstring for the
  residual noise-gaming risk this leaves open).

``CombinedLoss`` wraps ``total_loss`` (the existing weighted-Huber +
bias-penalty objective) together with these two terms, each gated by its own
weight. With ``fss_weight == 0`` and ``sharpness_weight == 0`` (the
defaults), ``CombinedLoss`` computes exactly ``total_loss(pred, target,
bias_penalty)`` — the new terms are skipped, not merely multiplied by zero,
so a live run using the old defaults is unaffected bit-for-bit and pays no
extra device-sync cost for term logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_SCALES: tuple[int, ...] = (1, 3, 5)
DEFAULT_TAU: float = 0.05


def weighted_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss weighted by ``(1 + obs)`` so heavy rain matters.

    Softened from the original ``(1 + obs)²``: the squared weight made the
    optimizer hedge precipitation upward everywhere, producing a persistent
    wet bias. Linear weighting keeps the heavy-rain emphasis without the
    systematic over-prediction.
    """
    delta = 1.0
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    lin = abs_diff - quad
    per_pixel = 0.5 * quad**2 + delta * lin
    weight = 1.0 + target
    return (per_pixel * weight).mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor, bias_penalty: float) -> torch.Tensor:
    """Weighted Huber + a penalty on the systematic (batch-mean) bias.

    The bias term directly punishes ``mean(pred) - mean(target)``, which is
    the exact quantity we saw drift to +0.14 mm/h. Keeps the model honest
    about *how much* rain, not just *where*.
    """
    base = weighted_huber(pred, target)
    bias = (pred.mean() - target.mean()).pow(2)
    return base + bias_penalty * bias


def fss_exceedance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    scales: tuple[int, ...] = DEFAULT_SCALES,
    tau: float = DEFAULT_TAU,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Fractions Skill Score loss (``1 - FSS``, averaged).

    For each rain-rate threshold, a soft exceedance ("is it raining at least
    this hard here") mask is built for both fields via
    ``sigmoid((x - threshold) / tau)`` — the same soft-indicator trick used in
    ``train_seamless.multiscale_loss(mode="exceedance")``. At each
    neighbourhood scale ``k`` the indicator is average-pooled (stride 1, same
    spatial size) to get the local wet *fraction*, and the FSS is computed
    **per sample** as the standard ratio

        FSS = 1 - mean_{H,W}((pf - tf)^2) / (mean_{H,W}(pf^2) + mean_{H,W}(tf^2))

    then averaged over the batch and over every (threshold, scale) pair — a
    single batch-wide ratio would let a mix of easy/hard samples cancel out
    rather than average their per-case skill, which is what the eval's FSS
    reports. The loss is ``1 - FSS``: 0 when the fields agree exactly,
    increasing as the wet/dry boundary is misplaced.

    All of the above runs in **float32**, independent of ``pred``/``target``'s
    incoming dtype: under fp16 (the GPU autocast dtype) the raw sigmoid
    outputs, their squares and ``eps`` itself can all underflow toward 0,
    turning an all-dry batch's ``0 / 0`` into NaN. Casting to float32 before
    the sigmoid/pooling and clamping (not adding) ``eps`` into the
    denominator keeps the ratio finite in every case that's been measured to
    break under raw fp16, including all-dry batches.

    ``pred``/``target`` are ``(B, 1, H, W)`` (or any shape ``avg_pool2d``
    accepts, with dim 0 treated as the batch to average the per-sample FSS
    over).
    """
    pred32 = pred.float()
    target32 = target.float()
    reduce_dims = tuple(range(1, pred32.dim()))

    terms = []
    for thr in thresholds:
        pf_src = torch.sigmoid((pred32 - thr) / tau)
        tf_src = torch.sigmoid((target32 - thr) / tau)
        for k in scales:
            if k <= 1:
                pf, tf = pf_src, tf_src
            else:
                pad = k // 2
                pf = F.avg_pool2d(pf_src, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
                tf = F.avg_pool2d(tf_src, kernel_size=k, stride=1, padding=pad, count_include_pad=False)
            num = ((pf - tf) ** 2).mean(dim=reduce_dims)
            den = (pf**2).mean(dim=reduce_dims) + (tf**2).mean(dim=reduce_dims)
            fss = 1.0 - num / den.clamp_min(eps)
            terms.append((1.0 - fss).mean())
    return torch.stack(terms).mean()


def _grad_energy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean magnitude of the finite-difference spatial gradient of ``x``.

    Uses the joint L2 magnitude ``sqrt(dx^2 + dy^2 + eps)`` rather than
    ``|dx| + |dy|``: the L1 form is *invariant* to monotone blurring along
    any straight transect (a box-blurred step edge has the same total
    variation as the sharp step, since ``|dx|`` summed along a monotone ramp
    always equals the jump it came from — measured on a 32x32 canvas holding
    an interior 8x8 sharp block at value 8.0 on a 0.0 background vs. the
    same canvas after a 5x5-average-pooled (``count_include_pad=False``)
    blur: identical L1 mean, 0.2664 both), so it fails to distinguish blur
    from sharpness — exactly the thing this loss exists to measure. The
    joint L2 form does discriminate (0.262 sharp vs 0.229 blurred, same
    32x32/8x8/8.0 canvas, ``eps=1e-6``) because combining ``dx``/``dy``
    through ``sqrt`` is concave, so smoothing a 2-D edge (which redistributes
    gradient between the two axes near corners, not just along one) lowers
    the mean magnitude.

    The singular derivative of plain ``sqrt(dx^2+dy^2)`` at 0 (``d/dx sqrt(x)
    -> inf`` as ``x -> 0``) is what caused NaN gradients on the flat/dry
    regions that dominate this ~95%-dry domain when ``eps`` itself underflowed
    under fp16. Two changes fix that without giving up L2's blur-sensitivity:
    computing in **float32** (so ``eps`` can't underflow) and sizing ``eps``
    at ``1e-6`` (large enough, in float32, to keep the derivative
    ``dx/sqrt(dx^2+dy^2+eps)`` bounded at ``dx=dy=0``, rather than the
    ``1e-8`` used previously which is still representable in float32 but
    small enough for the ratio's gradient to blow up numerically at 0).

    Cropped by one pixel on each trailing spatial axis so the horizontal and
    vertical differences broadcast to the same shape before combining.
    """
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = dx[..., :-1, :]
    dy = dy[..., :, :-1]
    mag = torch.sqrt(dx**2 + dy**2 + eps)
    return mag.mean()


def sharpness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dry_floor: float = 1e-2,
) -> torch.Tensor:
    """One-sided hinge on the gradient-energy *deficit* vs. the target.

    Computed as ``relu(target_e - pred_e) / target_e.clamp_min(dry_floor)``
    where ``target_e``/``pred_e`` are the mean gradient magnitude
    (``_grad_energy``) of each field. A prediction blurrier than the
    target (``pred_e < target_e``) is penalised in proportion to the
    fractional shortfall; a prediction that is *as sharp or sharper* than
    the target scores exactly 0 — sharper-than-truth must not be penalised,
    so this is one-sided (``relu``), not the symmetric ``|pred_e - target_e|``
    of an earlier version.

    Runs in float32 (irrespective of input dtype) for the same underflow
    reasons as ``fss_exceedance_loss``. Two guards keep the term bounded and
    well-posed on this domain's mostly-dry fields:

    * ``target_e`` is floored at ``dry_floor`` before dividing, and the term
      is zeroed outright when the target is at/under that floor (i.e.
      effectively flat/dry) rather than computed against a near-zero
      denominator — a target with no real structure has no sharpness to
      match, and dividing by a merely-floored ~0 would still manufacture a
      large, meaningless gradient signal. The dry/not-dry selection is a
      ``torch.where`` tensor mask, not a Python ``if`` on the (0-dim, but
      still device-resident) ``target_e`` tensor — the latter forces a
      host sync (``.item()``/``bool()``) every call, which is disallowed on
      any path that has to stay under CUDA graph capture.
    * the result is clamped to ``[0, 1]`` so one badly-blurred sample can't
      dominate a batch (measured: an unclamped, unfloored, two-sided version
      of this term reached ~5e3 against a Huber term of ~0.17).

    Residual gaming risk: the hinge means i.i.d. noise added to a blurred
    prediction can raise ``pred_e`` back up to (or past) ``target_e`` and
    drive this term to 0 without reconstructing the target's actual
    structure — it makes noise *free* once parity is reached, not
    *rewarded*. It does not by itself forbid noise; the Huber term (which
    rises with unstructured noise) is what has to keep that in check when
    both are combined.
    """
    pred_e = _grad_energy(pred.float())
    target_e = _grad_energy(target.float())
    deficit = torch.relu(target_e - pred_e)
    result = (deficit / target_e.clamp_min(dry_floor)).clamp(max=1.0)
    is_dry = target_e <= dry_floor
    return torch.where(is_dry, pred_e.new_zeros(()), result)


class CombinedLoss(torch.nn.Module):
    """``total_loss`` (weighted Huber + bias penalty) with optional FSS and
    sharpness terms, each gated by its own weight.

    With ``fss_weight == 0`` and ``sharpness_weight == 0`` (the defaults),
    ``forward`` returns exactly ``total_loss(pred, target, bias_penalty)`` —
    the FSS/sharpness terms are skipped entirely (not computed-then-zeroed)
    so there is no risk of a NaN/inf in an unused term leaking into the
    total, and no extra per-batch device sync from logging unused terms.
    ``bias_penalty`` is the same weight ``total_loss`` takes; there is no
    separate configurable Huber function.
    """

    def __init__(
        self,
        bias_penalty: float = 0.5,
        fss_weight: float = 0.0,
        fss_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        fss_scales: tuple[int, ...] = DEFAULT_SCALES,
        fss_tau: float = DEFAULT_TAU,
        sharpness_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bias_penalty = bias_penalty
        self.fss_weight = fss_weight
        self.fss_thresholds = tuple(fss_thresholds)
        self.fss_scales = tuple(fss_scales)
        self.fss_tau = fss_tau
        self.sharpness_weight = sharpness_weight
        self.last_terms: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = total_loss(pred, target, self.bias_penalty)

        if self.fss_weight <= 0 and self.sharpness_weight <= 0:
            # Default (live-run) path: no extra terms, no logging syncs.
            self.last_terms = {}
            return total

        # Only recomputed on the (non-default) path where extra terms are
        # enabled, purely so the components can be logged individually.
        huber = weighted_huber(pred, target)
        bias = (pred.mean() - target.mean()).pow(2)
        terms = {"huber": float(huber.detach().item()), "bias": float(bias.detach().item())}

        if self.fss_weight > 0:
            fss = fss_exceedance_loss(
                pred, target, self.fss_thresholds, self.fss_scales, self.fss_tau
            )
            total = total + self.fss_weight * fss
            terms["fss"] = float(fss.detach().item())

        if self.sharpness_weight > 0:
            sharp = sharpness_loss(pred, target)
            total = total + self.sharpness_weight * sharp
            terms["sharpness"] = float(sharp.detach().item())

        terms["total"] = float(total.detach().item())
        self.last_terms = terms
        return total
