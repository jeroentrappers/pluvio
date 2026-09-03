"""Structure-aware loss terms for the correction UNet (WP 2.1).

The deterministic Huber objective (see ``train.total_loss``) is minimised by
the conditional mean, which under positional uncertainty rewards a blurred
field — it wins pixelwise error but loses sharpness and Fractions Skill Score
(FSS) to a naive optical-flow advection baseline. This module adds two
differentiable structure terms that can be blended in:

* ``fss_exceedance_loss`` — a soft, differentiable Fractions Skill Score
  computed on exceedance (wet-mask) indicators, ported from the
  ``mode="exceedance"`` branch of ``train_seamless.multiscale_loss``. Unlike
  that helper (which returns a raw pooled-MSE structure term), this returns
  ``1 - FSS`` using the actual FSS ratio ``MSE / (mean(pf^2) + mean(tf^2))``
  so the value is bounded in ``[0, 1]`` and directly comparable across
  thresholds/scales, matching the eval's FSS metric.
* ``sharpness_loss`` — penalises the loss of gradient energy that comes with
  blurring: the relative difference between the mean image-gradient
  magnitude of the prediction and of the target.

``CombinedLoss`` wraps the existing weighted-Huber + bias-penalty objective
(``train.weighted_huber`` / ``train.total_loss``) together with these two
terms, each gated by its own weight. With ``fss_weight == 0`` and
``sharpness_weight == 0`` (the defaults), ``CombinedLoss`` computes exactly
``train.total_loss`` — the new terms are skipped, not merely multiplied by
zero, so a live run using the old defaults is unaffected bit-for-bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_SCALES: tuple[int, ...] = (1, 3, 5)


def fss_exceedance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    scales: tuple[int, ...] = DEFAULT_SCALES,
    tau: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable Fractions Skill Score loss (``1 - FSS``, averaged).

    For each rain-rate threshold, a soft exceedance ("is it raining at least
    this hard here") mask is built for both fields via
    ``sigmoid((x - threshold) / tau)`` — the same soft-indicator trick used in
    ``train_seamless.multiscale_loss(mode="exceedance")``. At each
    neighbourhood scale ``k`` the indicator is average-pooled (stride 1, same
    spatial size) to get the local wet *fraction*, and the FSS is the
    standard

        FSS = 1 - mean((pf - tf)^2) / (mean(pf^2) + mean(tf^2))

    The loss is ``1 - FSS`` averaged over every (threshold, scale) pair, so it
    is 0 when the fields agree exactly and increases as the wet/dry boundary
    is misplaced. ``pred``/``target`` are ``(B, 1, H, W)`` (or any shape
    ``avg_pool2d`` accepts).
    """
    terms = []
    for thr in thresholds:
        pf_src = torch.sigmoid((pred - thr) / tau)
        tf_src = torch.sigmoid((target - thr) / tau)
        for k in scales:
            if k <= 1:
                pf, tf = pf_src, tf_src
            else:
                pad = k // 2
                pf = F.avg_pool2d(pf_src, kernel_size=k, stride=1, padding=pad,
                                   count_include_pad=False)
                tf = F.avg_pool2d(tf_src, kernel_size=k, stride=1, padding=pad,
                                   count_include_pad=False)
            num = F.mse_loss(pf, tf)
            den = (pf**2).mean() + (tf**2).mean()
            fss = 1.0 - num / (den + eps)
            terms.append(1.0 - fss)
    return torch.stack(terms).mean()


def _grad_energy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean magnitude of the finite-difference spatial gradient of ``x``.

    Cropped by one pixel on each trailing spatial axis so the horizontal and
    vertical differences broadcast to the same shape before combining.
    """
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = dx[..., :-1, :]
    dy = dy[..., :, :-1]
    mag = torch.sqrt(dx**2 + dy**2 + eps)
    return mag.mean()


def sharpness_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative loss of gradient energy vs. the target (a proxy for blur).

    Computed as ``|E[|grad(pred)|] - E[|grad(target)|]| / (E[|grad(target)|] + eps)``.
    A prediction that is blurrier than the target has lower gradient energy,
    so this is > 0; it is 0 when the two fields have matching gradient
    energy. (A spectral high-frequency-power variant would work equally
    well — this finite-difference form is picked for its low cost and
    simplicity to test on CPU.)
    """
    pred_e = _grad_energy(pred, eps)
    target_e = _grad_energy(target, eps)
    return (pred_e - target_e).abs() / (target_e + eps)


class CombinedLoss(torch.nn.Module):
    """Weighted Huber (+ bias penalty) with optional FSS and sharpness terms.

    ``weighted_huber_fn`` and ``bias penalty`` mirror ``train.weighted_huber``
    / ``train.total_loss`` exactly, so that with ``fss_weight == 0`` and
    ``sharpness_weight == 0`` the value returned by ``forward`` is identical
    to the previous ``train.total_loss(pred, target, bias_penalty)`` — the
    FSS/sharpness terms are skipped entirely (not computed-then-zeroed) so
    there is no risk of a NaN/inf in an unused term leaking into the total.
    """

    def __init__(
        self,
        bias_penalty: float = 0.5,
        fss_weight: float = 0.0,
        fss_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        fss_scales: tuple[int, ...] = DEFAULT_SCALES,
        fss_tau: float = 0.05,
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

    @staticmethod
    def weighted_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = 1.0
        diff = pred - target
        abs_diff = diff.abs()
        quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
        lin = abs_diff - quad
        per_pixel = 0.5 * quad**2 + delta * lin
        weight = 1.0 + target
        return (per_pixel * weight).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = self.weighted_huber(pred, target)
        bias = (pred.mean() - target.mean()).pow(2)
        total = huber + self.bias_penalty * bias

        terms = {"huber": float(huber.detach().cpu()), "bias": float(bias.detach().cpu())}

        if self.fss_weight > 0:
            fss = fss_exceedance_loss(
                pred, target, self.fss_thresholds, self.fss_scales, self.fss_tau
            )
            total = total + self.fss_weight * fss
            terms["fss"] = float(fss.detach().cpu())

        if self.sharpness_weight > 0:
            sharp = sharpness_loss(pred, target)
            total = total + self.sharpness_weight * sharp
            terms["sharpness"] = float(sharp.detach().cpu())

        terms["total"] = float(total.detach().cpu())
        self.last_terms = terms
        return total
