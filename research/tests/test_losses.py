"""Tests for the FSS / sharpness structure losses (WP 2.1)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from model.losses import CombinedLoss, fss_exceedance_loss, sharpness_loss, total_loss


def _rand_field(seed: int, shape=(2, 1, 32, 32)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) * 3.0  # mm/h-ish, in [0, 3)


# --------------------------------------------------------------------------
# fss_exceedance_loss
# --------------------------------------------------------------------------

def test_fss_loss_zero_for_identical_fields():
    x = _rand_field(0)
    loss = fss_exceedance_loss(x, x.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_fss_loss_increases_with_displacement():
    torch.manual_seed(1)
    target = torch.zeros(1, 1, 32, 32)
    # a compact wet blob
    target[:, :, 10:14, 10:14] = 3.0

    pred_close = target.clone()
    pred_close[:, :, 11:15, 11:15] = torch.maximum(
        pred_close[:, :, 11:15, 11:15], torch.tensor(3.0)
    )  # shifted by 1px, still overlapping heavily

    pred_far = torch.zeros_like(target)
    pred_far[:, :, 24:28, 24:28] = 3.0  # shifted far away, no overlap

    loss_close = fss_exceedance_loss(pred_close, target)
    loss_far = fss_exceedance_loss(pred_far, target)

    assert loss_close.item() < loss_far.item()
    assert loss_far.item() > 0.0


def test_fss_loss_thresholds_and_scales_are_configurable():
    x = _rand_field(2)
    y = _rand_field(3)
    loss_a = fss_exceedance_loss(x, y, thresholds=(1.0,), scales=(1,))
    loss_b = fss_exceedance_loss(x, y, thresholds=(0.5, 1.0, 2.0), scales=(1, 3, 5))
    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)


def test_fss_loss_all_dry_batch_is_finite_not_nan():
    # 0/0 in the FSS ratio if eps is added rather than clamped and everything
    # underflows (the fp16 failure mode) — must stay finite even in float32.
    pred = torch.zeros(4, 1, 16, 16)
    target = torch.zeros(4, 1, 16, 16)
    loss = fss_exceedance_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------
# sharpness_loss
# --------------------------------------------------------------------------

def _blur(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)


def test_sharpness_loss_zero_for_identical_fields():
    x = _rand_field(4)
    loss = sharpness_loss(x, x.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_sharper_prediction_scores_lower_sharpness_loss():
    torch.manual_seed(5)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 4.0  # a sharp-edged block -> real gradient energy

    sharp_pred = target.clone()  # matches target's sharp edges exactly
    blurred_pred = _blur(target.clone())

    loss_sharp = sharpness_loss(sharp_pred, target)
    loss_blurred = sharpness_loss(blurred_pred, target)

    assert loss_sharp.item() == pytest.approx(0.0, abs=1e-6)
    assert loss_sharp.item() < loss_blurred.item()


def test_sharpness_loss_is_one_sided_sharper_than_truth_not_penalised():
    torch.manual_seed(9)
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 2.0

    # a prediction with MORE gradient energy than the target (e.g. add noise
    # on top of an exact copy) must score 0, not a positive "too sharp"
    # penalty — sharper-than-truth is not an error for this term.
    noisy_sharper = target.clone() + torch.randn(target.shape) * 0.5
    assert sharpness_loss(noisy_sharper, target).item() == pytest.approx(0.0, abs=1e-6)


def test_sharpness_loss_zeroed_and_bounded_on_dry_target():
    # A flat/dry target has no structure to match — must be exactly 0, not a
    # blown-up ratio from dividing by a near-zero (but nonzero) denominator.
    target = torch.zeros(2, 1, 16, 16)
    pred = torch.rand(2, 1, 16, 16) * 5.0  # arbitrary, even very "sharp" noise
    loss = sharpness_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_sharpness_loss_is_bounded():
    torch.manual_seed(10)
    target = torch.rand(1, 1, 16, 16) * 5.0
    pred = torch.zeros_like(target)  # maximally blurred: zero gradient energy
    loss = sharpness_loss(pred, target)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 1.0


# --------------------------------------------------------------------------
# CombinedLoss
# --------------------------------------------------------------------------

def test_combined_loss_zero_weights_matches_existing_huber_path():
    torch.manual_seed(6)
    pred = torch.rand(3, 1, 16, 16, requires_grad=True) * 2.0
    target = torch.rand(3, 1, 16, 16) * 2.0

    combined = CombinedLoss(bias_penalty=0.5, fss_weight=0.0, sharpness_weight=0.0)
    got = combined(pred, target)
    want = total_loss(pred, target, bias_penalty=0.5)

    assert torch.allclose(got, want, atol=1e-6)


def test_combined_loss_exposes_component_terms_when_enabled():
    torch.manual_seed(7)
    pred = torch.rand(2, 1, 16, 16) * 2.0
    target = torch.rand(2, 1, 16, 16) * 2.0

    combined = CombinedLoss(bias_penalty=0.5, fss_weight=0.3, sharpness_weight=0.2)
    loss = combined(pred, target)

    assert "huber" in combined.last_terms
    assert "bias" in combined.last_terms
    assert "fss" in combined.last_terms
    assert "sharpness" in combined.last_terms
    assert combined.last_terms["total"] == pytest.approx(loss.item(), abs=1e-5)


def test_combined_loss_disabled_terms_absent_from_logging():
    pred = torch.rand(2, 1, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    combined = CombinedLoss()
    combined(pred, target)
    assert "fss" not in combined.last_terms
    assert "sharpness" not in combined.last_terms


# --------------------------------------------------------------------------
# gradient flow / numerical stability
# --------------------------------------------------------------------------

def test_gradients_flow_float32():
    pred = (torch.rand(2, 1, 16, 16) * 2.0).requires_grad_(True)
    target = torch.rand(2, 1, 16, 16) * 2.0
    combined = CombinedLoss(bias_penalty=0.5, fss_weight=0.3, sharpness_weight=0.2)
    loss = combined(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert not torch.isnan(loss)


def _assert_finite_loss_and_grad(pred: torch.Tensor, target: torch.Tensor) -> None:
    combined = CombinedLoss(bias_penalty=0.5, fss_weight=0.3, sharpness_weight=0.2)
    loss = combined(pred, target)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), f"non-finite grad: {pred.grad}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gradients_finite_under_low_precision_inputs(dtype):
    """Explicit low-precision (not CPU-autocast, which doesn't cover these
    ops and would silently run everything in fp32) inputs, matching the GPU
    training dtype. No skip: this must pass outright."""
    torch.manual_seed(11)
    pred = (torch.rand(3, 1, 16, 16, dtype=dtype) * 2.0).requires_grad_(True)
    target = torch.rand(3, 1, 16, 16, dtype=dtype) * 2.0
    _assert_finite_loss_and_grad(pred, target)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gradients_finite_all_dry_low_precision(dtype):
    """All-dry batch under fp16/bf16 — the measured 0/0-NaN failure mode."""
    pred = torch.zeros(4, 1, 16, 16, dtype=dtype, requires_grad=True)
    target = torch.zeros(4, 1, 16, 16, dtype=dtype)
    _assert_finite_loss_and_grad(pred, target)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gradients_finite_flat_prediction_low_precision(dtype):
    """Constant (zero-gradient) prediction against a structured target under
    fp16/bf16 — the measured singular-derivative-at-0 failure mode."""
    target = torch.zeros(1, 1, 16, 16, dtype=dtype)
    target[:, :, 4:12, 4:12] = 3.0
    pred = torch.full((1, 1, 16, 16), 1.5, dtype=dtype, requires_grad=True)
    _assert_finite_loss_and_grad(pred, target)
