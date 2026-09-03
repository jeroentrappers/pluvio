"""Tests for the FSS / sharpness structure losses (WP 2.1)."""

from __future__ import annotations

import pytest
import torch

from model.losses import CombinedLoss, fss_exceedance_loss, sharpness_loss
from model.train import total_loss


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


# --------------------------------------------------------------------------
# sharpness_loss
# --------------------------------------------------------------------------

def _blur(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)


def test_sharpness_loss_zero_for_identical_fields():
    x = _rand_field(4)
    loss = sharpness_loss(x, x.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_sharper_prediction_scores_lower_sharpness_loss():
    torch.manual_seed(5)
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 4.0  # a sharp-edged block -> real gradient energy

    sharp_pred = target.clone()  # essentially matches target's sharp edges
    blurred_pred = _blur(target.clone())

    loss_sharp = sharpness_loss(sharp_pred, target)
    loss_blurred = sharpness_loss(blurred_pred, target)

    assert loss_sharp.item() < loss_blurred.item()


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gradients_flow_under_cpu_autocast(dtype):
    try:
        with torch.autocast(device_type="cpu", dtype=dtype):
            pred = (torch.rand(2, 1, 16, 16) * 2.0).requires_grad_(True)
            target = torch.rand(2, 1, 16, 16) * 2.0
            combined = CombinedLoss(bias_penalty=0.5, fss_weight=0.3, sharpness_weight=0.2)
            loss = combined(pred, target)
    except RuntimeError as exc:  # pragma: no cover
        pytest.skip(f"autocast dtype {dtype} unsupported on this CPU build: {exc}")
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert not torch.isnan(loss)
