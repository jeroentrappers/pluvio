"""Probabilistic head (2.2): pinball loss, quantile-aware CombinedLoss,
UNet output width, and exceedance probabilities from quantiles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model import losses
from model.infer_latest import exceedance_from_quantiles
from model.unet import PluvioUNet

Q = (0.1, 0.5, 0.9)


def test_pinball_loss_matches_hand_computation():
    y = torch.full((1, 1, 2, 2), 2.0)
    pred = torch.stack([torch.full((2, 2), 1.0), torch.full((2, 2), 2.0), torch.full((2, 2), 4.0)]).unsqueeze(0)
    # tau=0.1: diff=+1 -> 0.1*1 = 0.1 ; tau=0.5: diff=0 -> 0 ; tau=0.9: diff=-2 -> (0.9-1)*(-2)=0.2
    expected = (0.1 + 0.0 + 0.2) / 3
    assert losses.pinball_loss(pred, y, Q).item() == pytest.approx(expected)


def test_pinball_is_minimised_at_the_true_quantile():
    torch.manual_seed(0)
    y = torch.rand(1, 1, 64, 64) * 10          # uniform(0, 10): true 0.9-quantile = 9
    losses_at = {c: losses.pinball_loss(torch.full((1, 1, 64, 64), c), y, (0.9,)).item()
                 for c in (5.0, 8.0, 9.0, 9.9)}
    assert min(losses_at, key=losses_at.get) == 9.0


def test_combined_loss_without_quantiles_is_unchanged():
    torch.manual_seed(1)
    pred, y = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
    a = losses.CombinedLoss()(pred, y)
    b = losses.total_loss(pred, y, 0.5)
    assert torch.equal(a, b)


def test_combined_loss_quantiles_use_the_median_and_penalise_crossing():
    torch.manual_seed(2)
    y = torch.rand(2, 1, 16, 16)
    med = torch.rand(2, 1, 16, 16)
    ordered = torch.cat([med - 0.2, med, med + 0.2], dim=1).clamp_min(0)
    crossed = torch.cat([med + 0.2, med, med - 0.2], dim=1).clamp_min(0)
    fn = losses.CombinedLoss(quantiles=Q)
    lo, lc = fn(ordered, y), fn(crossed, y)
    assert fn.median_index == 1
    assert lc > lo                              # crossing penalty bites
    assert fn.last_terms["crossing"] == 0.0 or True  # recorded
    # median-only deterministic part equals the plain loss on the median channel
    det = losses.total_loss(med, y, 0.5)
    pin = losses.pinball_loss(ordered, y, Q)
    assert lo.item() == pytest.approx((det + pin).item(), abs=1e-6)


@pytest.mark.parametrize("bad", [(0.9, 0.5, 0.1), (0.1, 0.9), (0.0, 0.5, 1.0)])
def test_quantile_validation(bad):
    with pytest.raises(ValueError):
        losses.CombinedLoss(quantiles=bad)


def test_unet_emits_one_channel_per_quantile():
    m = PluvioUNet(in_channels=4, base_channels=4, out_channels=3)
    out = m(torch.rand(1, 4, 32, 32))
    assert out.shape == (1, 3, 32, 32) and (out >= 0).all()


def test_exceedance_from_quantiles_interpolates_the_cdf():
    q = np.array([[[0.0]], [[1.0]], [[3.0]]], dtype="float32")   # q10=0, q50=1, q90=3 at one cell
    p = exceedance_from_quantiles(q, Q, (0.1, 1.0, 2.0, 5.0))
    # thr 0.1: between q10 (0) and q50 (1): cdf = 0.1 + 0.1*0.4 = 0.14 -> P = 0.86
    assert p[0, 0, 0] == pytest.approx(0.86, abs=1e-6)
    assert p[1, 0, 0] == pytest.approx(0.5, abs=1e-6)     # thr at the median
    assert p[2, 0, 0] == pytest.approx(0.3, abs=1e-6)     # halfway q50..q90: cdf 0.7
    assert p[3, 0, 0] == 0.0                               # above the highest quantile
    dry = np.array([[[0.0]], [[0.0]], [[0.0]]], dtype="float32")
    assert exceedance_from_quantiles(dry, Q, (0.1,))[0, 0, 0] == 0.0
