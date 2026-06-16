"""Seamless, lead-conditioned multimodal precipitation network.

The successor to the fixed 0–120 min correction UNet (`model/unet.py`). One
network serves every lead 0 → 240 h: a shared multimodal encoder, **FiLM**
conditioning on lead-time + time-of-day/season, and a decoder producing the
precip field at the requested lead. Lifts the old `lead/120` hard cap to a
smooth lead embedding.

This is the P2 scaffold (see docs/seamless_model_plan.md §3). v0 is a single
lead-conditioned decoder; the dual-head (nowcast / AIFS-downscale outlook) +
learned seam and the probabilistic head are layered on next. ~1–5 M params.

    Input  : x    (B, C, H, W)  multimodal channel stack
             cond (B, cond_dim)  lead/time conditioning vector
    Output : (B, out_channels, H, W)  precip mm/h (≥ 0)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FiLM(nn.Module):
    """Feature-wise linear modulation: condition feature maps on a vector
    (lead-time, hour, season) by predicting a per-channel scale + shift."""

    def __init__(self, cond_dim: int, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_features * 2),
        )

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)  # each (B, n_features)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


def lead_time_encoding(lead_min: torch.Tensor, valid_hour: torch.Tensor,
                       valid_doy: torch.Tensor) -> torch.Tensor:
    """Build the conditioning vector from lead (minutes), hour-of-day, day-of-year.
    Normalised lead + sinusoidal lead/diurnal/seasonal terms → (B, 7)."""
    horizon = 240.0 * 60.0  # 10 days in minutes
    lead_norm = (lead_min / horizon).clamp(0, 1)
    two_pi = 2 * math.pi
    return torch.stack([
        lead_norm,
        torch.sin(two_pi * lead_norm), torch.cos(two_pi * lead_norm),
        torch.sin(two_pi * valid_hour / 24.0), torch.cos(two_pi * valid_hour / 24.0),
        torch.sin(two_pi * valid_doy / 366.0), torch.cos(two_pi * valid_doy / 366.0),
    ], dim=1)


COND_DIM = 7


class SeamlessUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32,
                 cond_dim: int = COND_DIM, out_channels: int = 1) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.bottleneck = conv_block(c * 2, c * 4)
        self.film = FiLM(cond_dim, c * 4)  # condition at the bottleneck
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = conv_block(c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = conv_block(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.Softplus(beta=2.0)  # non-negative precip

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.film(self.bottleneck(self.pool(e2)), cond)
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.activation(self.head(d2))


class _Decoder(nn.Module):
    """UNet decoder consuming the shared (e1, e2, bottleneck) features."""

    def __init__(self, c: int, out_channels: int) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = conv_block(c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec2 = conv_block(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, b, e2, e1):
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.head(d2)


class SeamlessNet(nn.Module):
    """Dual-head seamless model (docs/seamless_model_plan.md §3): one shared
    multimodal encoder with FiLM lead/time conditioning feeds a **nowcast** head
    (obs-driven, 0–6 h) and an **outlook** head (AIFS-downscaling, 6 h–10 d). A
    learned **seam** gate blends them by lead, so a single network is seamless
    across 0–240 h. The outlook is conditioned on AIFS via the input channels
    (the AIFS field for the target valid-time is in the stack).

    **Probabilistic outlook (rec #3).** Pass ``quantiles=(0.1, 0.5, 0.9)`` to make
    the outlook head emit one channel per quantile instead of a point estimate;
    the network then predicts a *distribution* whose spread widens with lead —
    the honest "chance of rain at day 5" the product needs, and the publishable
    contribution. Quantiles are enforced **non-crossing** (cumulative-softplus),
    and the seam blends each quantile with the (sharp) nowcast point, so spread
    collapses to ~0 at short lead (radar is confident) and grows into the outlook
    regime. ``quantiles=None`` (default) keeps the original deterministic head, so
    existing checkpoints and the CRPS-free training path are unchanged.
    """

    def __init__(self, in_channels: int, base_channels: int = 32,
                 cond_dim: int = COND_DIM, out_channels: int = 1,
                 quantiles: tuple[float, ...] | None = None) -> None:
        super().__init__()
        c = base_channels
        self.quantiles = tuple(quantiles) if quantiles else None
        self.out_channels = out_channels
        n_out = out_channels * (len(self.quantiles) if self.quantiles else 1)
        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.bottleneck = conv_block(c * 2, c * 4)
        self.film = FiLM(cond_dim, c * 4)
        self.nowcast = _Decoder(c, out_channels)  # point estimate (sharp radar regime)
        self.outlook = _Decoder(c, n_out)          # point OR quantiles
        # Seam: per-sample blend weight α(cond) ∈ (0,1); α·nowcast + (1-α)·outlook.
        self.seam = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.Softplus(beta=2.0)

    def _monotone_quantiles(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw outlook channels (B, Q, H, W) → non-negative, **non-crossing**
        quantiles: q₀ = softplus(raw₀); qₖ = q₀ + Σ softplus(Δ). Guarantees
        q₁₀ ≤ q₅₀ ≤ q₉₀ so the interval is never degenerate."""
        base = F.softplus(raw[:, :1], beta=2.0)
        if raw.shape[1] == 1:
            return base
        deltas = F.softplus(raw[:, 1:], beta=2.0)
        return torch.cat([base, base + torch.cumsum(deltas, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, return_parts: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.film(self.bottleneck(self.pool(e2)), cond)
        nc = self.activation(self.nowcast(b, e2, e1))  # (B, 1, H, W) point
        alpha = torch.sigmoid(self.seam(cond))[:, :, None, None]  # (B,1,1,1)
        if self.quantiles is None:
            ol = self.activation(self.outlook(b, e2, e1))
            out = alpha * nc + (1 - alpha) * ol
        else:
            ol = self._monotone_quantiles(self.outlook(b, e2, e1))  # (B, Q, H, W)
            out = alpha * nc + (1 - alpha) * ol  # nc broadcasts over the Q axis
        if return_parts:
            return out, nc, ol, alpha.squeeze()
        return out

    def median(self, out: torch.Tensor) -> torch.Tensor:
        """Point forecast from a forward() output: the deterministic field, or
        the q=0.5 channel (nearest) of a quantile output."""
        if self.quantiles is None:
            return out
        j = min(range(len(self.quantiles)), key=lambda i: abs(self.quantiles[i] - 0.5))
        return out[:, j:j + 1]


def num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    C = 40  # multimodal channel stack (radar history + lightning + GII + CTTH + … + AIFS + static)
    lead = torch.tensor([30.0, 360.0, 14400.0])      # +30 min, +6 h, +10 d
    cond = lead_time_encoding(lead, torch.tensor([14.0, 20.0, 6.0]), torch.tensor([196.0, 196.0, 197.0]))
    x = torch.zeros(3, C, 100, 100)

    single = SeamlessUNet(in_channels=C)
    print(f"SeamlessUNet (v0):  out={tuple(single(x, cond).shape)}  params={num_params(single):,}")

    dual = SeamlessNet(in_channels=C)
    out, nc, ol, alpha = dual(x, cond, return_parts=True)
    print(f"SeamlessNet (dual): out={tuple(out.shape)}  params={num_params(dual):,}")
    print(f"  seam α by lead (+30m,+6h,+10d): {[round(float(a),3) for a in alpha]}  (→1=nowcast, →0=outlook)")

    prob = SeamlessNet(in_channels=C, quantiles=(0.1, 0.5, 0.9))
    qout = prob(x, cond)
    spread = (qout[:, 2] - qout[:, 0]).mean(dim=(1, 2))  # q90-q10 by lead
    mono = bool((qout[:, 1:] >= qout[:, :-1] - 1e-4).all())
    print(f"SeamlessNet (prob): out={tuple(qout.shape)}  params={num_params(prob):,}  non-crossing={mono}")
    print(f"  q90-q10 spread by lead (+30m,+6h,+10d): {[round(float(s),3) for s in spread]}  (should widen)")
