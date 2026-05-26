"""Small UNet for the residual-correction head.

Input  : (B, 29, 100, 100)
Output : (B, 1, 100, 100) — corrected precipitation rate in mm/h (≥ 0)

~1 M parameters with the default channel sizes; trains in hours on a
single consumer GPU. The architecture is deliberately boring: two-level
encoder/decoder UNet with batch-norm, ReLU, and a Softplus output to keep
predictions non-negative without the gradient cliff of ReLU at zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PluvioUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 29,
        base_channels: int = 32,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _conv_block(in_channels, c)
        self.enc2 = _conv_block(c, c * 2)
        self.bottleneck = _conv_block(c * 2, c * 4)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c * 2, c)
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        # Softplus keeps the output non-negative without ReLU's zero-gradient zone.
        self.activation = nn.Softplus(beta=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)            # 100 × 100
        e2 = self.enc2(self.pool(e1))  # 50 × 50
        b = self.bottleneck(self.pool(e2))  # 25 × 25
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))  # 50
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))  # 100
        out = self.head(d2)
        return self.activation(out)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PluvioUNet()
    x = torch.zeros(2, 29, 100, 100)
    y = m(x)
    print(f"PluvioUNet output: {tuple(y.shape)}, params: {num_params(m):,}")
