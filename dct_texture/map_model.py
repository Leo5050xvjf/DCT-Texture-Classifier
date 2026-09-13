from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class MapGeneratorUNet(nn.Module):
    """Standalone U-Net-like noisy-image to clean-texture-map generator."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ResidualBlock(1, c)
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(c * 2, c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(c * 4, c * 4)
        self.down3 = nn.Conv2d(c * 4, c * 8, 4, stride=2, padding=1)
        self.bottleneck = ResidualBlock(c * 8, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ResidualBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ResidualBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ResidualBlock(c * 2, c)
        self.output = nn.Conv2d(c, 1, 1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.activation(self.down1(e1)))
        e3 = self.enc3(self.activation(self.down2(e2)))
        bottleneck = self.bottleneck(self.activation(self.down3(e3)))
        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

