from __future__ import annotations

import torch
from torch import nn


class DCTTextureClassifier(nn.Module):
    """Two convolution and three fully-connected layers, following the paper description."""

    def __init__(self, conv_channels: tuple[int, int] = (16, 32), fc_widths: tuple[int, int] = (64, 16)) -> None:
        super().__init__()
        c1, c2 = conv_channels
        f1, f2 = fc_widths
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c2 * 2 * 2, f1),
            nn.ReLU(inplace=True),
            nn.Linear(f1, f2),
            nn.ReLU(inplace=True),
            nn.Linear(f2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1)).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

