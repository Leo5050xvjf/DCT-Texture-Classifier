from __future__ import annotations

import torch
from torch import nn


class DCT8RobustClassifier(nn.Module):
    def __init__(self, conditional_sigma: bool = False) -> None:
        super().__init__()
        self.conditional_sigma = conditional_sigma
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2 + int(conditional_sigma), 64), nn.ReLU(inplace=True),
            nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Linear(16, 1),
        )

    def forward(self, dct: torch.Tensor, sigma_normalized: torch.Tensor | None = None) -> torch.Tensor:
        features = self.features(dct).flatten(1)
        if self.conditional_sigma:
            if sigma_normalized is None:
                raise ValueError("conditional model requires sigma_normalized")
            features = torch.cat([features, sigma_normalized[:, None]], dim=1)
        return self.classifier(features).squeeze(1)


class Spatial8Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Linear(16, 1),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(patch).flatten(1)).squeeze(1)


class Spatial32Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Linear(16, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(context).flatten(1)).squeeze(1)


class Spatial32ConditionalClassifier(nn.Module):
    """Small STCNN with an explicit normalized noise-level input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.sigma_encoder = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4 + 32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Linear(16, 1),
        )

    def forward(self, context: torch.Tensor, sigma_normalized: torch.Tensor) -> torch.Tensor:
        spatial = self.features(context).flatten(1)
        sigma_features = self.sigma_encoder(sigma_normalized[:, None])
        return self.classifier(torch.cat([spatial, sigma_features], dim=1)).squeeze(1)


class Spatial32LargeClassifier(nn.Module):
    """Higher-capacity STCNN teacher with the same 32x32 input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(context).flatten(1)).squeeze(1)


class HybridSpatialDCTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.frequency = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4 + 32 * 2 * 2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 16), nn.ReLU(inplace=True), nn.Linear(16, 1),
        )

    def forward(self, context: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(context).flatten(1)
        frequency = self.frequency(dct).flatten(1)
        return self.classifier(torch.cat([spatial, frequency], dim=1)).squeeze(1)


def make_robust_model(kind: str) -> nn.Module:
    if kind == "dct8":
        return DCT8RobustClassifier(False)
    if kind == "dct8_conditional":
        return DCT8RobustClassifier(True)
    if kind == "spatial8":
        return Spatial8Classifier()
    if kind == "spatial32":
        return Spatial32Classifier()
    if kind == "spatial32_conditional":
        return Spatial32ConditionalClassifier()
    if kind == "spatial32_large":
        return Spatial32LargeClassifier()
    if kind == "hybrid32":
        return HybridSpatialDCTClassifier()
    raise ValueError(f"Unknown robust model kind: {kind}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
