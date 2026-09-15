from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def effective_sigma(observed: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Per-image RMS corruption magnitude on the 8-bit intensity scale."""
    return (observed - clean).square().mean(dim=(-2, -1)).sqrt() * 255.0


def fixed_corruption(clean: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    """Apply a deterministic named corruption to a CPU tensor shaped [N,H,W]."""
    generator = torch.Generator().manual_seed(seed)
    if name == "clean":
        return clean.clone()
    family, value_text = name.rsplit("_", 1)
    value = float(value_text)
    if family == "awgn":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        return (clean + noise * (value / 255.0)).clamp(0.0, 1.0)
    if family == "corr_awgn":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        correlated = functional.avg_pool2d(noise[:, None], 3, stride=1, padding=1).squeeze(1) * 3.0
        return (clean + correlated * (value / 255.0)).clamp(0.0, 1.0)
    if family == "poisson_peak":
        # torch.poisson has no generator argument on all supported PyTorch versions.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return (torch.poisson(clean * value) / value).clamp(0.0, 1.0)
    if family == "speckle_std":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        return (clean + clean * noise * value).clamp(0.0, 1.0)
    if family == "saltpepper_prob":
        random = torch.rand(clean.shape, generator=generator, dtype=clean.dtype)
        observed = clean.clone()
        observed[random < value / 2.0] = 0.0
        observed[random > 1.0 - value / 2.0] = 1.0
        return observed
    if family == "sinusoid_amp":
        batch, height, width = clean.shape
        angles = torch.rand((batch, 1, 1), generator=generator) * math.pi
        periods = 2.0 + torch.rand((batch, 1, 1), generator=generator) * 6.0
        phases = torch.rand((batch, 1, 1), generator=generator) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=clean.dtype),
            torch.arange(width, dtype=clean.dtype),
            indexing="ij",
        )
        coordinate = xx[None] * torch.cos(angles) + yy[None] * torch.sin(angles)
        wave = torch.sin(2.0 * math.pi * coordinate / periods + phases)
        return (clean + wave * (value / 255.0)).clamp(0.0, 1.0)
    raise ValueError(f"Unknown corruption: {name}")


def random_awgn(clean: torch.Tensor, maximum_sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    intended = torch.rand((len(clean),), device=clean.device) * maximum_sigma
    observed = (clean + torch.randn_like(clean) * (intended[:, None, None] / 255.0)).clamp(0.0, 1.0)
    return observed, effective_sigma(observed, clean)


def random_mixed_corruption(clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of six corruption families independently to each image."""
    observed = clean.clone()
    families = torch.randint(0, 6, (len(clean),), device=clean.device)
    for family in range(6):
        indices = torch.where(families == family)[0]
        if len(indices) == 0:
            continue
        source = clean[indices]
        if family == 0:
            sigma = torch.rand((len(indices), 1, 1), device=clean.device) * 100.0
            result = source + torch.randn_like(source) * (sigma / 255.0)
        elif family == 1:
            sigma = 10.0 + torch.rand((len(indices), 1, 1), device=clean.device) * 50.0
            noise = torch.randn_like(source)
            correlated = functional.avg_pool2d(noise[:, None], 3, 1, 1).squeeze(1) * 3.0
            result = source + correlated * (sigma / 255.0)
        elif family == 2:
            peak = 5.0 + torch.rand((len(indices), 1, 1), device=clean.device) * 55.0
            result = torch.poisson(source * peak) / peak
        elif family == 3:
            std = 0.05 + torch.rand((len(indices), 1, 1), device=clean.device) * 0.25
            result = source + source * torch.randn_like(source) * std
        elif family == 4:
            probability = 0.005 + torch.rand((len(indices), 1, 1), device=clean.device) * 0.045
            random = torch.rand_like(source)
            result = source.clone()
            result[random < probability / 2.0] = 0.0
            result[random > 1.0 - probability / 2.0] = 1.0
        else:
            batch, height, width = source.shape
            angle = torch.rand((batch, 1, 1), device=clean.device) * math.pi
            period = 2.0 + torch.rand((batch, 1, 1), device=clean.device) * 6.0
            phase = torch.rand((batch, 1, 1), device=clean.device) * (2.0 * math.pi)
            amplitude = 5.0 + torch.rand((batch, 1, 1), device=clean.device) * 25.0
            yy, xx = torch.meshgrid(
                torch.arange(height, device=clean.device, dtype=clean.dtype),
                torch.arange(width, device=clean.device, dtype=clean.dtype),
                indexing="ij",
            )
            coordinate = xx[None] * torch.cos(angle) + yy[None] * torch.sin(angle)
            result = source + torch.sin(2.0 * math.pi * coordinate / period + phase) * (amplitude / 255.0)
        observed[indices] = result.clamp(0.0, 1.0)
    return observed, effective_sigma(observed, clean)
