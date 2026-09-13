from __future__ import annotations

import torch
import torch.nn.functional as functional

from .core import dct_2d_torch


@torch.inference_mode()
def teacher_maps_for_crops(
    clean: torch.Tensor,
    teacher,
    basis: torch.Tensor,
    dct_mean: torch.Tensor,
    dct_std: torch.Tensor,
    patch_batch_size: int = 262144,
) -> torch.Tensor:
    """Generate exact stride-1 overlap-averaged maps for [B,H,W] clean crops."""
    batch, height, width = clean.shape
    unfolded = functional.unfold(clean[:, None], kernel_size=8, stride=1)
    patches = unfolded.transpose(1, 2).reshape(-1, 8, 8)
    probabilities = []
    for start in range(0, len(patches), patch_batch_size):
        dct = dct_2d_torch(patches[start : start + patch_batch_size], basis)
        logits = teacher(((dct - dct_mean) / dct_std).unsqueeze(1))
        probabilities.append(torch.sigmoid(logits))
    grid = torch.cat(probabilities).reshape(batch, 1, height - 7, width - 7)
    kernel = torch.ones((1, 1, 8, 8), device=clean.device, dtype=clean.dtype)
    summed = functional.conv_transpose2d(grid, kernel)
    counts = functional.conv_transpose2d(torch.ones_like(grid), kernel)
    return (summed / counts).squeeze(1)

