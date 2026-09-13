from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def to_luma_u8(bgr: np.ndarray) -> np.ndarray:
    """Use OpenCV's BT.601-style BGR-to-gray conversion consistently."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def sobel_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    return cv2.magnitude(gx, gy)


def window_mean_map(values: np.ndarray, window: int) -> np.ndarray:
    """Return means for every fully contained window, indexed by top-left coordinate."""
    h, w = values.shape
    if h < window or w < window:
        raise ValueError(f"Image {w}x{h} is smaller than the {window}x{window} window")
    integral = cv2.integral(values, sdepth=cv2.CV_64F)
    sums = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return (sums / float(window * window)).astype(np.float32)


def extreme_windows(values: np.ndarray, window: int = 100) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    means = window_mean_map(values, window)
    high_index = int(np.argmax(means))
    low_index = int(np.argmin(means))
    high_y, high_x = np.unravel_index(high_index, means.shape)
    low_y, low_x = np.unravel_index(low_index, means.shape)
    return (
        (int(high_x), int(high_y)),
        (int(low_x), int(low_y)),
        float(means[high_y, high_x]),
        float(means[low_y, low_x]),
    )


def dct_basis(size: int = 8, dtype: np.dtype = np.float32) -> np.ndarray:
    basis = np.empty((size, size), dtype=np.float64)
    for k in range(size):
        alpha = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
        for n in range(size):
            basis[k, n] = alpha * math.cos(math.pi * (n + 0.5) * k / size)
    return basis.astype(dtype)


def dct_2d_numpy(patches: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal DCT-II for a batch shaped [N, 8, 8]."""
    x = np.asarray(patches, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    basis = dct_basis(x.shape[-1]) if basis is None else basis
    return np.matmul(np.matmul(basis[None], x), basis.T[None]).astype(np.float32)


def dct_2d_torch(patches: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II for [N, 8, 8], retaining signed coefficients."""
    return torch.matmul(torch.matmul(basis.unsqueeze(0), patches), basis.t().unsqueeze(0))


def high_frequency_ratio(dct: np.ndarray | torch.Tensor, cutoff_sum: int = 4):
    """Energy ratio in coefficients whose frequency indices satisfy u+v >= cutoff_sum."""
    size = dct.shape[-1]
    if isinstance(dct, torch.Tensor):
        indices = torch.arange(size, device=dct.device)
        mask = (indices[:, None] + indices[None, :]) >= cutoff_sum
        energy = dct.square()
        return energy[..., mask].sum(dim=-1) / energy.sum(dim=(-2, -1)).clamp_min(1e-12)
    indices = np.arange(size)
    mask = (indices[:, None] + indices[None, :]) >= cutoff_sum
    energy = np.square(dct)
    return energy[..., mask].sum(axis=-1) / np.maximum(energy.sum(axis=(-2, -1)), 1e-12)


def patch_sobel_scores(patches_u8: np.ndarray) -> np.ndarray:
    scores = []
    for patch in patches_u8:
        scores.append(float(sobel_magnitude(patch).mean()))
    return np.asarray(scores, dtype=np.float32)


def aggregate_patch_probabilities(prob_grid: np.ndarray, image_shape: tuple[int, int], patch_size: int = 8) -> np.ndarray:
    """Average each patch probability over all pixels covered by that patch."""
    h, w = image_shape
    expected = (h - patch_size + 1, w - patch_size + 1)
    if prob_grid.shape != expected:
        raise ValueError(f"Probability grid shape {prob_grid.shape} != expected {expected}")
    canvas = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.float32)
    canvas[: expected[0], : expected[1]] = prob_grid
    valid[: expected[0], : expected[1]] = 1.0
    kernel = np.ones((patch_size, patch_size), dtype=np.float32)
    summed = cv2.filter2D(canvas, cv2.CV_32F, kernel, anchor=(patch_size - 1, patch_size - 1), borderType=cv2.BORDER_CONSTANT)
    counts = cv2.filter2D(valid, cv2.CV_32F, kernel, anchor=(patch_size - 1, patch_size - 1), borderType=cv2.BORDER_CONSTANT)
    return summed / np.maximum(counts, 1.0)


def normalize_for_display(values: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 430), 36), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return output

