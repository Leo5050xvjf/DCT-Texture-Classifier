from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import (
    aggregate_patch_probabilities,
    dct_basis,
    dct_2d_torch,
    high_frequency_ratio,
    image_files,
    labeled_panel,
    normalize_for_display,
    read_bgr,
    resize_to_height,
    sobel_magnitude,
    to_luma_u8,
)
from dct_texture.model import DCTTextureClassifier


@torch.inference_mode()
def infer_grids(
    gray_u8: np.ndarray,
    model: DCTTextureClassifier,
    mean: torch.Tensor,
    std: torch.Tensor,
    basis: torch.Tensor,
    device: torch.device,
    row_block: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = gray_u8.astype(np.float32) / 255.0
    h, w = gray.shape
    ph, pw = h - 7, w - 7
    probability_grid = np.empty((ph, pw), dtype=np.float32)
    hf_grid = np.empty((ph, pw), dtype=np.float32)
    for y0 in range(0, ph, row_block):
        y1 = min(y0 + row_block, ph)
        stripe = gray[y0 : y1 + 7]
        windows = np.lib.stride_tricks.sliding_window_view(stripe, (8, 8))
        patches = np.ascontiguousarray(windows.reshape(-1, 8, 8))
        probabilities = []
        hf_scores = []
        for start in range(0, len(patches), batch_size):
            patch_tensor = torch.from_numpy(patches[start : start + batch_size]).to(device)
            dct = dct_2d_torch(patch_tensor, basis)
            logits = model(((dct - mean) / std).unsqueeze(1))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            hf_scores.append(high_frequency_ratio(dct).cpu().numpy())
        probability_grid[y0:y1] = np.concatenate(probabilities).reshape(y1 - y0, pw)
        hf_grid[y0:y1] = np.concatenate(hf_scores).reshape(y1 - y0, pw)
    return probability_grid, hf_grid


def save_outputs(
    source_path: Path,
    bgr: np.ndarray,
    probability_map: np.ndarray,
    hf_map: np.ndarray,
    sobel_map: np.ndarray,
    output_dir: Path,
) -> dict:
    image_dir = output_dir / source_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / "texture_probability.npy", probability_map.astype(np.float32))
    cv2.imwrite(str(image_dir / "texture_probability_u16.png"), np.rint(np.clip(probability_map, 0, 1) * 65535).astype(np.uint16))
    cv2.imwrite(str(image_dir / "mask_t030.png"), (probability_map >= 0.3).astype(np.uint8) * 255)
    cv2.imwrite(str(image_dir / "mask_t050.png"), (probability_map >= 0.5).astype(np.uint8) * 255)
    cv2.imwrite(str(image_dir / "mask_t070.png"), (probability_map >= 0.7).astype(np.uint8) * 255)

    display_height = min(800, bgr.shape[0])
    original = resize_to_height(bgr, display_height)
    target_size = (original.shape[1], original.shape[0])
    probability_u8 = np.rint(np.clip(probability_map, 0, 1) * 255).astype(np.uint8)
    probability_color = cv2.applyColorMap(probability_u8, cv2.COLORMAP_TURBO)
    hf_color = cv2.applyColorMap(normalize_for_display(hf_map), cv2.COLORMAP_TURBO)
    sobel_color = cv2.applyColorMap(normalize_for_display(sobel_map), cv2.COLORMAP_TURBO)
    probability_color = cv2.resize(probability_color, target_size, interpolation=cv2.INTER_AREA)
    hf_color = cv2.resize(hf_color, target_size, interpolation=cv2.INTER_AREA)
    sobel_color = cv2.resize(sobel_color, target_size, interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(original, 0.55, probability_color, 0.45, 0)
    comparison = np.hstack(
        [
            labeled_panel(original, "original"),
            labeled_panel(probability_color, "CNN P(texture), absolute 0..1"),
            labeled_panel(overlay, "CNN overlay"),
            labeled_panel(hf_color, "DCT HF, per-image normalized"),
            labeled_panel(sobel_color, "Sobel, per-image normalized"),
        ]
    )
    cv2.imwrite(str(image_dir / "comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "source": str(source_path.resolve()),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "probability_mean": float(probability_map.mean()),
        "probability_std": float(probability_map.std()),
        "fraction_ge_030": float((probability_map >= 0.3).mean()),
        "fraction_ge_050": float((probability_map >= 0.5).mean()),
        "fraction_ge_070": float((probability_map >= 0.7).mean()),
        "output": str(image_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style full-resolution soft texture maps")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    parser.add_argument("--evenly-spaced", action="store_true")
    parser.add_argument("--row-block", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()

    files = image_files(args.input_dir)
    if args.limit and len(files) > args.limit:
        if args.evenly_spaced:
            indices = np.linspace(0, len(files) - 1, args.limit, dtype=int)
            files = [files[index] for index in indices]
        else:
            files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DCTTextureClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = torch.from_numpy(checkpoint["dct_mean"]).to(device)
    std = torch.from_numpy(checkpoint["dct_std"]).to(device)
    basis = torch.from_numpy(dct_basis()).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total_started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        started = time.perf_counter()
        bgr = read_bgr(path)
        gray = to_luma_u8(bgr)
        probability_grid, hf_grid = infer_grids(
            gray, model, mean, std, basis, device, args.row_block, args.batch_size,
        )
        probability_map = aggregate_patch_probabilities(probability_grid, gray.shape)
        hf_map = aggregate_patch_probabilities(hf_grid, gray.shape)
        sobel_map = sobel_magnitude(gray)
        record = save_outputs(path, bgr, probability_map, hf_map, sobel_map, args.output_dir)
        record["elapsed_seconds"] = time.perf_counter() - started
        records.append(record)
        print(
            f"[{index}/{len(files)}] {path.name}: mean={record['probability_mean']:.4f}, "
            f">=0.5={record['fraction_ge_050']:.3f}, seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "device": str(device),
        "patch_size": 8,
        "stride": 1,
        "aggregation": "mean of probabilities from all overlapping 8x8 patches",
        "total_seconds": time.perf_counter() - total_started,
        "images": records,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

