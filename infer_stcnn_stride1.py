from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from dct_texture.core import (
    aggregate_patch_probabilities,
    image_files,
    labeled_panel,
    read_bgr,
    resize_to_height,
    to_luma_u8,
)
from dct_texture.robust_models import make_robust_model
from infer_spatial32_dense import add_awgn


@torch.inference_mode()
def infer_stride1(
    gray_u8: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    sigma: float,
    seed: int,
    context_size: int = 32,
    target_size: int = 8,
    kind: str = "spatial32",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict every valid 8x8 target origin and overlap-average its score."""
    if gray_u8.ndim != 2:
        raise ValueError(f"Expected a grayscale HxW image, got {gray_u8.shape}")
    if min(gray_u8.shape) < target_size:
        raise ValueError(f"Image {gray_u8.shape} is smaller than target size {target_size}")
    if context_size < target_size or (context_size - target_size) % 2:
        raise ValueError("context_size - target_size must be non-negative and even")
    noisy = add_awgn(gray_u8, sigma, seed)
    halo = (context_size - target_size) // 2
    padded = cv2.copyMakeBorder(noisy, halo, halo, halo, halo, cv2.BORDER_REFLECT_101)
    windows = sliding_window_view(padded, (context_size, context_size))
    grid_height, grid_width = windows.shape[:2]
    expected = (gray_u8.shape[0] - target_size + 1, gray_u8.shape[1] - target_size + 1)
    if (grid_height, grid_width) != expected:
        raise RuntimeError(f"Unexpected context grid {(grid_height, grid_width)} != {expected}")
    probabilities = np.empty((grid_height, grid_width), dtype=np.float32)
    flat_probabilities = probabilities.reshape(-1)
    model.eval()
    total = grid_height * grid_width
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        indices = np.arange(start, stop)
        ys = indices // grid_width
        xs = indices % grid_width
        contexts = np.ascontiguousarray(windows[ys, xs])
        tensor = torch.from_numpy(contexts[:, None]).to(device)
        tensor = (tensor - 0.5) / 0.25
        if kind == "spatial32_conditional":
            supplied_sigma = torch.full(
                (len(tensor),), sigma / 50.0, device=device, dtype=tensor.dtype
            )
            logits = model(tensor, supplied_sigma)
        else:
            logits = model(tensor)
        flat_probabilities[start:stop] = torch.sigmoid(logits).cpu().numpy()
    probability_map = aggregate_patch_probabilities(
        probabilities, gray_u8.shape, patch_size=target_size
    )
    return probability_map.astype(np.float32), noisy


def heatmap(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(values, cv2.COLORMAP_TURBO), size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stride-1 overlap-averaged dense STCNN inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stcnn_limit/stcnn_diverse_distilled.pt"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--input", type=Path, help="Run one image instead of a directory")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stcnn_stride1"))
    parser.add_argument("--gallery", type=Path, default=Path("results/stcnn_stride1/gallery.jpg"))
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gallery-width", type=int, default=1800)
    args = parser.parse_args()
    files = [args.input] if args.input is not None else image_files(args.input_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    kind = checkpoint["kind"]
    if kind not in {"spatial32", "spatial32_conditional", "spatial32_large"}:
        raise ValueError(f"Expected STCNN checkpoint, got {kind}")
    model = make_robust_model(kind).to(device)
    model.load_state_dict(checkpoint["model"])
    predictions: dict[str, dict[float, np.ndarray]] = {}
    records = []
    rows = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    for image_index, path in enumerate(files, start=1):
        bgr = read_bgr(path)
        gray = to_luma_u8(bgr)
        predictions[path.stem] = {}
        panels = [labeled_panel(resize_to_height(bgr, 520), "original")]
        display_size = (panels[0].shape[1], panels[0].shape[0])
        for sigma in args.sigmas:
            started = time.perf_counter()
            probability, noisy = infer_stride1(
                gray, model, device, args.batch_size, sigma, args.seed + image_index,
                kind=kind,
            )
            predictions[path.stem][sigma] = probability
            output_dir = args.output_root / f"sigma_{sigma:g}" / path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "texture_probability.npy", probability)
            cv2.imwrite(str(output_dir / "texture_probability_u16.png"), np.rint(probability * 65535.0).astype(np.uint16))
            cv2.imwrite(str(output_dir / "mask_t050.png"), (probability >= 0.5).astype(np.uint8) * 255)
            fraction = float((probability >= 0.5).mean())
            elapsed = time.perf_counter() - started
            records.append(
                {"image": path.name, "sigma": sigma, "fraction_ge_050": fraction, "elapsed_seconds": elapsed}
            )
            panels.append(labeled_panel(heatmap(probability, display_size), f"stride1 sigma={sigma:g}, >=.5 {fraction:.1%}"))
            print(f"[{image_index}/{len(files)}] {path.name} sigma={sigma:g}: >=.5 {fraction:.1%}, {elapsed:.2f}s", flush=True)
        row = np.hstack(panels)
        row_height = int(round(row.shape[0] * args.gallery_width / row.shape[1]))
        rows.append(cv2.resize(row, (args.gallery_width, row_height), interpolation=cv2.INTER_AREA))
    args.gallery.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.gallery), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])
    payload = {
        "checkpoint": str(args.checkpoint), "kind": kind,
        "input": str(args.input) if args.input is not None else None,
        "input_dir": str(args.input_dir) if args.input_dir is not None else None,
        "method": "stride-1 32x32 contexts; scalar central-8x8 predictions overlap-averaged",
        "resize": False, "sigmas": args.sigmas, "seed": args.seed,
        "device": str(device), "elapsed_seconds": time.perf_counter() - started_all,
        "records": records, "gallery": str(args.gallery),
    }
    args.gallery.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.gallery.resolve()}")


if __name__ == "__main__":
    main()
