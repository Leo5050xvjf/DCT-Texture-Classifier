from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import image_files, labeled_panel, read_bgr, resize_to_height, to_luma_u8
from dct_texture.map_model import make_map_generator, map_generator_logits


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def blend_window(tile_size: int) -> np.ndarray:
    one_dimensional = np.hanning(tile_size).astype(np.float32)
    window = np.outer(one_dimensional, one_dimensional)
    return np.maximum(window, 0.02).astype(np.float32)


@torch.inference_mode()
def infer_tiled(
    gray_u8: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int,
    overlap: int,
    halo: int,
    batch_size: int,
    sigma: float,
    seed: int,
    kind: str = "unet",
    condition_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    clean = gray_u8.astype(np.float32) / 255.0
    generator = np.random.default_rng(seed)
    noisy = np.clip(clean + generator.normal(0.0, sigma / 255.0, clean.shape), 0.0, 1.0).astype(np.float32)
    padded = cv2.copyMakeBorder(noisy, halo, halo, halo, halo, cv2.BORDER_REFLECT_101)
    clean_padded = cv2.copyMakeBorder(clean, halo, halo, halo, halo, cv2.BORDER_REFLECT_101)
    extra_bottom = max(0, tile_size - padded.shape[0])
    extra_right = max(0, tile_size - padded.shape[1])
    if extra_bottom or extra_right:
        padded = cv2.copyMakeBorder(padded, 0, extra_bottom, 0, extra_right, cv2.BORDER_REFLECT_101)
        clean_padded = cv2.copyMakeBorder(clean_padded, 0, extra_bottom, 0, extra_right, cv2.BORDER_REFLECT_101)

    stride = tile_size - overlap
    ys = tile_starts(padded.shape[0], tile_size, stride)
    xs = tile_starts(padded.shape[1], tile_size, stride)
    coordinates = [(y, x) for y in ys for x in xs]
    accumulator = np.zeros(padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    window = blend_window(tile_size)
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        batch = np.stack([padded[y : y + tile_size, x : x + tile_size] for y, x in batch_coordinates])
        if condition_sigma is None:
            clean_batch = np.stack(
                [clean_padded[y : y + tile_size, x : x + tile_size] for y, x in batch_coordinates]
            )
            supplied = np.sqrt(np.square(batch - clean_batch).mean(axis=(1, 2))) * 255.0
        else:
            supplied = np.full((len(batch),), condition_sigma, dtype=np.float32)
        tensor = torch.from_numpy(batch).to(device)
        sigma_tensor = torch.from_numpy(supplied.astype(np.float32)).to(device)
        predictions = torch.sigmoid(map_generator_logits(model, kind, tensor, sigma_tensor)).cpu().numpy()
        for prediction, (y, x) in zip(predictions, batch_coordinates):
            accumulator[y : y + tile_size, x : x + tile_size] += prediction * window
            weights[y : y + tile_size, x : x + tile_size] += window
    prediction_padded = accumulator / np.maximum(weights, 1e-8)
    h, w = clean.shape
    prediction = prediction_padded[halo : halo + h, halo : halo + w]
    return prediction.astype(np.float32), noisy


def save_case(path: Path, bgr: np.ndarray, noisy: np.ndarray, probability: np.ndarray, output_root: Path) -> dict:
    output_dir = output_root / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "texture_probability.npy", probability)
    cv2.imwrite(
        str(output_dir / "texture_probability_u16.png"),
        np.rint(np.clip(probability, 0.0, 1.0) * 65535.0).astype(np.uint16),
    )
    for threshold in (0.3, 0.5, 0.7):
        cv2.imwrite(
            str(output_dir / f"mask_t{int(threshold * 100):03d}.png"),
            (probability >= threshold).astype(np.uint8) * 255,
        )

    display_height = min(800, bgr.shape[0])
    original_display = resize_to_height(bgr, display_height)
    display_size = (original_display.shape[1], original_display.shape[0])
    noisy_bgr = cv2.cvtColor(np.rint(noisy * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    noisy_display = cv2.resize(noisy_bgr, display_size, interpolation=cv2.INTER_AREA)
    heatmap = cv2.applyColorMap(np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.resize(heatmap, display_size, interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(original_display, 0.55, heatmap, 0.45, 0.0)
    comparison = np.hstack(
        [
            labeled_panel(original_display, "original"),
            labeled_panel(noisy_display, "model input"),
            labeled_panel(heatmap, "G P(texture), absolute 0..1"),
            labeled_panel(overlay, "G overlay"),
        ]
    )
    cv2.imwrite(str(output_dir / "comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "source": str(path.resolve()),
        "height": int(bgr.shape[0]),
        "width": int(bgr.shape[1]),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "fraction_ge_030": float((probability >= 0.3).mean()),
        "fraction_ge_050": float((probability >= 0.5).mean()),
        "fraction_ge_070": float((probability >= 0.7).mean()),
        "output": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the robust full-image map generator without resizing")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/robust_v2/map_generator/best.pt"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=0.0, help="Optional synthetic AWGN on the 8-bit intensity scale")
    parser.add_argument(
        "--condition-sigma", type=float, default=None,
        help="Sigma supplied to conditional G. Default uses oracle RMS for synthetic noise; real noisy input needs an estimate.",
    )
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.tile_size <= args.overlap:
        raise ValueError("tile-size must be greater than overlap")

    files = image_files(args.input_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    kind = checkpoint.get("kind", "unet")
    model = make_map_generator(kind, int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_started = time.perf_counter()
    for index, path in enumerate(files, start=1):
        started = time.perf_counter()
        bgr = read_bgr(path)
        probability, noisy = infer_tiled(
            to_luma_u8(bgr), model, device, args.tile_size, args.overlap, args.halo,
            args.batch_size, args.sigma, args.seed + index,
            kind, args.condition_sigma,
        )
        record = save_case(path, bgr, noisy, probability, args.output_dir)
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
        "resize": False,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "halo": args.halo,
        "blend": "floored 2-D Hann window",
        "synthetic_noise_sigma": args.sigma,
        "model_kind": kind,
        "condition_sigma": args.condition_sigma,
        "total_seconds": time.perf_counter() - total_started,
        "images": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
