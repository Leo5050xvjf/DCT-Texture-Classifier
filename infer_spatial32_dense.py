from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from dct_texture.core import image_files, labeled_panel, read_bgr, resize_to_height, to_luma_u8
from dct_texture.robust_models import make_robust_model


def add_awgn(gray_u8: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    clean = gray_u8.astype(np.float32) / 255.0
    generator = np.random.default_rng(seed)
    return np.clip(
        clean + generator.normal(0.0, sigma / 255.0, clean.shape),
        0.0,
        1.0,
    ).astype(np.float32)


@torch.inference_mode()
def infer_blockwise(
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
    """Tile central-target predictions back into a full-resolution map.

    The classifier was trained on a 32x32 context and one label for its central
    8x8 patch. Inference therefore evaluates non-overlapping 8x8 targets and
    paints each scalar probability over its target block. Reflection padding
    supplies context at image borders; the result is cropped to native size.
    """
    if gray_u8.ndim != 2:
        raise ValueError(f"Expected a grayscale HxW image, got {gray_u8.shape}")
    if context_size < target_size or (context_size - target_size) % 2:
        raise ValueError("context_size - target_size must be non-negative and even")

    noisy = add_awgn(gray_u8, sigma, seed)
    height, width = noisy.shape
    padded_height = math.ceil(height / target_size) * target_size
    padded_width = math.ceil(width / target_size) * target_size
    tiled = cv2.copyMakeBorder(
        noisy,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REFLECT_101,
    )

    halo = (context_size - target_size) // 2
    padded = cv2.copyMakeBorder(
        tiled,
        halo,
        halo,
        halo,
        halo,
        cv2.BORDER_REFLECT_101,
    )
    rows = padded_height // target_size
    columns = padded_width // target_size
    coordinates = [
        (row * target_size, column * target_size)
        for row in range(rows)
        for column in range(columns)
    ]
    block_probabilities = np.empty((rows, columns), dtype=np.float32)

    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        contexts = np.stack(
            [padded[y : y + context_size, x : x + context_size] for y, x in batch_coordinates]
        )
        tensor = torch.from_numpy(contexts[:, None]).to(device)
        # Match train_robust_patch.prepare_inputs exactly. The classifier was
        # never trained on raw [0, 1] tensors.
        tensor = (tensor - 0.5) / 0.25
        if kind == "spatial32_conditional":
            supplied_sigma = torch.full(
                (len(tensor),), sigma / 50.0, device=device, dtype=tensor.dtype
            )
            logits = model(tensor, supplied_sigma)
        else:
            logits = model(tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for probability, (y, x) in zip(probabilities, batch_coordinates):
            block_probabilities[y // target_size, x // target_size] = probability

    probability_map = np.repeat(
        np.repeat(block_probabilities, target_size, axis=0),
        target_size,
        axis=1,
    )[:height, :width]
    return probability_map.astype(np.float32), noisy


def heatmap(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(
        cv2.applyColorMap(values, cv2.COLORMAP_TURBO),
        size,
        interpolation=cv2.INTER_NEAREST,
    )


def binary_agreement(first: np.ndarray, second: np.ndarray, threshold: float = 0.5) -> float:
    return float(((first >= threshold) == (second >= threshold)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native-resolution block maps from the spatial 32x32 patch classifier"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/spatial32_mixed.pt"),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/robust_v2/spatial32_demo"),
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("results/robust_v2/spatial32_demo_noise_comparison.jpg"),
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 15.0, 50.0])
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gallery-width", type=int, default=1800)
    args = parser.parse_args()

    files = image_files(args.input_dir)
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")
    if 0.0 not in args.sigmas:
        raise ValueError("--sigmas must include 0 so noise stability can be measured")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    kind = checkpoint.get("kind")
    supported_kinds = {"spatial32", "spatial32_conditional", "spatial32_large"}
    if kind not in supported_kinds:
        raise ValueError(f"Expected an STCNN checkpoint, got {kind!r}")
    model = make_robust_model(kind).to(device)
    model.load_state_dict(checkpoint["model"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[float, np.ndarray]] = {}
    records = []
    total_started = time.perf_counter()

    for image_index, path in enumerate(files, start=1):
        bgr = read_bgr(path)
        gray = to_luma_u8(bgr)
        predictions[path.stem] = {}
        for sigma in args.sigmas:
            started = time.perf_counter()
            probability, noisy = infer_blockwise(
                gray,
                model,
                device,
                args.batch_size,
                sigma,
                args.seed + image_index,
                kind=kind,
            )
            predictions[path.stem][sigma] = probability
            case_dir = args.output_root / f"sigma_{sigma:g}" / path.stem
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(case_dir / "texture_probability.npy", probability)
            cv2.imwrite(
                str(case_dir / "texture_probability_u16.png"),
                np.rint(probability * 65535.0).astype(np.uint16),
            )
            cv2.imwrite(
                str(case_dir / "mask_t050.png"),
                (probability >= 0.5).astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(case_dir / "model_input.jpg"),
                np.rint(noisy * 255.0).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            record = {
                "image": path.name,
                "sigma": sigma,
                "height": int(gray.shape[0]),
                "width": int(gray.shape[1]),
                "probability_mean": float(probability.mean()),
                "probability_std": float(probability.std()),
                "fraction_ge_050": float((probability >= 0.5).mean()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            print(
                f"[{image_index}/{len(files)}] {path.name} sigma={sigma:g}: "
                f">=0.5={record['fraction_ge_050']:.1%}, "
                f"seconds={record['elapsed_seconds']:.2f}",
                flush=True,
            )

    rows = []
    comparisons = []
    for path in files:
        original = resize_to_height(read_bgr(path), 520)
        size = (original.shape[1], original.shape[0])
        panels = [labeled_panel(original, "original")]
        clean_prediction = predictions[path.stem][0.0]
        for sigma in args.sigmas:
            probability = predictions[path.stem][sigma]
            fraction = float((probability >= 0.5).mean())
            agreement = binary_agreement(clean_prediction, probability)
            label = f"Spatial32 sigma={sigma:g}, P>=0.5: {fraction:.1%}"
            if sigma != 0.0:
                label += f", vs s0: {agreement:.1%}"
            panels.append(labeled_panel(heatmap(probability, size), label))
            comparisons.append(
                {
                    "image": path.name,
                    "sigma": sigma,
                    "binary_agreement_with_sigma0": agreement,
                    "probability_correlation_with_sigma0": float(
                        np.corrcoef(probability.ravel(), clean_prediction.ravel())[0, 1]
                    ),
                    "mean_absolute_difference_from_sigma0": float(
                        np.abs(probability - clean_prediction).mean()
                    ),
                }
            )
        row = np.hstack(panels)
        row_height = int(round(row.shape[0] * args.gallery_width / row.shape[1]))
        row = cv2.resize(
            row,
            (args.gallery_width, row_height),
            interpolation=cv2.INTER_AREA,
        )
        title = np.zeros((46, args.gallery_width, 3), dtype=np.uint8)
        cv2.putText(
            title,
            path.stem,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([title, row]))

    args.gallery.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.gallery),
        np.vstack(rows),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise RuntimeError(f"Could not write {args.gallery}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "input_dir": str(args.input_dir),
        "device": str(device),
        "method": "non-overlapping 8x8 targets from 32x32 reflected contexts",
        "model_kind": kind,
        "conditional_sigma": "nominal synthetic AWGN sigma" if kind == "spatial32_conditional" else None,
        "input_normalization": "(luminance_[0,1] - 0.5) / 0.25",
        "resize": False,
        "sigmas": args.sigmas,
        "seed": args.seed,
        "total_seconds": time.perf_counter() - total_started,
        "predictions": records,
        "noise_stability": comparisons,
        "gallery": str(args.gallery),
    }
    summary_path = args.gallery.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.gallery.resolve()}")
    print(f"wrote {summary_path.resolve()}")


if __name__ == "__main__":
    main()
