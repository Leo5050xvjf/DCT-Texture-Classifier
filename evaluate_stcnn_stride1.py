from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import labeled_panel
from dct_texture.robust_models import make_robust_model
from infer_spatial32_dense import infer_blockwise
from infer_stcnn_stride1 import infer_stride1


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    if first_flat.std() == 0 or second_flat.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_flat, second_flat)[0, 1])


def comparison_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "binary_agreement": float(((reference >= 0.5) == (prediction >= 0.5)).mean()),
        "mae": float(np.abs(reference - prediction).mean()),
        "rmse": float(np.sqrt(np.square(reference - prediction).mean())),
        "correlation": correlation(reference, prediction),
        "reference_texture_fraction": float((reference >= 0.5).mean()),
        "prediction_texture_fraction": float((prediction >= 0.5).mean()),
    }


def blockiness_metrics(maps: np.ndarray, block_size: int = 8) -> dict[str, float]:
    vertical = np.abs(np.diff(maps, axis=2))
    horizontal = np.abs(np.diff(maps, axis=1))
    vertical_seam = (np.arange(vertical.shape[2]) + 1) % block_size == 0
    horizontal_seam = (np.arange(horizontal.shape[1]) + 1) % block_size == 0
    seam_values = np.concatenate(
        (vertical[:, :, vertical_seam].reshape(-1), horizontal[:, horizontal_seam, :].reshape(-1))
    )
    interior_values = np.concatenate(
        (vertical[:, :, ~vertical_seam].reshape(-1), horizontal[:, ~horizontal_seam, :].reshape(-1))
    )
    seam_mean = float(seam_values.mean())
    interior_mean = float(interior_values.mean())
    return {
        "grid_seam_gradient_mean": seam_mean,
        "non_grid_gradient_mean": interior_mean,
        "grid_to_non_grid_gradient_ratio": float(seam_mean / max(interior_mean, 1e-12)),
        "exactly_flat_neighbor_fraction": float(
            np.mean(np.concatenate((vertical.reshape(-1), horizontal.reshape(-1))) < 1e-8)
        ),
    }


def heatmap(probability: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(
        cv2.applyColorMap(values, cv2.COLORMAP_TURBO), size, interpolation=cv2.INTER_NEAREST
    )


def gray_panel(gray: np.ndarray, title: str) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return labeled_panel(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST), title)


def save_gallery(
    path: Path,
    clean: np.ndarray,
    dct: np.ndarray,
    blockwise: np.ndarray,
    stride1: np.ndarray,
    noisy_stride1: np.ndarray,
) -> list[int]:
    fractions = (stride1 >= 0.5).mean(axis=(1, 2))
    order = np.argsort(fractions)
    indices = sorted(set(int(order[round(q * (len(order) - 1))]) for q in np.linspace(0, 1, 6)))
    rows = []
    for index in indices:
        panels = [
            gray_panel(clean[index], f"crop {index}: clean"),
            labeled_panel(heatmap(dct[index]), "old DCT teacher"),
            labeled_panel(heatmap(blockwise[index]), "STCNN stride=8"),
            labeled_panel(heatmap(stride1[index]), "STCNN stride=1 avg"),
            labeled_panel(heatmap(noisy_stride1[index]), "stride=1 AWGN 50"),
        ]
        rows.append(np.hstack(panels))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return indices


def translated(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


def translation_metrics(
    clean: np.ndarray,
    model: torch.nn.Module,
    kind: str,
    device: torch.device,
    batch_size: int,
    sample_count: int,
) -> dict:
    shifts = ((1, 1), (3, 5), (7, 7))
    methods = {"stride8_blockwise": infer_blockwise, "stride1_overlap": infer_stride1}
    results: dict[str, dict[str, dict[str, float]]] = {name: {} for name in methods}
    for name, inference in methods.items():
        base = [
            inference(image, model, device, batch_size, 0.0, 0, kind=kind)[0]
            for image in clean[:sample_count]
        ]
        for dy, dx in shifts:
            maes = []
            agreements = []
            correlations = []
            margin = 24
            for image, reference in zip(clean[:sample_count], base):
                shifted_map = inference(
                    translated(image, dy, dx), model, device, batch_size, 0.0, 0, kind=kind
                )[0]
                aligned = translated(shifted_map, -dy, -dx)
                reference_inner = reference[margin:-margin, margin:-margin]
                aligned_inner = aligned[margin:-margin, margin:-margin]
                metrics = comparison_metrics(reference_inner, aligned_inner)
                maes.append(metrics["mae"])
                agreements.append(metrics["binary_agreement"])
                correlations.append(metrics["correlation"])
            results[name][f"dy{dy}_dx{dx}"] = {
                "mae": float(np.mean(maes)),
                "binary_agreement": float(np.mean(agreements)),
                "correlation": float(np.nanmean(correlations)),
                "interior_margin": margin,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stride-1 overlap-averaged STCNN maps")
    parser.add_argument("--data", type=Path, default=Path("data/map_generator_v2/val.npz"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/stcnn_limit/stcnn_diverse_distilled.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stcnn_stride1"))
    parser.add_argument(
        "--gallery", type=Path, default=Path("results/stcnn_stride1/validation_comparison.jpg")
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 15.0, 50.0, 100.0])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--translation-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    data = np.load(args.data)
    clean = data["clean"]
    old_dct = data["target_u16"].astype(np.float32) / 65535.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    kind = checkpoint["kind"]
    model = make_robust_model(kind).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[float, np.ndarray]] = {
        "stride8_blockwise": {}, "stride1_overlap": {}
    }
    rows: list[dict] = []
    started = time.perf_counter()
    for method_name, inference in (
        ("stride8_blockwise", infer_blockwise), ("stride1_overlap", infer_stride1)
    ):
        for sigma in args.sigmas:
            maps = []
            condition_started = time.perf_counter()
            for index, image in enumerate(clean):
                prediction, _ = inference(
                    image, model, device, args.batch_size, sigma,
                    args.seed + index, kind=kind,
                )
                maps.append(prediction)
            stacked = np.stack(maps)
            predictions[method_name][sigma] = stacked
            elapsed = time.perf_counter() - condition_started
            print(f"{method_name} sigma={sigma:g}: {len(stacked)} crops in {elapsed:.2f}s", flush=True)
            if sigma == 0:
                metrics = comparison_metrics(old_dct, stacked)
                reference_name = "old_clean_DCT_teacher_descriptive_only"
            else:
                metrics = comparison_metrics(predictions[method_name][0.0], stacked)
                reference_name = "same_method_clean_output"
            row = {
                "method": method_name,
                "sigma": sigma,
                "reference": reference_name,
                "elapsed_seconds": elapsed,
                **metrics,
            }
            rows.append(row)

    clean_method_comparison = comparison_metrics(
        predictions["stride8_blockwise"][0.0], predictions["stride1_overlap"][0.0]
    )
    blockiness = {
        method: blockiness_metrics(predictions[method][0.0]) for method in predictions
    }
    translations = translation_metrics(
        clean, model, kind, device, args.batch_size, args.translation_samples
    )
    gallery_sigma = 50.0 if 50.0 in predictions["stride1_overlap"] else args.sigmas[-1]
    gallery_indices = save_gallery(
        args.gallery, clean, old_dct, predictions["stride8_blockwise"][0.0],
        predictions["stride1_overlap"][0.0], predictions["stride1_overlap"][gallery_sigma],
    )

    clean_stride1 = predictions["stride1_overlap"][0.0]
    np.savez_compressed(
        args.output_dir / "validation_stride1_clean.npz",
        probability_u16=np.rint(clean_stride1 * 65535.0).astype(np.uint16),
        image_names=data["image_names"], x=data["x"], y=data["y"],
    )
    payload = {
        "protocol": {
            "checkpoint": str(args.checkpoint), "kind": kind,
            "parameters": checkpoint.get("parameters"), "validation_crops": len(clean),
            "crop_size": int(clean.shape[-1]), "sigmas": args.sigmas,
            "noise": "independent AWGN in 8-bit intensity units",
            "stride1": "all valid central 8x8 target origins; overlapping scalar probabilities averaged",
            "stride8": "non-overlapping central 8x8 targets; scalar probability repeated over each block",
            "old_dct_comparison": "descriptive only; old DCT map is not ground truth",
            "device": str(device), "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "conditions": rows,
        "clean_stride8_vs_stride1": clean_method_comparison,
        "blockiness": blockiness,
        "translation_equivariance": translations,
        "gallery_indices": gallery_indices,
        "gallery": str(args.gallery),
    }
    output_json = args.output_dir / "evaluation.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
