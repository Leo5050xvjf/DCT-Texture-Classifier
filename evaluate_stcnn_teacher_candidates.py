from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import labeled_panel, read_bgr, to_luma_u8
from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from infer_stcnn_stride1 import infer_stride1
from train_stcnn_limit import predict


DEFAULT_MODELS = (
    "spatial32_mixed=checkpoints/spatial32_mixed.pt",
    "awgn100_paired=checkpoints/stcnn_limit/stcnn_awgn100_paired.pt",
    "diverse_distilled=checkpoints/stcnn_limit/stcnn_diverse_distilled.pt",
    "diverse_sigma_robust=checkpoints/stcnn_limit/stcnn_diverse_sigma_robust.pt",
    "diverse_conditional_oracle=checkpoints/stcnn_limit/stcnn_diverse_conditioned_oracle.pt",
    "diverse_large_teacher=checkpoints/stcnn_limit/stcnn_diverse_teacher.pt",
)


def parse_models(items: list[str], device: torch.device) -> dict[str, tuple]:
    models = {}
    for item in items:
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        kind = checkpoint["kind"]
        model = make_robust_model(kind).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = (model, kind, path, parameter_count(model))
    return models


def heatmap(probability: np.ndarray, width: int = 455) -> np.ndarray:
    height = round(probability.shape[0] * width / probability.shape[1])
    u8 = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, (width, height), interpolation=cv2.INTER_AREA)


def image_panel(bgr: np.ndarray, width: int, title: str) -> np.ndarray:
    height = round(bgr.shape[0] * width / bgr.shape[1])
    return labeled_panel(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate STCNN models as a clean dense teacher")
    parser.add_argument(
        "--image", type=Path,
        default=Path("../Freq-Aware-Seg/data/external_grass_test/piqsels_grass.jpg"),
    )
    parser.add_argument("--patch-data", type=Path, default=Path("data/robust_v2/val_contexts.npz"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 50.0, 100.0])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/stcnn_stride1/teacher_candidates.json")
    )
    parser.add_argument(
        "--gallery", type=Path, default=Path("results/stcnn_stride1/teacher_candidates_piqsels.jpg")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = parse_models(args.models, device)
    patch_data = np.load(args.patch_data)
    contexts = torch.from_numpy(patch_data["contexts"].astype(np.float32) / 255.0)
    labels = patch_data["labels"].astype(np.int64)
    bgr = read_bgr(args.image)
    gray = to_luma_u8(bgr)
    results = {}
    dense_maps: dict[str, dict[float, np.ndarray]] = {}
    started = time.perf_counter()

    for model_index, (name, (model, kind, path, parameters)) in enumerate(models.items()):
        patch_metrics = {}
        dense_maps[name] = {}
        clean_dense = None
        for sigma in args.sigmas:
            patch_scores = predict(
                model, kind, contexts, sigma, args.seed + 1000, device, args.batch_size
            )
            patch_result = binary_metrics(labels, patch_scores, 0.5)
            patch_metrics[str(sigma)] = patch_result
            dense, _ = infer_stride1(
                gray, model, device, args.batch_size, sigma,
                args.seed + model_index, kind=kind,
            )
            dense_maps[name][sigma] = dense
            if sigma == 0:
                clean_dense = dense
        if clean_dense is None:
            raise ValueError("sigmas must include 0")
        dense_metrics = {}
        for sigma in args.sigmas:
            dense = dense_maps[name][sigma]
            dense_metrics[str(sigma)] = {
                "texture_fraction": float((dense >= 0.5).mean()),
                "agreement_with_clean": float(((dense >= 0.5) == (clean_dense >= 0.5)).mean()),
                "mae_from_clean": float(np.abs(dense - clean_dense).mean()),
                "correlation_with_clean": float(np.corrcoef(dense.reshape(-1), clean_dense.reshape(-1))[0, 1]),
            }
        results[name] = {
            "checkpoint": str(path), "kind": kind, "parameters": parameters,
            "patch_validation": patch_metrics, "dense_image": dense_metrics,
        }
        print(
            f"{name}: clean patch bal_acc={patch_metrics['0.0']['balanced_accuracy']:.4f}, "
            f"dense >=.5={dense_metrics['0.0']['texture_fraction']:.1%}, "
            f"sigma50 agreement={dense_metrics['50.0']['agreement_with_clean']:.1%}",
            flush=True,
        )

    panel_width = 455
    rows = []
    for sigma in (0.0, 50.0):
        first_title = "original clean" if sigma == 0 else "original (maps use AWGN 50)"
        panels = [image_panel(bgr, panel_width, first_title)]
        for name in models:
            metrics = results[name]["dense_image"][str(sigma)]
            title = f"{name}, >=.5 {metrics['texture_fraction']:.1%}"
            if sigma:
                title += f", agree {metrics['agreement_with_clean']:.1%}"
            panels.append(labeled_panel(heatmap(dense_maps[name][sigma], panel_width), title))
        rows.append(np.hstack(panels))
    args.gallery.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.gallery), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 95])
    payload = {
        "protocol": {
            "image": str(args.image), "native_shape": list(gray.shape),
            "patch_validation_samples": len(labels),
            "patch_labels": "clean-image Sobel-extreme hard pseudo-labels",
            "dense_method": "stride-1 32x32 context; central-8x8 scalar overlap average",
            "sigmas": args.sigmas, "device": str(device), "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "selection_warning": "Dense image has no human ground truth; texture fraction is descriptive, not accuracy.",
        },
        "models": results, "gallery": str(args.gallery),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {args.gallery.resolve()}")


if __name__ == "__main__":
    main()
