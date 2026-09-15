from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import labeled_panel
from dct_texture.corruptions import effective_sigma, fixed_corruption
from evaluate_g_limit import parse_models
from train_g_limit import predict_observed


DEFAULT_MODELS = (
    "baseline=outputs/robust_v2/map_generator/best.pt",
    "AWGN100 conditional=outputs/g_limit/g_awgn100_conditional/best.pt",
    "AWGN100 large=outputs/g_limit/g_awgn100_large/best.pt",
    "diverse small=outputs/g_limit/g_diverse_paired/best.pt",
    "diverse distilled=outputs/g_limit/g_diverse_distilled/best.pt",
    "diverse large=outputs/g_limit/g_diverse_large/best.pt",
)


def heatmap(values: np.ndarray, size: int) -> np.ndarray:
    color = cv2.applyColorMap(np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.resize(color, (size, size), interpolation=cv2.INTER_NEAREST)


def gray_panel(values: np.ndarray, size: int) -> np.ndarray:
    gray = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (size, size), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual comparison of G limit models")
    parser.add_argument("--data", type=Path, default=Path("data/map_generator_v2/val.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/g_limit"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--corruptions", nargs="+", default=["awgn_100", "corr_awgn_50"])
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261915)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    all_clean = data["clean"].astype(np.float32) / 255.0
    all_target = data["target_u16"].astype(np.float32) / 65535.0
    ordering = np.argsort(all_target.mean(axis=(1, 2)))
    quantile_positions = np.linspace(0, len(ordering) - 1, args.samples + 2, dtype=int)[1:-1]
    indices = ordering[quantile_positions]
    clean = torch.from_numpy(all_clean[indices])
    target = all_target[indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = parse_models(args.models, device)
    manifests = []
    for corruption_index, corruption in enumerate(args.corruptions):
        observed = fixed_corruption(clean, corruption, args.seed + corruption_index)
        sigma = effective_sigma(observed, clean)
        predictions = {
            name: predict_observed(model, kind, observed, sigma, device, args.batch_size)
            for name, (model, kind, _, _, _) in models.items()
        }
        rows = []
        for row_index, dataset_index in enumerate(indices):
            panels = [
                labeled_panel(gray_panel(clean[row_index].numpy(), args.panel_size), f"clean #{dataset_index}"),
                labeled_panel(heatmap(target[row_index], args.panel_size), "clean teacher target"),
                labeled_panel(gray_panel(observed[row_index].numpy(), args.panel_size), corruption),
            ]
            panels.extend(
                labeled_panel(heatmap(values[row_index], args.panel_size), name)
                for name, values in predictions.items()
            )
            rows.append(np.hstack(panels))
        gallery = np.vstack(rows)
        output = args.output_dir / f"g_limit_{corruption}_gallery.jpg"
        cv2.imwrite(str(output), gallery, [cv2.IMWRITE_JPEG_QUALITY, 94])
        manifests.append(
            {
                "corruption": corruption, "output": str(output),
                "validation_indices": indices.tolist(),
                "mean_effective_sigma": float(sigma.mean()),
                "models": {name: str(record[3]) for name, record in models.items()},
            }
        )
        print(f"wrote {output.resolve()}")
    (args.output_dir / "gallery_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
