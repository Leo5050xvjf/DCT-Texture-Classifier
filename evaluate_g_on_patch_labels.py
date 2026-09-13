from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.map_model import MapGeneratorUNet
from dct_texture.metrics import binary_metrics


def fixed_noise(clean: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + int(round(sigma * 100)))
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    return (clean + noise * (sigma / 255.0)).clamp(0.0, 1.0)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-context G on the same patch labels as robust classifiers")
    parser.add_argument("--contexts", type=Path, default=Path("data/robust_v2/val_contexts.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/robust_v2/map_generator/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/robust_v2/map_generator/patch_label_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20269913)
    args = parser.parse_args()
    data = np.load(args.contexts)
    clean = torch.from_numpy(data["contexts"].astype(np.float32) / 255.0)
    labels = data["labels"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MapGeneratorUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    by_sigma = {}
    for sigma in (0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0):
        noisy = fixed_noise(clean, sigma, args.seed)
        scores = []
        for start in range(0, len(noisy), args.batch_size):
            batch = noisy[start : start + args.batch_size].to(device)
            probability = torch.sigmoid(model(batch[:, None]))
            scores.append(probability[:, 12:20, 12:20].mean(dim=(1, 2)).cpu().numpy())
        scores_np = np.concatenate(scores)
        metrics = binary_metrics(labels, scores_np, 0.5)
        metrics["texture_probability_mean"] = float(scores_np[labels == 1].mean())
        metrics["non_texture_probability_mean"] = float(scores_np[labels == 0].mean())
        by_sigma[str(sigma)] = metrics
        print(f"sigma={sigma:.0f} accuracy={metrics['accuracy']:.4f} auc={metrics['auroc']:.4f}")
    result = {
        "input": "same held-out 32x32 contexts and central 8x8 labels as patch experiments",
        "score": "mean G probability over central 8x8",
        "samples": len(labels),
        "noise_seed": args.seed,
        "by_sigma": by_sigma,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
