from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.core import dct_2d_numpy
from dct_texture.model import DCTTextureClassifier


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how closely the CNN follows simple frequency baselines")
    parser.add_argument("--data", type=Path, default=Path("data/paper_v1/val.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/paper_v1/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_v1/validation_analysis.json"))
    args = parser.parse_args()

    data = np.load(args.data)
    labels = data["labels"].astype(np.int64)
    sobel = data["sobel_scores"].astype(np.float64)
    hf = data["dct_hf_ratio"].astype(np.float64)
    dct = dct_2d_numpy(data["patches"].astype(np.float32) / 255.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DCTTextureClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = checkpoint["dct_mean"]
    std = checkpoint["dct_std"]
    x = torch.from_numpy(((dct - mean) / std)[:, None]).float().to(device)
    probabilities = torch.sigmoid(model(x)).cpu().numpy().astype(np.float64)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    sobel_threshold = metrics["sobel_baseline"]["threshold_selected_on_train"]
    hf_threshold = metrics["dct_high_frequency_baseline"]["threshold_selected_on_train"]
    cnn_pred = probabilities >= 0.5
    sobel_pred = sobel >= sobel_threshold
    hf_pred = hf >= hf_threshold
    cnn_correct = cnn_pred == labels
    sobel_correct = sobel_pred == labels
    result = {
        "samples": int(len(labels)),
        "pearson_cnn_probability_vs_sobel_score": float(np.corrcoef(probabilities, sobel)[0, 1]),
        "pearson_cnn_probability_vs_dct_hf_ratio": float(np.corrcoef(probabilities, hf)[0, 1]),
        "cnn_sobel_binary_agreement": float((cnn_pred == sobel_pred).mean()),
        "cnn_dct_hf_binary_agreement": float((cnn_pred == hf_pred).mean()),
        "cnn_correct_sobel_wrong": int(np.logical_and(cnn_correct, ~sobel_correct).sum()),
        "sobel_correct_cnn_wrong": int(np.logical_and(sobel_correct, ~cnn_correct).sum()),
        "both_wrong": int(np.logical_and(~cnn_correct, ~sobel_correct).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
