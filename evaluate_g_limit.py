from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.corruptions import effective_sigma, fixed_corruption
from dct_texture.map_model import make_map_generator, map_generator_logits, parameter_count
from train_g_limit import predict_observed
from train_map_generator import map_metrics


DEFAULT_MODELS = (
    "baseline_awgn50=outputs/robust_v2/map_generator/best.pt",
    "awgn50_paired=outputs/g_limit/g_awgn50_paired/best.pt",
    "awgn50_conditional=outputs/g_limit/g_awgn50_conditional/best.pt",
    "awgn100_tiny=outputs/g_limit/g_awgn100_tiny/best.pt",
    "awgn100=outputs/g_limit/g_awgn100_paired/best.pt",
    "awgn100_conditional=outputs/g_limit/g_awgn100_conditional/best.pt",
    "awgn100_large=outputs/g_limit/g_awgn100_large/best.pt",
    "awgn100_xlarge=outputs/g_limit/g_awgn100_xlarge/best.pt",
    "diverse_tiny=outputs/g_limit/g_diverse_tiny/best.pt",
    "diverse=outputs/g_limit/g_diverse_paired/best.pt",
    "diverse_conditional_oracle=outputs/g_limit/g_diverse_conditional_oracle/best.pt",
    "diverse_conditional_robust=outputs/g_limit/g_diverse_conditional_robust/best.pt",
    "diverse_large=outputs/g_limit/g_diverse_large/best.pt",
    "diverse_xlarge=outputs/g_limit/g_diverse_xlarge/best.pt",
)

CORRUPTIONS = (
    "clean",
    "awgn_15", "awgn_50", "awgn_75", "awgn_100", "awgn_125", "awgn_150",
    "corr_awgn_25", "corr_awgn_50",
    "poisson_peak_30", "poisson_peak_10",
    "speckle_std_0.1", "speckle_std_0.25",
    "saltpepper_prob_0.01", "saltpepper_prob_0.05",
    "sinusoid_amp_15", "sinusoid_amp_30",
)


def extended_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    result = map_metrics(target, prediction)
    specificity = result["tn"] / max(result["tn"] + result["fp"], 1)
    result["binary_specificity"] = float(specificity)
    result["binary_balanced_accuracy"] = float((result["binary_recall"] + specificity) / 2.0)
    return result


def parse_models(items: list[str], device: torch.device) -> dict[str, tuple]:
    models = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Model must use name=checkpoint syntax: {item}")
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        kind = checkpoint.get("kind", "unet")
        base_channels = int(checkpoint["base_channels"])
        model = make_map_generator(kind, base_channels).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = (model, kind, base_channels, path, parameter_count(model))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G limit models on fixed noise families")
    parser.add_argument("--data", type=Path, default=Path("data/map_generator_v2/val.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/g_limit/evaluation.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261915)
    parser.add_argument(
        "--target-description",
        default="clean v1 DCT teacher stride-1 overlap-averaged soft map",
    )
    args = parser.parse_args()
    data = np.load(args.data)
    clean = torch.from_numpy(data["clean"].astype(np.float32) / 255.0)
    target = data["target_u16"].astype(np.float32) / 65535.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = parse_models(args.models, device)
    rows = []
    results = {}

    for model_name, (model, kind, base_channels, path, parameters) in models.items():
        print(f"evaluating {model_name} ({parameters:,} parameters)", flush=True)
        zero_sigma = torch.zeros((len(clean),), dtype=clean.dtype)
        clean_prediction = predict_observed(model, kind, clean, zero_sigma, device, args.batch_size)
        by_corruption = {}
        for corruption_index, corruption in enumerate(CORRUPTIONS):
            predictions = []
            effective_levels = []
            agreements = []
            clean_maes = []
            clean_correlations = []
            repeat_count = 1 if corruption == "clean" else args.repeats
            for repeat in range(repeat_count):
                observed = fixed_corruption(
                    clean, corruption, args.seed + corruption_index * 100 + repeat
                )
                levels = effective_sigma(observed, clean)
                prediction = predict_observed(
                    model, kind, observed, levels, device, args.batch_size
                )
                predictions.append(prediction)
                effective_levels.append(float(levels.mean()))
                agreements.append(float(((prediction >= 0.5) == (clean_prediction >= 0.5)).mean()))
                clean_maes.append(float(np.abs(prediction - clean_prediction).mean()))
                clean_correlations.append(float(np.corrcoef(prediction.reshape(-1), clean_prediction.reshape(-1))[0, 1]))
            stacked_prediction = np.concatenate(predictions)
            stacked_target = np.tile(target, (repeat_count, 1, 1))
            metrics = extended_metrics(stacked_target, stacked_prediction)
            metrics.update(
                {
                    "mean_effective_sigma": float(np.mean(effective_levels)),
                    "binary_agreement_with_own_clean_output": float(np.mean(agreements)),
                    "mae_from_own_clean_output": float(np.mean(clean_maes)),
                    "correlation_with_own_clean_output": float(np.mean(clean_correlations)),
                    "noise_repeats": repeat_count,
                }
            )
            by_corruption[corruption] = metrics
            rows.append(
                {
                    "model": model_name,
                    "kind": kind,
                    "base_channels": base_channels,
                    "parameters": parameters,
                    "corruption": corruption,
                    "mean_effective_sigma": metrics["mean_effective_sigma"],
                    "rmse_to_clean_teacher": metrics["rmse"],
                    "binary_accuracy_to_clean_teacher": metrics["binary_accuracy"],
                    "binary_balanced_accuracy_to_clean_teacher": metrics["binary_balanced_accuracy"],
                    "binary_agreement_with_own_clean_output": metrics["binary_agreement_with_own_clean_output"],
                    "mae_from_own_clean_output": metrics["mae_from_own_clean_output"],
                }
            )
        results[model_name] = {
            "checkpoint": str(path),
            "kind": kind,
            "base_channels": base_channels,
            "parameters": parameters,
            "by_corruption": by_corruption,
        }
        selected = by_corruption
        print(
            " ".join(
                f"{name}={selected[name]['binary_accuracy']:.3f}"
                for name in ("clean", "awgn_50", "awgn_100", "corr_awgn_50", "sinusoid_amp_30")
            ),
            flush=True,
        )

    payload = {
        "protocol": {
            "validation_crops": len(clean),
            "crop_size": int(clean.shape[-1]),
            "repeats": args.repeats,
            "target": args.target_description,
            "conditional_sigma": "oracle per-crop RMS corruption level; deployment upper bound",
            "device": str(device),
            "seed": args.seed,
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {csv_path.resolve()}")


if __name__ == "__main__":
    main()
