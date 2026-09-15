from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.corruptions import effective_sigma, fixed_corruption
from evaluate_g_limit import extended_metrics, parse_models
from train_g_limit import predict_observed


DEFAULT_MODELS = (
    "awgn50_conditional=outputs/g_limit/g_awgn50_conditional/best.pt",
    "awgn100_conditional=outputs/g_limit/g_awgn100_conditional/best.pt",
    "diverse_conditional_oracle=outputs/g_limit/g_diverse_conditional_oracle/best.pt",
    "diverse_conditional_robust=outputs/g_limit/g_diverse_conditional_robust/best.pt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure conditional G sensitivity to supplied sigma")
    parser.add_argument("--data", type=Path, default=Path("data/map_generator_v2/val.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/g_limit/sigma_sensitivity.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--true-sigmas", type=float, nargs="+", default=[50.0, 100.0])
    parser.add_argument("--claimed-sigmas", type=float, nargs="+", default=[0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20262915)
    args = parser.parse_args()
    data = np.load(args.data)
    clean = torch.from_numpy(data["clean"].astype(np.float32) / 255.0)
    target = data["target_u16"].astype(np.float32) / 65535.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = parse_models(args.models, device)
    rows = []
    results = {}
    for model_name, (model, kind, _, path, parameters) in models.items():
        if kind != "unet_conditional":
            raise ValueError(f"{model_name} is not conditional")
        model_results = {}
        for true_sigma in args.true_sigmas:
            observed_by_repeat = [
                fixed_corruption(clean, f"awgn_{true_sigma:g}", args.seed + repeat)
                for repeat in range(args.repeats)
            ]
            effective_by_repeat = [effective_sigma(observed, clean) for observed in observed_by_repeat]
            claim_results = {}
            claim_specs: list[tuple[str, float | None]] = [
                (f"{claim:g}", claim) for claim in args.claimed_sigmas
            ] + [("oracle_effective", None)]
            for claim_name, claim in claim_specs:
                predictions = []
                for observed, oracle in zip(observed_by_repeat, effective_by_repeat):
                    supplied = oracle if claim is None else torch.full_like(oracle, claim)
                    predictions.append(
                        predict_observed(model, kind, observed, supplied, device, args.batch_size)
                    )
                prediction = np.concatenate(predictions)
                repeated_target = np.tile(target, (args.repeats, 1, 1))
                metrics = extended_metrics(repeated_target, prediction)
                claim_results[claim_name] = metrics
                rows.append(
                    {
                        "model": model_name,
                        "true_nominal_sigma": true_sigma,
                        "mean_effective_sigma": float(torch.cat(effective_by_repeat).mean()),
                        "supplied_sigma": claim_name,
                        "rmse": metrics["rmse"],
                        "binary_accuracy": metrics["binary_accuracy"],
                        "binary_balanced_accuracy": metrics["binary_balanced_accuracy"],
                    }
                )
            model_results[str(true_sigma)] = {
                "mean_effective_sigma": float(torch.cat(effective_by_repeat).mean()),
                "by_supplied_sigma": claim_results,
            }
        results[model_name] = {
            "checkpoint": str(path), "parameters": parameters, "by_true_sigma": model_results,
        }
    payload = {
        "protocol": {
            "validation_crops": len(clean), "repeats": args.repeats,
            "target": "clean v1 DCT teacher soft map", "device": str(device),
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output.resolve()}")
    for model_name, record in results.items():
        scores = record["by_true_sigma"]["100.0"]["by_supplied_sigma"]
        print(
            model_name,
            " ".join(f"claim={key}:{value['binary_accuracy']:.3f}" for key, value in scores.items()),
        )


if __name__ == "__main__":
    main()
