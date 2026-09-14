from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from train_stcnn_limit import fixed_noise, model_logits


DEFAULT_MODELS = (
    "conditioned50=outputs/stcnn_limit/core/student_paired_conditional/best.pt",
    "conditioned100=outputs/stcnn_limit/train_sigma100/student_paired_conditional/best.pt",
    "diverse_conditioned=outputs/stcnn_limit/diverse/student_diverse_conditional/best.pt",
)


@torch.inference_mode()
def predict_with_claim(
    model: torch.nn.Module,
    kind: str,
    observed: torch.Tensor,
    claimed_sigma: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores = []
    for start in range(0, len(observed), batch_size):
        batch = observed[start : start + batch_size].to(device)
        sigma = claimed_sigma[start : start + batch_size].to(device)
        scores.append(torch.sigmoid(model_logits(model, kind, batch, sigma)).cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test conditional STCNN sensitivity to an incorrect sigma input")
    parser.add_argument("--data", type=Path, default=Path("data/robust_v2/val_contexts.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stcnn_limit/sigma_input_sensitivity.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--true-sigmas", nargs="+", type=float, default=[15.0, 50.0, 100.0])
    parser.add_argument("--claimed-sigmas", nargs="+", type=float, default=[0.0, 15.0, 25.0, 50.0, 75.0, 100.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20264915)
    args = parser.parse_args()

    data = np.load(args.data)
    clean = torch.from_numpy(data["contexts"].astype(np.float32) / 255.0)
    labels = data["labels"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for item in args.models:
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        kind = checkpoint["kind"]
        if kind != "spatial32_conditional":
            raise ValueError(f"{name} is not conditional: {kind}")
        model = make_robust_model(kind).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = (model, kind, path, parameter_count(model))

    records = []
    results = {}
    for model_name, (model, kind, path, parameters) in models.items():
        model_results = {}
        for true_sigma in args.true_sigmas:
            observed_by_repeat = [
                fixed_noise(clean, true_sigma, args.seed + repeat * 1000)
                for repeat in range(args.repeats)
            ]
            claims = [(f"fixed_{value:g}", None, value) for value in args.claimed_sigmas]
            claims.append(("oracle_effective_rms", "oracle", None))
            true_results = {}
            for claim_name, claim_mode, fixed_value in claims:
                score_parts = []
                supplied_means = []
                for observed in observed_by_repeat:
                    if claim_mode == "oracle":
                        supplied = ((observed - clean).square().mean(dim=(1, 2)).sqrt() * 255.0)
                    else:
                        supplied = torch.full((len(clean),), float(fixed_value))
                    supplied_means.append(float(supplied.mean()))
                    score_parts.append(
                        predict_with_claim(model, kind, observed, supplied, device, args.batch_size)
                    )
                scores = np.concatenate(score_parts)
                repeated_labels = np.tile(labels, args.repeats)
                metrics = binary_metrics(repeated_labels, scores, 0.5)
                metrics["mean_supplied_sigma"] = float(np.mean(supplied_means))
                true_results[claim_name] = metrics
                records.append(
                    {
                        "model": model_name,
                        "true_sigma": true_sigma,
                        "claim": claim_name,
                        "mean_supplied_sigma": metrics["mean_supplied_sigma"],
                        "accuracy": metrics["accuracy"],
                        "recall": metrics["recall"],
                        "specificity": metrics["specificity"],
                    }
                )
            model_results[str(true_sigma)] = true_results
        results[model_name] = {
            "checkpoint": str(path),
            "parameters": parameters,
            "by_true_sigma": model_results,
        }

    payload = {
        "protocol": {
            "samples": len(labels),
            "repeats": args.repeats,
            "noise": "clipped AWGN",
            "seed": args.seed,
            "device": str(device),
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {args.output.resolve()}")
    for model_name, result in results.items():
        metrics = result["by_true_sigma"]["50.0"]
        print(
            model_name,
            " ".join(
                f"claim_{claim}={metrics[f'fixed_{claim}']['accuracy']:.3f}"
                for claim in (0, 25, 50, 75, 100)
            ),
        )


if __name__ == "__main__":
    main()
