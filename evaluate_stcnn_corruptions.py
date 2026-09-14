from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from train_stcnn_limit import model_logits


DEFAULT_MODELS = (
    "paired50=outputs/stcnn_limit/core/student_paired/best.pt",
    "conditioned50=outputs/stcnn_limit/core/student_paired_conditional/best.pt",
    "paired100=outputs/stcnn_limit/train_sigma100/student_paired/best.pt",
    "conditioned100=outputs/stcnn_limit/train_sigma100/student_paired_conditional/best.pt",
    "teacher100=outputs/stcnn_limit/train_sigma100/teacher_large_standard/best.pt",
    "distilled100=outputs/stcnn_limit/distillation_sigma100/student_paired_distilled/best.pt",
)


def seeded_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def corrupt(clean: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    generator = seeded_generator(seed)
    if name == "clean":
        return clean.clone()
    family, value_text = name.rsplit("_", 1)
    value = float(value_text)
    if family == "awgn":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        return (clean + noise * (value / 255.0)).clamp(0.0, 1.0)
    if family == "corr_awgn":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        correlated = functional.avg_pool2d(noise[:, None], 3, stride=1, padding=1).squeeze(1) * 3.0
        return (clean + correlated * (value / 255.0)).clamp(0.0, 1.0)
    if family == "poisson_peak":
        torch.manual_seed(seed)
        return (torch.poisson(clean * value) / value).clamp(0.0, 1.0)
    if family == "speckle_std":
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        return (clean + clean * noise * value).clamp(0.0, 1.0)
    if family == "saltpepper_prob":
        random = torch.rand(clean.shape, generator=generator, dtype=clean.dtype)
        observed = clean.clone()
        observed[random < value / 2.0] = 0.0
        observed[random > 1.0 - value / 2.0] = 1.0
        return observed
    if family == "sinusoid_amp":
        batch, height, width = clean.shape
        angles = torch.rand((batch, 1, 1), generator=generator) * math.pi
        frequencies = 2.0 + torch.rand((batch, 1, 1), generator=generator) * 6.0
        phases = torch.rand((batch, 1, 1), generator=generator) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=clean.dtype),
            torch.arange(width, dtype=clean.dtype),
            indexing="ij",
        )
        coordinate = xx[None] * torch.cos(angles) + yy[None] * torch.sin(angles)
        wave = torch.sin(2.0 * math.pi * coordinate / frequencies + phases)
        return (clean + wave * (value / 255.0)).clamp(0.0, 1.0)
    raise ValueError(f"Unknown corruption: {name}")


@torch.inference_mode()
def predict_observed(
    model: torch.nn.Module,
    kind: str,
    observed: torch.Tensor,
    clean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    # Conditional models receive an oracle per-patch RMS corruption level.
    # This is an upper bound because deployment would need to estimate it.
    effective_sigmas = ((observed - clean).square().mean(dim=(1, 2)).sqrt() * 255.0)
    scores = []
    model.eval()
    for start in range(0, len(observed), batch_size):
        batch = observed[start : start + batch_size].to(device)
        sigmas = effective_sigmas[start : start + batch_size].to(device)
        scores.append(torch.sigmoid(model_logits(model, kind, batch, sigmas)).cpu().numpy())
    return np.concatenate(scores), float(effective_sigmas.mean())


def parse_models(items: list[str], device: torch.device) -> dict[str, tuple]:
    models = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Model must use name=checkpoint syntax: {item}")
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        kind = checkpoint["kind"]
        model = make_robust_model(kind).to(device)
        model.load_state_dict(checkpoint["model"])
        models[name] = (model, kind, path, parameter_count(model))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STCNNs on unseen synthetic corruption families")
    parser.add_argument("--data", type=Path, default=Path("data/robust_v2/val_contexts.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stcnn_limit/corruptions.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20262915)
    args = parser.parse_args()

    data = np.load(args.data)
    clean = torch.from_numpy(data["contexts"].astype(np.float32) / 255.0)
    labels = data["labels"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = parse_models(args.models, device)
    corruptions = [
        "clean",
        "awgn_15",
        "awgn_50",
        "awgn_100",
        "corr_awgn_25",
        "corr_awgn_50",
        "poisson_peak_30",
        "poisson_peak_10",
        "speckle_std_0.1",
        "speckle_std_0.25",
        "saltpepper_prob_0.01",
        "saltpepper_prob_0.05",
        "sinusoid_amp_15",
        "sinusoid_amp_30",
    ]
    score_store = {
        model_name: {corruption: [] for corruption in corruptions}
        for model_name in models
    }
    effective_store = {corruption: [] for corruption in corruptions}
    for repeat in range(args.repeats):
        for corruption_index, corruption in enumerate(corruptions):
            observed = corrupt(clean, corruption, args.seed + repeat * 1000 + corruption_index)
            for model_name, (model, kind, _, _) in models.items():
                scores, effective_sigma = predict_observed(
                    model, kind, observed, clean, device, args.batch_size
                )
                score_store[model_name][corruption].append(scores)
                if model_name == next(iter(models)):
                    effective_store[corruption].append(effective_sigma)

    results = {}
    rows = []
    for model_name, (_, kind, path, parameters) in models.items():
        model_results = {}
        for corruption in corruptions:
            scores = np.concatenate(score_store[model_name][corruption])
            repeated_labels = np.tile(labels, args.repeats)
            metrics = binary_metrics(repeated_labels, scores, 0.5)
            metrics["mean_effective_sigma"] = float(np.mean(effective_store[corruption]))
            model_results[corruption] = metrics
            rows.append(
                {
                    "model": model_name,
                    "kind": kind,
                    "parameters": parameters,
                    "corruption": corruption,
                    "mean_effective_sigma": metrics["mean_effective_sigma"],
                    "accuracy": metrics["accuracy"],
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "auroc": metrics["auroc"],
                }
            )
        results[model_name] = {
            "checkpoint": str(path),
            "kind": kind,
            "parameters": parameters,
            "by_corruption": model_results,
        }

    payload = {
        "protocol": {
            "samples": len(labels),
            "repeats": args.repeats,
            "conditional_sigma": "oracle per-patch RMS difference from clean; deployment upper bound",
            "seed": args.seed,
            "device": str(device),
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
    for model_name in models:
        values = results[model_name]["by_corruption"]
        print(
            model_name,
            " ".join(
                f"{name}={values[name]['accuracy']:.3f}"
                for name in ("awgn_50", "corr_awgn_50", "poisson_peak_10", "sinusoid_amp_30")
            ),
        )


if __name__ == "__main__":
    main()
