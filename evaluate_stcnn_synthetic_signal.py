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
from evaluate_stcnn_corruptions import corrupt, predict_observed


DEFAULT_MODELS = (
    "paired100=outputs/stcnn_limit/train_sigma100/student_paired/best.pt",
    "diverse_small=outputs/stcnn_limit/diverse/student_diverse_paired/best.pt",
    "diverse_distilled=outputs/stcnn_limit/diverse_distilled/best.pt",
    "diverse_conditioned=outputs/stcnn_limit/diverse/student_diverse_conditional/best.pt",
    "diverse_teacher=outputs/stcnn_limit/diverse_seed3/teacher_diverse_large/best.pt",
)


def synthetic_signals(samples_per_class: int, seed: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    size = 32
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    xx = xx.float()[None]
    yy = yy.float()[None]

    # Non-texture: constants and gently varying planes.
    base = 0.2 + torch.rand((samples_per_class, 1, 1), generator=generator) * 0.6
    slope_x = (torch.rand((samples_per_class, 1, 1), generator=generator) - 0.5) * 0.008
    slope_y = (torch.rand((samples_per_class, 1, 1), generator=generator) - 0.5) * 0.008
    smooth = (base + slope_x * (xx - 15.5) + slope_y * (yy - 15.5)).clamp(0.0, 1.0)

    # Clearly structured texture: oriented sinusoidal/checker-like patterns.
    periodic_count = samples_per_class // 2
    base = 0.25 + torch.rand((periodic_count, 1, 1), generator=generator) * 0.5
    angle = torch.rand((periodic_count, 1, 1), generator=generator) * math.pi
    period = 3.0 + torch.rand((periodic_count, 1, 1), generator=generator) * 9.0
    phase = torch.rand((periodic_count, 1, 1), generator=generator) * (2.0 * math.pi)
    amplitude = 0.06 + torch.rand((periodic_count, 1, 1), generator=generator) * 0.22
    coordinate = xx * torch.cos(angle) + yy * torch.sin(angle)
    periodic = (base + amplitude * torch.sin(2.0 * math.pi * coordinate / period + phase)).clamp(0.0, 1.0)

    # Material-like stochastic texture: a correlated random field is part of
    # the clean signal, not observation noise.
    stochastic_count = samples_per_class - periodic_count
    field = torch.randn((stochastic_count, 1, size, size), generator=generator)
    field = functional.avg_pool2d(field, 3, stride=1, padding=1).squeeze(1)
    field = field - field.mean(dim=(1, 2), keepdim=True)
    field = field / field.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    base = 0.25 + torch.rand((stochastic_count, 1, 1), generator=generator) * 0.5
    amplitude = 0.04 + torch.rand((stochastic_count, 1, 1), generator=generator) * 0.16
    stochastic = (base + amplitude * field).clamp(0.0, 1.0)

    clean = torch.cat([smooth, periodic, stochastic], dim=0).float()
    labels = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(periodic_count + stochastic_count, dtype=np.int64),
        ]
    )
    families = np.array(
        ["smooth"] * samples_per_class
        + ["periodic"] * periodic_count
        + ["stochastic"] * stochastic_count
    )
    return clean, labels, families


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether noise robustness suppresses true synthetic texture")
    parser.add_argument("--output", type=Path, default=Path("outputs/stcnn_limit/synthetic_signal.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20266915)
    args = parser.parse_args()

    clean, labels, families = synthetic_signals(args.samples_per_class, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for item in args.models:
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        kind = checkpoint["kind"]
        model = make_robust_model(kind).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = (model, kind, path, parameter_count(model))

    corruptions = ["clean", "awgn_15", "awgn_50", "awgn_100", "corr_awgn_25", "corr_awgn_50"]
    results = {}
    rows = []
    for model_name, (model, kind, path, parameters) in models.items():
        by_corruption = {}
        for corruption_index, corruption_name in enumerate(corruptions):
            score_parts = []
            for repeat in range(args.repeats):
                observed = corrupt(
                    clean,
                    corruption_name,
                    args.seed + 10000 + repeat * 1000 + corruption_index,
                )
                scores, _ = predict_observed(
                    model, kind, observed, clean, device, args.batch_size
                )
                score_parts.append(scores)
            scores = np.concatenate(score_parts)
            repeated_labels = np.tile(labels, args.repeats)
            repeated_families = np.tile(families, args.repeats)
            metrics = binary_metrics(repeated_labels, scores, 0.5)
            family_rates = {
                family: float((scores[repeated_families == family] >= 0.5).mean())
                for family in ("smooth", "periodic", "stochastic")
            }
            metrics["predicted_texture_rate_by_clean_signal"] = family_rates
            by_corruption[corruption_name] = metrics
            rows.append(
                {
                    "model": model_name,
                    "corruption": corruption_name,
                    "accuracy": metrics["accuracy"],
                    "smooth_false_positive_rate": family_rates["smooth"],
                    "periodic_recall": family_rates["periodic"],
                    "stochastic_recall": family_rates["stochastic"],
                }
            )
        results[model_name] = {
            "checkpoint": str(path),
            "kind": kind,
            "parameters": parameters,
            "by_corruption": by_corruption,
        }

    payload = {
        "protocol": {
            "samples_per_binary_class": args.samples_per_class,
            "positive_split": "half periodic, half correlated stochastic clean texture",
            "negative": "constant and gently varying clean planes",
            "repeats": args.repeats,
            "conditional_sigma": "oracle per-patch RMS difference from clean",
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
    for model_name in models:
        print(model_name)
        for corruption_name in ("clean", "awgn_50", "corr_awgn_50"):
            rates = results[model_name]["by_corruption"][corruption_name]["predicted_texture_rate_by_clean_signal"]
            print(
                f"  {corruption_name}: smooth_FP={rates['smooth']:.3f} "
                f"periodic_R={rates['periodic']:.3f} stochastic_R={rates['stochastic']:.3f}"
            )


if __name__ == "__main__":
    main()
