from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from evaluate_stcnn_corruptions import corrupt, predict_observed
from train_stcnn_limit import model_logits, set_seed


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    sigma_jitter: float = 0.0
    sigma_randomization_probability: float = 0.0


EXPERIMENTS = (
    Experiment("student_diverse_paired", "spatial32"),
    Experiment("student_diverse_conditional", "spatial32_conditional"),
    Experiment("student_diverse_conditional_robust", "spatial32_conditional", 10.0, 0.1),
    Experiment("teacher_diverse_large", "spatial32_large"),
)


def random_mixed_corruption(clean: torch.Tensor) -> torch.Tensor:
    """Apply one of six noise families independently to every sample."""
    observed = clean.clone()
    families = torch.randint(0, 6, (len(clean),), device=clean.device)
    for family in range(6):
        indices = torch.where(families == family)[0]
        if len(indices) == 0:
            continue
        source = clean[indices]
        if family == 0:  # AWGN, through sigma 100.
            sigma = torch.rand((len(indices), 1, 1), device=clean.device) * 100.0
            result = source + torch.randn_like(source) * (sigma / 255.0)
        elif family == 1:  # Spatially correlated Gaussian noise.
            sigma = 10.0 + torch.rand((len(indices), 1, 1), device=clean.device) * 50.0
            noise = torch.randn_like(source)
            correlated = functional.avg_pool2d(noise[:, None], 3, 1, 1).squeeze(1) * 3.0
            result = source + correlated * (sigma / 255.0)
        elif family == 2:  # Signal-dependent Poisson noise.
            peak = 5.0 + torch.rand((len(indices), 1, 1), device=clean.device) * 55.0
            result = torch.poisson(source * peak) / peak
        elif family == 3:  # Multiplicative speckle.
            std = 0.05 + torch.rand((len(indices), 1, 1), device=clean.device) * 0.25
            result = source + source * torch.randn_like(source) * std
        elif family == 4:  # Impulse noise.
            probability = 0.005 + torch.rand((len(indices), 1, 1), device=clean.device) * 0.045
            random = torch.rand_like(source)
            result = source.clone()
            result[random < probability / 2.0] = 0.0
            result[random > 1.0 - probability / 2.0] = 1.0
        else:  # Periodic/banding-like corruption.
            batch, height, width = source.shape
            angle = torch.rand((batch, 1, 1), device=clean.device) * math.pi
            period = 2.0 + torch.rand((batch, 1, 1), device=clean.device) * 6.0
            phase = torch.rand((batch, 1, 1), device=clean.device) * (2.0 * math.pi)
            amplitude = 5.0 + torch.rand((batch, 1, 1), device=clean.device) * 25.0
            yy, xx = torch.meshgrid(
                torch.arange(height, device=clean.device, dtype=clean.dtype),
                torch.arange(width, device=clean.device, dtype=clean.dtype),
                indexing="ij",
            )
            coordinate = xx[None] * torch.cos(angle) + yy[None] * torch.sin(angle)
            wave = torch.sin(2.0 * math.pi * coordinate / period + phase)
            result = source + wave * (amplitude / 255.0)
        observed[indices] = result.clamp(0.0, 1.0)
    return observed


def train_one(
    experiment: Experiment,
    train_contexts: torch.Tensor,
    train_labels: torch.Tensor,
    val_contexts: torch.Tensor,
    val_labels: np.ndarray,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    consistency_weight: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = make_robust_model(experiment.kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_contexts, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    experiment_dir = output_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    selection_corruptions = [
        "clean",
        "awgn_50",
        "awgn_100",
        "corr_awgn_25",
        "corr_awgn_50",
        "poisson_peak_10",
        "speckle_std_0.25",
        "saltpepper_prob_0.05",
        "sinusoid_amp_30",
    ]
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_hard = 0.0
        total_consistency = 0.0
        total_count = 0
        for clean, labels in loader:
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            noisy = random_mixed_corruption(clean)
            effective_sigma = ((noisy - clean).square().mean(dim=(1, 2)).sqrt() * 255.0)
            supplied_sigma = effective_sigma
            if experiment.sigma_jitter:
                supplied_sigma = (
                    supplied_sigma + torch.randn_like(supplied_sigma) * experiment.sigma_jitter
                ).clamp(0.0, 150.0)
            if experiment.sigma_randomization_probability:
                randomized = torch.rand_like(supplied_sigma) < experiment.sigma_randomization_probability
                supplied_sigma = torch.where(
                    randomized,
                    torch.rand_like(supplied_sigma) * 100.0,
                    supplied_sigma,
                )
            zero_sigma = torch.zeros_like(effective_sigma)
            noisy_logits = model_logits(model, experiment.kind, noisy, supplied_sigma)
            clean_logits = model_logits(model, experiment.kind, clean, zero_sigma)
            hard = 0.5 * (criterion(noisy_logits, labels) + criterion(clean_logits, labels))
            consistency = nn.functional.mse_loss(
                torch.sigmoid(noisy_logits), torch.sigmoid(clean_logits)
            )
            loss = hard + consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(labels)
            total_loss += float(loss.item()) * count
            total_hard += float(hard.item()) * count
            total_consistency += float(consistency.item()) * count
            total_count += count

        selection_metrics = []
        for corruption_index, corruption_name in enumerate(selection_corruptions):
            observed = corrupt(val_contexts, corruption_name, seed + 9000 + corruption_index)
            scores, _ = predict_observed(
                model, experiment.kind, observed, val_contexts, device, max(batch_size, 1024)
            )
            selection_metrics.append(binary_metrics(val_labels, scores, 0.5))
        score = float(np.mean([metric["balanced_accuracy"] for metric in selection_metrics]))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "hard_loss": total_hard / total_count,
            "consistency_loss": total_consistency / total_count,
            "selection_mean_balanced_accuracy": score,
            "selection_min_balanced_accuracy": float(
                min(metric["balanced_accuracy"] for metric in selection_metrics)
            ),
        }
        history.append(row)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "kind": experiment.kind,
                    "experiment": experiment.__dict__,
                    "parameters": parameter_count(model),
                    "epoch": epoch,
                    "seed": seed,
                    "input_normalization": "(luminance_[0,1] - 0.5) / 0.25",
                    "training_noise": "six-family mixture with paired clean/noisy consistency",
                    "conditional_sigma": "per-patch oracle RMS corruption level" if experiment.kind.endswith("conditional") else None,
                },
                experiment_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{experiment.name} epoch={epoch:03d} loss={row['train_loss']:.5f} "
                f"mean_bal_acc={score:.4f} min={row['selection_min_balanced_accuracy']:.4f}",
                flush=True,
            )
        if stale >= patience:
            break

    with (experiment_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(experiment_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model, {
        "name": experiment.name,
        "kind": experiment.kind,
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_mean_balanced_accuracy": best_score,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diverse-noise STCNN models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/robust_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stcnn_limit/diverse"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20263915)
    parser.add_argument("--models", nargs="*", default=[item.name for item in EXPERIMENTS])
    args = parser.parse_args()

    selected = [experiment for experiment in EXPERIMENTS if experiment.name in args.models]
    missing = sorted(set(args.models) - {experiment.name for experiment in selected})
    if missing:
        raise ValueError(f"Unknown experiments: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = np.load(args.data_dir / "train_contexts.npz")
    val_data = np.load(args.data_dir / "val_contexts.npz")
    train_contexts = torch.from_numpy(train_data["contexts"].astype(np.float32) / 255.0)
    train_labels = torch.from_numpy(train_data["labels"].astype(np.float32))
    natural_train_samples = len(train_labels)
    if args.synthetic_samples_per_class:
        from evaluate_stcnn_synthetic_signal import synthetic_signals

        synthetic_contexts, synthetic_labels, _ = synthetic_signals(
            args.synthetic_samples_per_class,
            args.seed + 50000,
        )
        train_contexts = torch.cat([train_contexts, synthetic_contexts], dim=0)
        train_labels = torch.cat(
            [train_labels, torch.from_numpy(synthetic_labels.astype(np.float32))],
            dim=0,
        )
    val_contexts = torch.from_numpy(val_data["contexts"].astype(np.float32) / 255.0)
    val_labels = val_data["labels"].astype(np.int64)
    summary = {
        "protocol": {
            "train_samples": len(train_labels),
            "natural_train_samples": natural_train_samples,
            "synthetic_anchor_samples": len(train_labels) - natural_train_samples,
            "validation_samples": len(val_labels),
            "noise_families": [
                "AWGN", "correlated AWGN", "Poisson", "speckle", "salt-and-pepper", "sinusoidal"
            ],
            "paired_clean_noisy": True,
            "seed": args.seed,
            "device": str(device),
        },
        "models": {},
    }
    for index, experiment in enumerate(selected):
        _, record = train_one(
            experiment,
            train_contexts,
            train_labels,
            val_contexts,
            val_labels,
            args.output_dir,
            device,
            args.epochs,
            args.patience,
            args.batch_size,
            args.lr,
            args.consistency_weight,
            args.seed + (index + 1) * 100,
        )
        summary["models"][experiment.name] = record
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"DONE {experiment.name}: params={record['parameters']:,}, "
            f"selection={record['best_selection_mean_balanced_accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
