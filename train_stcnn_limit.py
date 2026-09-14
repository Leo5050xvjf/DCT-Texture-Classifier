from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    paired: bool


EXPERIMENTS = (
    Experiment("student_standard", "spatial32", False),
    Experiment("student_paired", "spatial32", True),
    Experiment("student_paired_conditional", "spatial32_conditional", True),
    Experiment("teacher_large_standard", "spatial32_large", False),
    Experiment("teacher_large_paired", "spatial32_large", True),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_noise(clean: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    scale = sigmas[:, None, None] / 255.0
    return (clean + torch.randn_like(clean) * scale).clamp(0.0, 1.0)


def fixed_noise(clean: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + int(round(sigma * 100)))
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    return (clean + noise * (sigma / 255.0)).clamp(0.0, 1.0)


def model_logits(
    model: nn.Module,
    kind: str,
    contexts: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    spatial = ((contexts - 0.5) / 0.25).unsqueeze(1)
    if kind == "spatial32_conditional":
        return model(spatial, sigmas / 50.0)
    return model(spatial)


@torch.inference_mode()
def predict(
    model: nn.Module,
    kind: str,
    clean_cpu: torch.Tensor,
    sigma: float,
    noise_seed: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    observed = fixed_noise(clean_cpu, sigma, noise_seed)
    outputs = []
    for start in range(0, len(observed), batch_size):
        context = observed[start : start + batch_size].to(device)
        sigmas = torch.full((len(context),), sigma, device=device)
        outputs.append(torch.sigmoid(model_logits(model, kind, context, sigmas)).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_curve(
    model: nn.Module,
    kind: str,
    clean_cpu: torch.Tensor,
    labels: np.ndarray,
    sigmas: list[float],
    noise_seeds: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict]:
    clean_scores = predict(model, kind, clean_cpu, 0.0, noise_seeds[0], device, batch_size)
    by_sigma = {}
    for sigma in sigmas:
        scores_by_seed = [
            predict(model, kind, clean_cpu, sigma, seed, device, batch_size)
            for seed in noise_seeds
        ]
        stacked_scores = np.concatenate(scores_by_seed)
        stacked_labels = np.tile(labels, len(scores_by_seed))
        result = binary_metrics(stacked_labels, stacked_scores, 0.5)
        result.update(
            {
                "probability_texture_mean": float(stacked_scores[stacked_labels == 1].mean()),
                "probability_non_texture_mean": float(stacked_scores[stacked_labels == 0].mean()),
                "binary_agreement_with_clean": float(
                    np.mean([
                        ((scores >= 0.5) == (clean_scores >= 0.5)).mean()
                        for scores in scores_by_seed
                    ])
                ),
                "probability_correlation_with_clean": float(
                    np.mean([
                        np.corrcoef(scores, clean_scores)[0, 1]
                        for scores in scores_by_seed
                    ])
                ),
                "mean_absolute_difference_from_clean": float(
                    np.mean([np.abs(scores - clean_scores).mean() for scores in scores_by_seed])
                ),
                "noise_repeats": len(noise_seeds),
            }
        )
        by_sigma[str(float(sigma))] = result
    return by_sigma


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
    train_max_sigma: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = make_robust_model(experiment.kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_contexts, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    experiment_dir = output_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    selection_sigmas = (
        [0.0, 15.0, 25.0, 50.0]
        if train_max_sigma == 50.0
        else [0.0, 0.25 * train_max_sigma, 0.5 * train_max_sigma, train_max_sigma]
    )
    selection_seed = seed + 9000

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_hard = 0.0
        total_consistency = 0.0
        total_count = 0
        for clean, labels in loader:
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sigmas = torch.rand(len(clean), device=device) * train_max_sigma
            noisy = add_noise(clean, sigmas)
            noisy_logits = model_logits(model, experiment.kind, noisy, sigmas)
            if experiment.paired:
                clean_sigmas = torch.zeros_like(sigmas)
                clean_logits = model_logits(model, experiment.kind, clean, clean_sigmas)
                hard_loss = 0.5 * (criterion(noisy_logits, labels) + criterion(clean_logits, labels))
                consistency = nn.functional.mse_loss(
                    torch.sigmoid(noisy_logits), torch.sigmoid(clean_logits)
                )
                loss = hard_loss + consistency_weight * consistency
            else:
                hard_loss = criterion(noisy_logits, labels)
                consistency = torch.zeros((), device=device)
                loss = hard_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(labels)
            total_loss += float(loss.item()) * count
            total_hard += float(hard_loss.item()) * count
            total_consistency += float(consistency.item()) * count
            total_count += count

        selection_results = []
        for sigma in selection_sigmas:
            scores = predict(
                model, experiment.kind, val_contexts, sigma, selection_seed,
                device, max(batch_size, 1024),
            )
            selection_results.append(binary_metrics(val_labels, scores, 0.5))
        score = float(np.mean([item["balanced_accuracy"] for item in selection_results]))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "hard_label_loss": total_hard / total_count,
            "consistency_loss": total_consistency / total_count,
            "selection_mean_balanced_accuracy": score,
            "selection_min_balanced_accuracy": float(
                min(item["balanced_accuracy"] for item in selection_results)
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
                    "training_sigma": f"uniform [0,{train_max_sigma:g}] per sample",
                },
                experiment_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{experiment.name} epoch={epoch:03d} loss={row['train_loss']:.5f} "
                f"mean_bal_acc={score:.4f}",
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
    record = {
        "name": experiment.name,
        "kind": experiment.kind,
        "paired": experiment.paired,
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_mean_balanced_accuracy": best_score,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return model, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first STCNN limit-study ablations")
    parser.add_argument("--data-dir", type=Path, default=Path("data/robust_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stcnn_limit/core"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--train-max-sigma", type=float, default=50.0)
    parser.add_argument(
        "--evaluation-sigmas",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0, 75.0, 100.0],
    )
    parser.add_argument("--seed", type=int, default=20260915)
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
    val_contexts = torch.from_numpy(val_data["contexts"].astype(np.float32) / 255.0)
    val_labels = val_data["labels"].astype(np.int64)
    evaluation_sigmas = args.evaluation_sigmas
    evaluation_seeds = [args.seed + 20000 + repeat * 1000 for repeat in range(5)]
    summary = {
        "protocol": {
            "train_samples": len(train_labels),
            "validation_samples": len(val_labels),
            "labels": "clean-image Sobel-extreme hard pseudo-labels",
            "training_noise": f"clipped AWGN; sigma uniform [0,{args.train_max_sigma:g}] per sample",
            "evaluation_sigmas": evaluation_sigmas,
            "evaluation_noise_repeats": len(evaluation_seeds),
            "seed": args.seed,
            "device": str(device),
        },
        "models": {},
    }

    for index, experiment in enumerate(selected):
        model, record = train_one(
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
            args.train_max_sigma,
            args.seed + (index + 1) * 100,
        )
        record["by_sigma"] = evaluate_curve(
            model,
            experiment.kind,
            val_contexts,
            val_labels,
            evaluation_sigmas,
            evaluation_seeds,
            device,
            max(args.batch_size, 1024),
        )
        summary["models"][experiment.name] = record
        (args.output_dir / experiment.name / "metrics.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"DONE {experiment.name}: params={record['parameters']:,}, "
            f"sigma50={record['by_sigma']['50.0']['accuracy']:.4f}, "
            f"sigma75={record['by_sigma']['75.0']['accuracy']:.4f}, "
            f"sigma100={record['by_sigma']['100.0']['accuracy']:.4f}",
            flush=True,
        )

    compact = {
        name: {
            "parameters": result["parameters"],
            **{
                f"sigma_{sigma:g}": result["by_sigma"][str(float(sigma))]["accuracy"]
                for sigma in evaluation_sigmas
            },
        }
        for name, result in summary["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
