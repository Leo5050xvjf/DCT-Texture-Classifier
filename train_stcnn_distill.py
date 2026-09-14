from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from train_stcnn_limit import (
    add_noise,
    evaluate_curve,
    model_logits,
    predict,
    set_seed,
)


@dataclass(frozen=True)
class DistillExperiment:
    name: str
    kind: str
    paired: bool


EXPERIMENTS = (
    DistillExperiment("student_distilled", "spatial32", False),
    DistillExperiment("student_paired_distilled", "spatial32", True),
    DistillExperiment("student_conditional_distilled", "spatial32_conditional", False),
)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    target = torch.sigmoid(teacher_logits / temperature)
    return nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature,
        target,
    ) * (temperature ** 2)


def train_one(
    experiment: DistillExperiment,
    teacher: nn.Module,
    teacher_kind: str,
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
    alpha: float,
    temperature: float,
    consistency_weight: float,
    train_max_sigma: float,
    seed: int,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    student = make_robust_model(experiment.kind).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
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
    selection_sigmas = (
        [0.0, 15.0, 25.0, 50.0]
        if train_max_sigma == 50.0
        else [0.0, 0.25 * train_max_sigma, 0.5 * train_max_sigma, train_max_sigma]
    )
    selection_seed = seed + 9000
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0
        total_consistency = 0.0
        total_count = 0
        for clean, labels in loader:
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sigmas = torch.rand(len(clean), device=device) * train_max_sigma
            noisy = add_noise(clean, sigmas)
            student_noisy = model_logits(student, experiment.kind, noisy, sigmas)
            with torch.no_grad():
                teacher_noisy = model_logits(teacher, teacher_kind, noisy, sigmas)
            hard = criterion(student_noisy, labels)
            kd = distillation_loss(student_noisy, teacher_noisy, temperature)
            consistency = torch.zeros((), device=device)
            if experiment.paired:
                zero_sigmas = torch.zeros_like(sigmas)
                student_clean = model_logits(student, experiment.kind, clean, zero_sigmas)
                with torch.no_grad():
                    teacher_clean = model_logits(teacher, teacher_kind, clean, zero_sigmas)
                hard = 0.5 * (hard + criterion(student_clean, labels))
                kd = 0.5 * (kd + distillation_loss(student_clean, teacher_clean, temperature))
                consistency = nn.functional.mse_loss(
                    torch.sigmoid(student_noisy), torch.sigmoid(student_clean)
                )
            loss = (1.0 - alpha) * hard + alpha * kd + consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(labels)
            total_loss += float(loss.item()) * count
            total_hard += float(hard.item()) * count
            total_kd += float(kd.item()) * count
            total_consistency += float(consistency.item()) * count
            total_count += count

        results = []
        for sigma in selection_sigmas:
            scores = predict(
                student, experiment.kind, val_contexts, sigma, selection_seed,
                device, max(batch_size, 1024),
            )
            results.append(binary_metrics(val_labels, scores, 0.5))
        score = float(np.mean([result["balanced_accuracy"] for result in results]))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "hard_loss": total_hard / total_count,
            "distillation_loss": total_kd / total_count,
            "consistency_loss": total_consistency / total_count,
            "selection_mean_balanced_accuracy": score,
        }
        history.append(row)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": student.state_dict(),
                    "kind": experiment.kind,
                    "experiment": experiment.__dict__,
                    "parameters": parameter_count(student),
                    "epoch": epoch,
                    "seed": seed,
                    "distillation_alpha": alpha,
                    "distillation_temperature": temperature,
                    "teacher_kind": teacher_kind,
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
    student.load_state_dict(checkpoint["model"])
    return student, {
        "name": experiment.name,
        "kind": experiment.kind,
        "paired": experiment.paired,
        "parameters": parameter_count(student),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_mean_balanced_accuracy": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "distillation_alpha": alpha,
        "temperature": temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a large robust STCNN into compact students")
    parser.add_argument("--data-dir", type=Path, default=Path("data/robust_v2"))
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("outputs/stcnn_limit/core/teacher_large_standard/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stcnn_limit/distillation"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--train-max-sigma", type=float, default=50.0)
    parser.add_argument(
        "--evaluation-sigmas",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0, 75.0, 100.0],
    )
    parser.add_argument("--models", nargs="*", default=[item.name for item in EXPERIMENTS])
    parser.add_argument("--seed", type=int, default=20261915)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher_kind = teacher_checkpoint["kind"]
    teacher = make_robust_model(teacher_kind).to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    train_data = np.load(args.data_dir / "train_contexts.npz")
    val_data = np.load(args.data_dir / "val_contexts.npz")
    train_contexts = torch.from_numpy(train_data["contexts"].astype(np.float32) / 255.0)
    train_labels = torch.from_numpy(train_data["labels"].astype(np.float32))
    val_contexts = torch.from_numpy(val_data["contexts"].astype(np.float32) / 255.0)
    val_labels = val_data["labels"].astype(np.int64)
    selected = [experiment for experiment in EXPERIMENTS if experiment.name in args.models]
    missing = sorted(set(args.models) - {experiment.name for experiment in selected})
    if missing:
        raise ValueError(f"Unknown experiments: {missing}")
    sigmas = args.evaluation_sigmas
    noise_seeds = [args.seed + 20000 + repeat * 1000 for repeat in range(5)]
    summary = {
        "protocol": {
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_kind": teacher_kind,
            "teacher_parameters": parameter_count(teacher),
            "training_noise": f"clipped AWGN; sigma uniform [0,{args.train_max_sigma:g}] per sample",
            "alpha": args.alpha,
            "temperature": args.temperature,
            "evaluation_noise_repeats": len(noise_seeds),
            "seed": args.seed,
            "device": str(device),
        },
        "models": {},
    }
    for index, experiment in enumerate(selected):
        student, record = train_one(
            experiment,
            teacher,
            teacher_kind,
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
            args.alpha,
            args.temperature,
            args.consistency_weight,
            args.train_max_sigma,
            args.seed + (index + 1) * 100,
        )
        record["by_sigma"] = evaluate_curve(
            student,
            experiment.kind,
            val_contexts,
            val_labels,
            sigmas,
            noise_seeds,
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
            f"DONE {experiment.name}: sigma50={record['by_sigma']['50.0']['accuracy']:.4f}, "
            f"sigma75={record['by_sigma']['75.0']['accuracy']:.4f}, "
            f"sigma100={record['by_sigma']['100.0']['accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
