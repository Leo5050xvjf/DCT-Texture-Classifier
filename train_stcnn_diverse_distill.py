from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.metrics import binary_metrics
from dct_texture.robust_models import make_robust_model, parameter_count
from evaluate_stcnn_corruptions import corrupt, predict_observed
from train_stcnn_distill import distillation_loss
from train_stcnn_diverse import random_mixed_corruption
from train_stcnn_limit import model_logits, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a diverse-noise teacher into a compact STCNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data/robust_v2"))
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("outputs/stcnn_limit/diverse_seed3/teacher_diverse_large/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stcnn_limit/diverse_distilled"))
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20265915)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher_kind = teacher_checkpoint["kind"]
    teacher = make_robust_model(teacher_kind).to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student_kind = "spatial32"
    student = make_robust_model(student_kind).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_data = np.load(args.data_dir / "train_contexts.npz")
    val_data = np.load(args.data_dir / "val_contexts.npz")
    train_contexts = torch.from_numpy(train_data["contexts"].astype(np.float32) / 255.0)
    train_labels = torch.from_numpy(train_data["labels"].astype(np.float32))
    val_contexts = torch.from_numpy(val_data["contexts"].astype(np.float32) / 255.0)
    val_labels = val_data["labels"].astype(np.int64)
    loader = DataLoader(
        TensorDataset(train_contexts, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(args.seed),
    )
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

    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {"loss": 0.0, "hard": 0.0, "kd": 0.0, "consistency": 0.0}
        total_count = 0
        for clean, labels in loader:
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            noisy = random_mixed_corruption(clean)
            effective_sigma = ((noisy - clean).square().mean(dim=(1, 2)).sqrt() * 255.0)
            zero_sigma = torch.zeros_like(effective_sigma)
            student_noisy = model_logits(student, student_kind, noisy, effective_sigma)
            student_clean = model_logits(student, student_kind, clean, zero_sigma)
            with torch.no_grad():
                teacher_noisy = model_logits(teacher, teacher_kind, noisy, effective_sigma)
                teacher_clean = model_logits(teacher, teacher_kind, clean, zero_sigma)
            hard = 0.5 * (
                criterion(student_noisy, labels) + criterion(student_clean, labels)
            )
            kd = 0.5 * (
                distillation_loss(student_noisy, teacher_noisy, args.temperature)
                + distillation_loss(student_clean, teacher_clean, args.temperature)
            )
            consistency = nn.functional.mse_loss(
                torch.sigmoid(student_noisy), torch.sigmoid(student_clean)
            )
            loss = (
                (1.0 - args.alpha) * hard
                + args.alpha * kd
                + args.consistency_weight * consistency
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(labels)
            totals["loss"] += float(loss.item()) * count
            totals["hard"] += float(hard.item()) * count
            totals["kd"] += float(kd.item()) * count
            totals["consistency"] += float(consistency.item()) * count
            total_count += count

        selection_metrics = []
        for corruption_index, corruption_name in enumerate(selection_corruptions):
            observed = corrupt(val_contexts, corruption_name, args.seed + 9000 + corruption_index)
            scores, _ = predict_observed(
                student, student_kind, observed, val_contexts, device, max(args.batch_size, 1024)
            )
            selection_metrics.append(binary_metrics(val_labels, scores, 0.5))
        score = float(np.mean([metric["balanced_accuracy"] for metric in selection_metrics]))
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / total_count,
            "hard_loss": totals["hard"] / total_count,
            "distillation_loss": totals["kd"] / total_count,
            "consistency_loss": totals["consistency"] / total_count,
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
                    "model": student.state_dict(),
                    "kind": student_kind,
                    "parameters": parameter_count(student),
                    "epoch": epoch,
                    "seed": args.seed,
                    "teacher_checkpoint": str(args.teacher_checkpoint),
                    "teacher_kind": teacher_kind,
                    "distillation_alpha": args.alpha,
                    "distillation_temperature": args.temperature,
                    "training_noise": "six-family mixture with paired clean/noisy consistency",
                    "input_normalization": "(luminance_[0,1] - 0.5) / 0.25",
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.5f} "
                f"mean_bal_acc={score:.4f} min={row['selection_min_balanced_accuracy']:.4f}",
                flush=True,
            )
        if stale >= args.patience:
            break

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_kind": teacher_kind,
        "teacher_parameters": parameter_count(teacher),
        "student_kind": student_kind,
        "student_parameters": parameter_count(student),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_mean_balanced_accuracy": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
