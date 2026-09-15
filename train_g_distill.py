from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.corruptions import fixed_corruption, random_mixed_corruption
from dct_texture.map_model import make_map_generator, map_generator_logits, parameter_count
from train_g_limit import predict_observed, set_seed
from train_map_generator import map_metrics


def teacher_probability(model, kind, observed, sigma):
    return torch.sigmoid(map_generator_logits(model, kind, observed, sigma))


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a diverse large G into compact G")
    parser.add_argument("--data-dir", type=Path, default=Path("data/map_generator_v2"))
    parser.add_argument("--teacher", type=Path, default=Path("outputs/g_limit/g_diverse_large/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g_limit/g_diverse_distilled"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of clean-teacher target versus large-G distillation")
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20263015)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = np.load(args.data_dir / "train.npz")
    val_data = np.load(args.data_dir / "val.npz")
    train_clean = torch.from_numpy(train_data["clean"].astype(np.float32) / 255.0)
    train_target = torch.from_numpy(train_data["target_u16"].astype(np.float32) / 65535.0)
    val_clean = torch.from_numpy(val_data["clean"].astype(np.float32) / 255.0)
    val_target = val_data["target_u16"].astype(np.float32) / 65535.0
    loader = DataLoader(
        TensorDataset(train_clean, train_target), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(args.seed),
    )
    teacher_checkpoint = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher_kind = teacher_checkpoint.get("kind", "unet")
    teacher = make_map_generator(teacher_kind, int(teacher_checkpoint["base_channels"])).to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    student = make_map_generator("unet", 16).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-5)
    selection_names = [
        "clean", "awgn_100", "corr_awgn_50", "poisson_peak_10",
        "speckle_std_0.25", "saltpepper_prob_0.05", "sinusoid_amp_30",
    ]
    selection_clean = val_clean[:32]
    selection_target = val_target[:32]
    history = []
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        student.train()
        totals = {"loss": 0.0, "hard": 0.0, "distill": 0.0, "count": 0}
        for clean, target in loader:
            clean = clean.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            observed, sigma = random_mixed_corruption(clean)
            zero = torch.zeros_like(sigma)
            with torch.no_grad():
                teacher_noisy = teacher_probability(teacher, teacher_kind, observed, sigma)
            student_noisy = torch.sigmoid(map_generator_logits(student, "unet", observed, sigma))
            student_clean = torch.sigmoid(map_generator_logits(student, "unet", clean, zero))
            hard = 0.5 * (
                functional.mse_loss(student_noisy, target)
                + functional.mse_loss(student_clean, target)
            )
            distill = functional.mse_loss(student_noisy, teacher_noisy)
            consistency = functional.mse_loss(student_noisy, student_clean)
            loss = args.alpha * hard + (1.0 - args.alpha) * distill + args.consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(clean)
            totals["loss"] += float(loss.item()) * count
            totals["hard"] += float(hard.item()) * count
            totals["distill"] += float(distill.item()) * count
            totals["count"] += count
        condition_metrics = []
        for index, corruption in enumerate(selection_names):
            observed = fixed_corruption(selection_clean, corruption, args.seed + 9000 + index)
            sigma = (observed - selection_clean).square().mean(dim=(-2, -1)).sqrt() * 255.0
            prediction = predict_observed(student, "unet", observed, sigma, device, args.batch_size)
            condition_metrics.append(map_metrics(selection_target, prediction))
        selection_rmse = float(np.mean([item["rmse"] for item in condition_metrics]))
        row = {
            "epoch": epoch, "loss": totals["loss"] / totals["count"],
            "hard_mse": totals["hard"] / totals["count"],
            "distill_mse": totals["distill"] / totals["count"],
            "selection_mean_rmse": selection_rmse,
        }
        history.append(row)
        if selection_rmse < best_rmse - 1e-5:
            best_rmse = selection_rmse
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": student.state_dict(), "kind": "unet", "base_channels": 16,
                    "parameters": parameter_count(student), "epoch": epoch, "seed": args.seed,
                    "experiment": {
                        "name": "g_diverse_distilled", "teacher": str(args.teacher),
                        "alpha": args.alpha, "noise_scheme": "diverse",
                    },
                    "target": "clean v1 DCT teacher plus large diverse G soft outputs",
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"g_diverse_distilled epoch={epoch:03d} loss={row['loss']:.6f} "
                f"selection_rmse={selection_rmse:.5f}", flush=True,
            )
        if stale >= args.patience:
            break
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "parameters": parameter_count(student), "teacher_parameters": parameter_count(teacher),
        "alpha": args.alpha, "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_selection_mean_rmse": best_rmse, "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
