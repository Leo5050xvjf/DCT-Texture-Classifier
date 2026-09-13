from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.core import dct_basis
from dct_texture.map_model import MapGeneratorUNet, parameter_count
from dct_texture.model import DCTTextureClassifier
from dct_texture.teacher_map import teacher_maps_for_crops


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fixed_noise(clean: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + int(round(sigma * 100)))
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    return (clean + noise * (sigma / 255.0)).clamp(0.0, 1.0)


def map_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target_flat = target.reshape(-1).astype(np.float64)
    prediction_flat = prediction.reshape(-1).astype(np.float64)
    target_binary = target_flat >= 0.5
    prediction_binary = prediction_flat >= 0.5
    tp = int(np.logical_and(target_binary, prediction_binary).sum())
    tn = int(np.logical_and(~target_binary, ~prediction_binary).sum())
    fp = int(np.logical_and(~target_binary, prediction_binary).sum())
    fn = int(np.logical_and(target_binary, ~prediction_binary).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    difference = prediction_flat - target_flat
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "pearson": float(np.corrcoef(target_flat, prediction_flat)[0, 1]),
        "binary_accuracy": float((target_binary == prediction_binary).mean()),
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "binary_f1": float(f1),
        "target_mean": float(target_flat.mean()),
        "prediction_mean": float(prediction_flat.mean()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@torch.inference_mode()
def predict_g(model: nn.Module, noisy_cpu: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(noisy_cpu), batch_size):
        noisy = noisy_cpu[start : start + batch_size].to(device)
        outputs.append(torch.sigmoid(model(noisy[:, None])).cpu().numpy())
    return np.concatenate(outputs)


@torch.inference_mode()
def predict_direct_teacher(
    teacher: nn.Module,
    noisy_cpu: torch.Tensor,
    device: torch.device,
    basis: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(noisy_cpu), batch_size):
        noisy = noisy_cpu[start : start + batch_size].to(device)
        outputs.append(teacher_maps_for_crops(noisy, teacher, basis, mean, std).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-style full-context noisy map generator G")
    parser.add_argument("--data-dir", type=Path, default=Path("data/map_generator_v2"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robust_v2/map_generator"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = np.load(args.data_dir / "train.npz")
    val_data = np.load(args.data_dir / "val.npz")
    train_clean = torch.from_numpy(train_data["clean"].astype(np.float32) / 255.0)
    train_target = torch.from_numpy(train_data["target_u16"].astype(np.float32) / 65535.0)
    val_clean = torch.from_numpy(val_data["clean"].astype(np.float32) / 255.0)
    val_target_np = val_data["target_u16"].astype(np.float32) / 65535.0
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_clean, train_target), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
    )
    model = MapGeneratorUNet(base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    history = []
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    selection_clean = val_clean[:64]
    selection_target = val_target_np[:64]
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for clean, target in loader:
            clean = clean.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sigmas = torch.rand(len(clean), device=device) * 50.0
            noisy = (clean + torch.randn_like(clean) * (sigmas[:, None, None] / 255.0)).clamp(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.sigmoid(model(noisy[:, None]))
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(clean)
            sample_count += len(clean)
        selection_metrics = []
        for sigma in (15.0, 25.0, 50.0):
            noisy = fixed_noise(selection_clean, sigma, args.seed + 8000)
            prediction = predict_g(model, noisy, device, args.batch_size)
            selection_metrics.append(map_metrics(selection_target, prediction))
        selection_rmse = float(np.mean([item["rmse"] for item in selection_metrics]))
        row = {"epoch": epoch, "train_mse": loss_sum / sample_count, "selection_mean_rmse": selection_rmse}
        history.append(row)
        print(f"G epoch={epoch:03d} train_mse={row['train_mse']:.6f} selection_rmse={selection_rmse:.6f}", flush=True)
        if selection_rmse < best_rmse - 1e-5:
            best_rmse = selection_rmse
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model": model.state_dict(), "base_channels": 16, "epoch": epoch, "seed": args.seed},
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher = DCTTextureClassifier(**teacher_checkpoint["model_config"]).to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    basis = torch.from_numpy(dct_basis()).to(device)
    mean = torch.from_numpy(teacher_checkpoint["dct_mean"]).to(device)
    std = torch.from_numpy(teacher_checkpoint["dct_std"]).to(device)
    sigmas = [0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0]
    evaluation = {}
    for sigma in sigmas:
        noisy = fixed_noise(val_clean, sigma, args.seed + 8000)
        prediction_g = predict_g(model, noisy, device, args.batch_size)
        prediction_direct = predict_direct_teacher(teacher, noisy, device, basis, mean, std, 8)
        evaluation[str(sigma)] = {
            "map_generator_g": map_metrics(val_target_np, prediction_g),
            "clean_teacher_applied_directly_to_noisy": map_metrics(val_target_np, prediction_direct),
        }
        print(
            f"sigma={sigma:.0f} G_rmse={evaluation[str(sigma)]['map_generator_g']['rmse']:.4f} "
            f"direct_rmse={evaluation[str(sigma)]['clean_teacher_applied_directly_to_noisy']['rmse']:.4f}",
            flush=True,
        )
    result = {
        "protocol": {
            "train_crops": len(train_clean), "validation_crops": len(val_clean),
            "crop_size": int(train_clean.shape[-1]), "mixed_training_sigma": "uniform [0,50] per crop",
            "target": "clean v1 DCT teacher soft map", "loss": "pixel MSE (L2)",
            "seed": args.seed, "device": str(device),
        },
        "model": {
            "parameters": parameter_count(model), "best_epoch": best_epoch,
            "epochs_completed": len(history), "elapsed_seconds_including_final_eval": time.perf_counter() - started,
        },
        "by_sigma": evaluation,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
