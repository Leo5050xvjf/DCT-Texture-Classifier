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

from dct_texture.core import dct_basis, dct_2d_torch
from dct_texture.metrics import binary_metrics
from dct_texture.model import DCTTextureClassifier
from dct_texture.robust_models import make_robust_model, parameter_count


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    noise_mode: str
    fixed_sigma: float | None = None


EXPERIMENTS = [
    Experiment("dct8_fixed15", "dct8", "fixed", 15.0),
    Experiment("dct8_fixed25", "dct8", "fixed", 25.0),
    Experiment("dct8_fixed50", "dct8", "fixed", 50.0),
    Experiment("dct8_mixed", "dct8", "mixed"),
    Experiment("dct8_mixed_conditional", "dct8_conditional", "mixed"),
    Experiment("spatial8_mixed", "spatial8", "mixed"),
    Experiment("spatial32_mixed", "spatial32", "mixed"),
    Experiment("hybrid32_mixed", "hybrid32", "mixed"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_noise(clean: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    return (clean + torch.randn_like(clean) * (sigmas[:, None, None] / 255.0)).clamp(0.0, 1.0)


def fixed_validation_noise(clean: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + int(round(sigma * 100)))
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    return (clean + noise * (sigma / 255.0)).clamp(0.0, 1.0)


def prepare_inputs(
    contexts: torch.Tensor,
    kind: str,
    sigmas: torch.Tensor,
    basis: torch.Tensor,
    dct_mean: torch.Tensor,
    dct_std: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    spatial = ((contexts - 0.5) / 0.25).unsqueeze(1)
    center = contexts[:, 12:20, 12:20]
    dct = ((dct_2d_torch(center, basis) - dct_mean) / dct_std).unsqueeze(1)
    if kind == "dct8":
        return (dct,)
    if kind == "dct8_conditional":
        return dct, sigmas / 50.0
    if kind == "spatial8":
        return (spatial[:, :, 12:20, 12:20],)
    if kind == "spatial32":
        return (spatial,)
    if kind == "hybrid32":
        return spatial, dct
    raise ValueError(kind)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    kind: str,
    clean_contexts_cpu: torch.Tensor,
    labels: np.ndarray,
    sigma: float,
    device: torch.device,
    basis: torch.Tensor,
    dct_mean: torch.Tensor,
    dct_std: torch.Tensor,
    batch_size: int,
    noise_seed: int,
) -> dict:
    model.eval()
    noisy_cpu = fixed_validation_noise(clean_contexts_cpu, sigma, noise_seed)
    probabilities = []
    for start in range(0, len(noisy_cpu), batch_size):
        context = noisy_cpu[start : start + batch_size].to(device)
        sigmas = torch.full((len(context),), sigma, device=device)
        inputs = prepare_inputs(context, kind, sigmas, basis, dct_mean, dct_std)
        probabilities.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    scores = np.concatenate(probabilities)
    result = binary_metrics(labels, scores, 0.5)
    result["probability_texture_mean"] = float(scores[labels == 1].mean())
    result["probability_non_texture_mean"] = float(scores[labels == 0].mean())
    return result


@torch.inference_mode()
def evaluate_clean_only_baseline(
    checkpoint_path: Path,
    clean_contexts_cpu: torch.Tensor,
    labels: np.ndarray,
    sigmas: list[float],
    device: torch.device,
    basis: torch.Tensor,
    batch_size: int,
    noise_seed: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DCTTextureClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = torch.from_numpy(checkpoint["dct_mean"]).to(device)
    std = torch.from_numpy(checkpoint["dct_std"]).to(device)
    by_sigma = {}
    for sigma in sigmas:
        noisy_cpu = fixed_validation_noise(clean_contexts_cpu, sigma, noise_seed)
        scores = []
        for start in range(0, len(noisy_cpu), batch_size):
            context = noisy_cpu[start : start + batch_size].to(device)
            center = context[:, 12:20, 12:20]
            dct = ((dct_2d_torch(center, basis) - mean) / std).unsqueeze(1)
            scores.append(torch.sigmoid(model(dct)).cpu().numpy())
        probabilities = np.concatenate(scores)
        result = binary_metrics(labels, probabilities, 0.5)
        result["probability_texture_mean"] = float(probabilities[labels == 1].mean())
        result["probability_non_texture_mean"] = float(probabilities[labels == 0].mean())
        by_sigma[str(sigma)] = result
    return {"name": "dct8_clean_only_v1", "parameters": sum(p.numel() for p in model.parameters()), "by_sigma": by_sigma}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train robust DCT/spatial patch-classifier comparisons")
    parser.add_argument("--data-dir", type=Path, default=Path("data/robust_v2"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robust_v2/patch_models"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--models", nargs="*", default=[item.name for item in EXPERIMENTS])
    parser.add_argument("--selection-metric", choices=("auroc", "accuracy"), default="auroc")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = np.load(args.data_dir / "train_contexts.npz")
    val_data = np.load(args.data_dir / "val_contexts.npz")
    train_contexts = torch.from_numpy(train_data["contexts"].astype(np.float32) / 255.0)
    train_labels = torch.from_numpy(train_data["labels"].astype(np.float32))
    val_contexts = torch.from_numpy(val_data["contexts"].astype(np.float32) / 255.0)
    val_labels_np = val_data["labels"].astype(np.int64)
    teacher = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    basis = torch.from_numpy(dct_basis()).to(device)
    dct_mean = torch.from_numpy(teacher["dct_mean"]).to(device)
    dct_std = torch.from_numpy(teacher["dct_std"]).to(device)
    validation_sigmas = [0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0]
    selected = [item for item in EXPERIMENTS if item.name in args.models]
    missing = sorted(set(args.models) - {item.name for item in selected})
    if missing:
        raise ValueError(f"Unknown model names: {missing}")

    summary = {
        "protocol": {
            "train_samples": len(train_labels),
            "validation_samples": len(val_labels_np),
            "noise": "AWGN clipped to [0,1]",
            "mixed_training_sigma": "uniform [0,50] per sample",
            "validation_sigmas": validation_sigmas,
            "validation_noise_seed": args.seed + 9000,
            "seed": args.seed,
            "device": str(device),
        },
        "models": {},
    }
    summary["models"]["dct8_clean_only_v1"] = evaluate_clean_only_baseline(
        args.teacher_checkpoint, val_contexts, val_labels_np, validation_sigmas,
        device, basis, 1024, args.seed + 9000,
    )

    for experiment_index, experiment in enumerate(selected):
        model_seed = args.seed + 100 * (experiment_index + 1)
        set_seed(model_seed)
        model = make_robust_model(experiment.kind).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(model_seed)
        loader = DataLoader(
            TensorDataset(train_contexts, train_labels), batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
        )
        model_dir = args.output_dir / experiment.name
        model_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best_score = -1.0
        best_epoch = 0
        stale = 0
        started = time.perf_counter()
        early_sigmas = [experiment.fixed_sigma] if experiment.noise_mode == "fixed" else [0.0, 15.0, 25.0, 50.0]
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_count = 0
            for clean, labels in loader:
                clean = clean.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if experiment.noise_mode == "fixed":
                    sigmas_batch = torch.full((len(clean),), float(experiment.fixed_sigma), device=device)
                else:
                    sigmas_batch = torch.rand(len(clean), device=device) * 50.0
                noisy = add_noise(clean, sigmas_batch)
                inputs = prepare_inputs(noisy, experiment.kind, sigmas_batch, basis, dct_mean, dct_std)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*inputs), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                total_count += len(labels)
            early_results = [
                evaluate(
                    model, experiment.kind, val_contexts, val_labels_np, float(sigma), device,
                    basis, dct_mean, dct_std, 1024, args.seed + 9000,
                )
                for sigma in early_sigmas
            ]
            score = float(np.mean([item[args.selection_metric] for item in early_results]))
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_count,
                "selection_score": score,
                "selection_metric": args.selection_metric,
                "selection_mean_accuracy": float(np.mean([item["accuracy"] for item in early_results])),
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
                        "dct_mean": teacher["dct_mean"],
                        "dct_std": teacher["dct_std"],
                        "seed": model_seed,
                        "epoch": epoch,
                    },
                    model_dir / "best.pt",
                )
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"{experiment.name} epoch={epoch:03d} loss={row['train_loss']:.5f} "
                    f"selection_{args.selection_metric}={score:.4f}", flush=True,
                )
            if stale >= args.patience:
                break
        with (model_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        by_sigma = {
            str(sigma): evaluate(
                model, experiment.kind, val_contexts, val_labels_np, sigma, device,
                basis, dct_mean, dct_std, 1024, args.seed + 9000,
            )
            for sigma in validation_sigmas
        }
        record = {
            "name": experiment.name,
            "kind": experiment.kind,
            "noise_mode": experiment.noise_mode,
            "fixed_sigma": experiment.fixed_sigma,
            "parameters": parameter_count(model),
            "selection_metric": args.selection_metric,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "elapsed_seconds": time.perf_counter() - started,
            "by_sigma": by_sigma,
        }
        summary["models"][experiment.name] = record
        (model_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"DONE {experiment.name}: sigma15_acc={by_sigma['15.0']['accuracy']:.4f}, "
            f"sigma50_acc={by_sigma['50.0']['accuracy']:.4f}, seconds={record['elapsed_seconds']:.1f}",
            flush=True,
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({name: {s: r["accuracy"] for s, r in item["by_sigma"].items()} for name, item in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
