from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dct_texture.corruptions import fixed_corruption, random_awgn, random_mixed_corruption
from dct_texture.map_model import make_map_generator, map_generator_logits, parameter_count
from train_map_generator import map_metrics


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    base_channels: int
    noise_scheme: str
    sigma_jitter: float = 0.0
    sigma_randomization_probability: float = 0.0


EXPERIMENTS = (
    Experiment("g_awgn50_paired", "unet", 16, "awgn50"),
    Experiment("g_awgn50_conditional", "unet_conditional", 16, "awgn50"),
    Experiment("g_awgn100_tiny", "unet", 8, "awgn100"),
    Experiment("g_awgn100_paired", "unet", 16, "awgn100"),
    Experiment("g_awgn100_conditional", "unet_conditional", 16, "awgn100"),
    Experiment("g_awgn100_large", "unet", 32, "awgn100"),
    Experiment("g_awgn100_xlarge", "unet", 64, "awgn100"),
    Experiment("g_diverse_tiny", "unet", 8, "diverse"),
    Experiment("g_diverse_paired", "unet", 16, "diverse"),
    Experiment("g_diverse_conditional_oracle", "unet_conditional", 16, "diverse"),
    Experiment("g_diverse_conditional_robust", "unet_conditional", 16, "diverse", 10.0, 0.1),
    Experiment("g_diverse_large", "unet", 32, "diverse"),
    Experiment("g_diverse_xlarge", "unet", 64, "diverse"),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def predict_observed(
    model: nn.Module,
    kind: str,
    observed: torch.Tensor,
    supplied_sigma: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(observed), batch_size):
        batch = observed[start : start + batch_size].to(device)
        sigmas = supplied_sigma[start : start + batch_size].to(device)
        logits = map_generator_logits(model, kind, batch, sigmas)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def selection_corruptions(noise_scheme: str) -> list[str]:
    if noise_scheme == "awgn50":
        return ["clean", "awgn_25", "awgn_50"]
    if noise_scheme == "awgn100":
        return ["clean", "awgn_50", "awgn_100"]
    return [
        "clean", "awgn_100", "corr_awgn_50", "poisson_peak_10",
        "speckle_std_0.25", "saltpepper_prob_0.05", "sinusoid_amp_30",
    ]


def train_one(
    experiment: Experiment,
    train_clean: torch.Tensor,
    train_target: torch.Tensor,
    val_clean: torch.Tensor,
    val_target: np.ndarray,
    output_root: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    consistency_weight: float,
    seed: int,
    target_description: str,
) -> dict:
    set_seed(seed)
    model = make_map_generator(experiment.kind, experiment.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(train_clean, train_target),
        batch_size=(
            batch_size if experiment.base_channels <= 16
            else max(4, batch_size // 2) if experiment.base_channels <= 32
            else max(2, batch_size // 4)
        ),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    output_dir = output_root / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    selection_clean = val_clean[:32]
    selection_target = val_target[:32]

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "supervised": 0.0, "consistency": 0.0, "count": 0}
        for clean, target in loader:
            clean = clean.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if experiment.noise_scheme.startswith("awgn"):
                maximum_sigma = float(experiment.noise_scheme.removeprefix("awgn"))
                observed, true_sigma = random_awgn(clean, maximum_sigma)
            else:
                observed, true_sigma = random_mixed_corruption(clean)
            supplied_sigma = true_sigma
            if experiment.sigma_jitter:
                supplied_sigma = (supplied_sigma + torch.randn_like(supplied_sigma) * experiment.sigma_jitter).clamp(0.0, 150.0)
            if experiment.sigma_randomization_probability:
                randomized = torch.rand_like(supplied_sigma) < experiment.sigma_randomization_probability
                supplied_sigma = torch.where(randomized, torch.rand_like(supplied_sigma) * 100.0, supplied_sigma)

            noisy_probability = torch.sigmoid(map_generator_logits(model, experiment.kind, observed, supplied_sigma))
            clean_probability = torch.sigmoid(
                map_generator_logits(model, experiment.kind, clean, torch.zeros_like(true_sigma))
            )
            supervised = 0.5 * (
                functional.mse_loss(noisy_probability, target)
                + functional.mse_loss(clean_probability, target)
            )
            consistency = functional.mse_loss(noisy_probability, clean_probability)
            loss = supervised + consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(clean)
            running["loss"] += float(loss.item()) * count
            running["supervised"] += float(supervised.item()) * count
            running["consistency"] += float(consistency.item()) * count
            running["count"] += count

        condition_metrics = []
        for index, corruption in enumerate(selection_corruptions(experiment.noise_scheme)):
            observed = fixed_corruption(selection_clean, corruption, seed + 9000 + index)
            oracle_sigma = (observed - selection_clean).square().mean(dim=(-2, -1)).sqrt() * 255.0
            prediction = predict_observed(
                model, experiment.kind, observed, oracle_sigma, device, batch_size
            )
            condition_metrics.append(map_metrics(selection_target, prediction))
        selection_rmse = float(np.mean([item["rmse"] for item in condition_metrics]))
        row = {
            "epoch": epoch,
            "train_loss": running["loss"] / running["count"],
            "supervised_mse": running["supervised"] / running["count"],
            "consistency_mse": running["consistency"] / running["count"],
            "selection_mean_rmse": selection_rmse,
        }
        history.append(row)
        if selection_rmse < best_rmse - 1e-5:
            best_rmse = selection_rmse
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "kind": experiment.kind,
                    "base_channels": experiment.base_channels,
                    "parameters": parameter_count(model),
                    "epoch": epoch,
                    "seed": seed,
                    "experiment": asdict(experiment),
                    "input_range": "luminance [0,1]",
                    "sigma_normalization": "sigma / 100" if experiment.kind == "unet_conditional" else None,
                    "target": target_description,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{experiment.name} epoch={epoch:03d} loss={row['train_loss']:.6f} "
                f"selection_rmse={selection_rmse:.5f}",
                flush=True,
            )
        if stale >= patience:
            break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "experiment": asdict(experiment),
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_selection_mean_rmse": best_rmse,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G capacity/noise-limit experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/map_generator_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g_limit"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--target-description",
        default="clean v1 DCT teacher stride-1 overlap-averaged soft map",
    )
    parser.add_argument("--models", nargs="*", default=[item.name for item in EXPERIMENTS])
    args = parser.parse_args()
    selected = [item for item in EXPERIMENTS if item.name in args.models]
    missing = sorted(set(args.models) - {item.name for item in selected})
    if missing:
        raise ValueError(f"Unknown experiments: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = np.load(args.data_dir / "train.npz")
    val_data = np.load(args.data_dir / "val.npz")
    train_clean = torch.from_numpy(train_data["clean"].astype(np.float32) / 255.0)
    train_target = torch.from_numpy(train_data["target_u16"].astype(np.float32) / 65535.0)
    val_clean = torch.from_numpy(val_data["clean"].astype(np.float32) / 255.0)
    val_target = val_data["target_u16"].astype(np.float32) / 65535.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for experiment in selected:
        summaries.append(
            train_one(
                experiment, train_clean, train_target, val_clean, val_target,
                args.output_dir, device, args.epochs, args.patience, args.batch_size,
                args.lr, args.consistency_weight, args.seed,
                args.target_description,
            )
        )
    payload = {
        "protocol": {
            "train_crops": len(train_clean),
            "validation_crops": len(val_clean),
            "crop_size": int(train_clean.shape[-1]),
            "paired_clean_noisy_supervision": True,
            "consistency_weight": args.consistency_weight,
            "device": str(device),
            "target": args.target_description,
        },
        "models": summaries,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
