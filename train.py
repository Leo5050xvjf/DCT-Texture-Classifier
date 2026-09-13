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

from dct_texture.core import dct_2d_numpy
from dct_texture.metrics import best_balanced_threshold, binary_metrics
from dct_texture.model import DCTTextureClassifier, parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    patches = data["patches"].astype(np.float32) / 255.0
    labels = data["labels"].astype(np.float32)
    dct = dct_2d_numpy(patches)
    return dct, labels, data["sobel_scores"].astype(np.float32), data["dct_hf_ratio"].astype(np.float32)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size].to(device, non_blocking=True))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-style DCT texture classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/paper_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_v1"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dct, train_labels, train_sobel, train_hf = load_split(args.data_dir / "train.npz")
    val_dct, val_labels, val_sobel, val_hf = load_split(args.data_dir / "val.npz")

    mean = train_dct.mean(axis=0, keepdims=True)
    std = train_dct.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-5)
    train_x = torch.from_numpy(((train_dct - mean) / std)[:, None]).float()
    val_x = torch.from_numpy(((val_dct - mean) / std)[:, None]).float()
    train_y = torch.from_numpy(train_labels).float()
    val_y = torch.from_numpy(val_labels).float()
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCTTextureClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_auc = -1.0
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_y)
            count += len(batch_y)
        val_prob = predict(model, val_x, device)
        metrics = binary_metrics(val_labels, val_prob, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["f1"],
            "val_auroc": metrics["auroc"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1']:.4f} val_auc={row['val_auroc']:.4f}",
            flush=True,
        )
        if float(metrics["auroc"]) > best_auc + 1e-6:
            best_auc = float(metrics["auroc"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "dct_mean": mean.astype(np.float32),
                    "dct_std": std.astype(np.float32),
                    "model_config": {"conv_channels": (16, 32), "fc_widths": (64, 16)},
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_prob = predict(model, train_x, device)
    val_prob = predict(model, val_x, device)
    sobel_threshold = best_balanced_threshold(train_labels, train_sobel)
    hf_threshold = best_balanced_threshold(train_labels, train_hf)
    results = {
        "run": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "parameters": parameter_count(model),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "cnn": {
            "train": binary_metrics(train_labels, train_prob, 0.5),
            "validation": binary_metrics(val_labels, val_prob, 0.5),
        },
        "sobel_baseline": {
            "threshold_selected_on_train": sobel_threshold,
            "train": binary_metrics(train_labels, train_sobel, sobel_threshold),
            "validation": binary_metrics(val_labels, val_sobel, sobel_threshold),
        },
        "dct_high_frequency_baseline": {
            "cutoff": "u+v >= 4, energy ratio",
            "threshold_selected_on_train": hf_threshold,
            "train": binary_metrics(train_labels, train_hf, hf_threshold),
            "validation": binary_metrics(val_labels, val_hf, hf_threshold),
        },
        "probability_summary": {
            "validation_texture_mean": float(val_prob[val_labels == 1].mean()),
            "validation_non_texture_mean": float(val_prob[val_labels == 0].mean()),
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)}, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

