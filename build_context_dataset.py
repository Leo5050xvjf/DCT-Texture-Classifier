from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from dct_texture.core import read_bgr, to_luma_u8


def extract_context(gray: np.ndarray, x: int, y: int, context_size: int = 32, target_size: int = 8) -> np.ndarray:
    center_x = x + target_size // 2
    center_y = y + target_size // 2
    half = context_size // 2
    x0, y0 = center_x - half, center_y - half
    x1, y1 = x0 + context_size, y0 + context_size
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - gray.shape[1]), max(0, y1 - gray.shape[0])
    if left or top or right or bottom:
        gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        x0 += left
        x1 += left
        y0 += top
        y1 += top
    context = gray[y0:y1, x0:x1]
    if context.shape != (context_size, context_size):
        raise RuntimeError(f"Bad context shape {context.shape}")
    return context.copy()


def build(metadata_path: Path, patch_data_path: Path, image_dir: Path, output_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    original = np.load(patch_data_path)
    labels = original["labels"].astype(np.uint8)
    patches = original["patches"].astype(np.uint8)
    if len(records) != len(labels):
        raise RuntimeError(f"Metadata count {len(records)} != patch count {len(labels)}")

    contexts = np.empty((len(records), 32, 32), dtype=np.uint8)
    current_name = None
    gray = None
    mismatch = 0
    for index, record in enumerate(records):
        if record["image"] != current_name:
            current_name = record["image"]
            gray = to_luma_u8(read_bgr(image_dir / current_name))
        x, y = int(record["x"]), int(record["y"])
        contexts[index] = extract_context(gray, x, y)
        if not np.array_equal(contexts[index, 12:20, 12:20], patches[index]):
            mismatch += 1
    if mismatch:
        raise RuntimeError(f"Central 8x8 patch mismatches: {mismatch}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, contexts=contexts, labels=labels)
    return {
        "metadata": str(metadata_path.resolve()),
        "patch_data": str(patch_data_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "output": str(output_path.resolve()),
        "samples": len(labels),
        "context_size": 32,
        "target_size": 8,
        "center_patch_mismatches": mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover 32x32 spatial contexts around v1 8x8 pseudo-labelled patches")
    parser.add_argument("--paper-data", type=Path, default=Path("data/paper_v1"))
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/robust_v2"))
    args = parser.parse_args()
    train = build(
        args.paper_data / "train_metadata.csv", args.paper_data / "train.npz",
        args.train_images, args.output_dir / "train_contexts.npz",
    )
    validation = build(
        args.paper_data / "val_metadata.csv", args.paper_data / "val.npz",
        args.val_images, args.output_dir / "val_contexts.npz",
    )
    summary = {"train": train, "validation": validation}
    (args.output_dir / "context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

