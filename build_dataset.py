from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from dct_texture.core import (
    dct_2d_numpy,
    extreme_windows,
    high_frequency_ratio,
    image_files,
    patch_sobel_scores,
    read_bgr,
    sobel_magnitude,
    to_luma_u8,
)


def allocate_counts(num_images: int, total_pairs: int, rng: np.random.Generator) -> np.ndarray:
    if total_pairs < num_images:
        selected = rng.choice(num_images, size=total_pairs, replace=False)
        counts = np.zeros(num_images, dtype=np.int64)
        counts[selected] = 1
        return counts
    counts = np.full(num_images, total_pairs // num_images, dtype=np.int64)
    remainder = total_pairs - int(counts.sum())
    if remainder:
        counts[rng.choice(num_images, size=remainder, replace=False)] += 1
    return counts


def boxes_overlap_iou(a: tuple[int, int], b: tuple[int, int], size: int) -> float:
    ax, ay = a
    bx, by = b
    iw = max(0, min(ax + size, bx + size) - max(ax, bx))
    ih = max(0, min(ay + size, by + size) - max(ay, by))
    intersection = iw * ih
    return intersection / max(2 * size * size - intersection, 1)


def save_region_audit(bgr: np.ndarray, high: tuple[int, int], low: tuple[int, int], size: int, output: Path) -> None:
    display = bgr.copy()
    hx, hy = high
    lx, ly = low
    cv2.rectangle(display, (hx, hy), (hx + size - 1, hy + size - 1), (0, 0, 255), 5)
    cv2.rectangle(display, (lx, ly), (lx + size - 1, ly + size - 1), (255, 0, 0), 5)
    cv2.putText(display, "HIGH = texture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(display, "LOW = non-texture", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3, cv2.LINE_AA)
    scale = min(1.0, 1400.0 / max(display.shape[:2]))
    if scale < 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output), display)


def save_patch_contact(pairs: list[tuple[np.ndarray, np.ndarray]], output: Path, max_pairs: int = 48) -> None:
    pairs = pairs[:max_pairs]
    scale = 8
    cell = 8 * scale
    columns = 8
    rows = int(np.ceil(len(pairs) / columns))
    canvas = np.full((rows * (cell * 2 + 28), columns * (cell + 8), 3), 245, dtype=np.uint8)
    for index, (high, low) in enumerate(pairs):
        row, column = divmod(index, columns)
        x = column * (cell + 8)
        y = row * (cell * 2 + 28)
        high_big = cv2.resize(high, (cell, cell), interpolation=cv2.INTER_NEAREST)
        low_big = cv2.resize(low, (cell, cell), interpolation=cv2.INTER_NEAREST)
        canvas[y : y + cell, x : x + cell] = cv2.cvtColor(high_big, cv2.COLOR_GRAY2BGR)
        canvas[y + cell + 20 : y + 2 * cell + 20, x : x + cell] = cv2.cvtColor(low_big, cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, "T", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "N", (x + 2, y + cell + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite(str(output), canvas)


def build_split(
    source: Path,
    output: Path,
    split: str,
    total_pairs: int,
    seed: int,
    region_size: int,
    patch_size: int,
    audit_images: int,
) -> dict:
    files = image_files(source)
    if not files:
        raise RuntimeError(f"No images found in {source}")
    rng = np.random.default_rng(seed)
    counts = allocate_counts(len(files), total_pairs, rng)
    audit_indices = set(np.linspace(0, len(files) - 1, min(audit_images, len(files)), dtype=int).tolist())
    audit_dir = output / "audit" / split
    audit_dir.mkdir(parents=True, exist_ok=True)

    patches: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict] = []
    audit_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    image_stats: list[dict] = []

    for image_index, (path, pair_count) in enumerate(zip(files, counts)):
        if pair_count == 0:
            continue
        bgr = read_bgr(path)
        gray = to_luma_u8(bgr)
        gradient = sobel_magnitude(gray)
        high, low, high_score, low_score = extreme_windows(gradient, region_size)
        overlap_iou = boxes_overlap_iou(high, low, region_size)
        image_stats.append(
            {
                "image": path.name,
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                "high_region_score": high_score,
                "low_region_score": low_score,
                "score_ratio": high_score / max(low_score, 1e-12),
                "high_low_iou": overlap_iou,
            }
        )
        if image_index in audit_indices:
            save_region_audit(bgr, high, low, region_size, audit_dir / f"{path.stem}_regions.jpg")

        for pair_index in range(int(pair_count)):
            offsets = rng.integers(0, region_size - patch_size + 1, size=(2, 2))
            pair_values = []
            for label, origin, offset in ((1, high, offsets[0]), (0, low, offsets[1])):
                origin_x, origin_y = origin
                offset_x, offset_y = int(offset[0]), int(offset[1])
                x, y = origin_x + offset_x, origin_y + offset_y
                patch = gray[y : y + patch_size, x : x + patch_size].copy()
                if patch.shape != (patch_size, patch_size):
                    raise RuntimeError(f"Bad patch at {path}:{x},{y}")
                patches.append(patch)
                labels.append(label)
                records.append(
                    {
                        "split": split,
                        "image": path.name,
                        "pair_index": pair_index,
                        "label": label,
                        "x": x,
                        "y": y,
                        "region_x": origin_x,
                        "region_y": origin_y,
                        "region_gradient_mean": high_score if label else low_score,
                    }
                )
                pair_values.append(patch)
            if len(audit_pairs) < 96:
                audit_pairs.append((pair_values[0], pair_values[1]))

        if (image_index + 1) % 100 == 0 or image_index + 1 == len(files):
            print(f"[{split}] processed {image_index + 1}/{len(files)} images", flush=True)

    patch_array = np.stack(patches).astype(np.uint8)
    label_array = np.asarray(labels, dtype=np.uint8)
    dct = dct_2d_numpy(patch_array.astype(np.float32) / 255.0)
    sobel_scores = patch_sobel_scores(patch_array)
    hf_scores = high_frequency_ratio(dct).astype(np.float32)
    np.savez_compressed(
        output / f"{split}.npz",
        patches=patch_array,
        labels=label_array,
        sobel_scores=sobel_scores,
        dct_hf_ratio=hf_scores,
    )
    with (output / f"{split}_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_patch_contact(audit_pairs, audit_dir / "patch_pairs.jpg")

    positives = label_array == 1
    summary = {
        "split": split,
        "source": str(source.resolve()),
        "images": len(files),
        "pairs": total_pairs,
        "patches": len(label_array),
        "texture_patches": int(positives.sum()),
        "non_texture_patches": int((~positives).sum()),
        "region_size": region_size,
        "patch_size": patch_size,
        "seed": seed,
        "region_score": {
            "high_mean": float(np.mean([item["high_region_score"] for item in image_stats])),
            "low_mean": float(np.mean([item["low_region_score"] for item in image_stats])),
            "median_ratio": float(np.median([item["score_ratio"] for item in image_stats])),
            "overlapping_region_count": int(sum(item["high_low_iou"] > 0 for item in image_stats)),
            "max_iou": float(max(item["high_low_iou"] for item in image_stats)),
        },
        "patch_score": {
            "texture_sobel_mean": float(sobel_scores[positives].mean()),
            "non_texture_sobel_mean": float(sobel_scores[~positives].mean()),
            "texture_dct_hf_mean": float(hf_scores[positives].mean()),
            "non_texture_dct_hf_mean": float(hf_scores[~positives].mean()),
        },
    }
    with (output / f"{split}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "images": image_stats}, handle, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style Sobel/DCT pseudo-label datasets")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/paper_v1"))
    parser.add_argument("--train-pairs", type=int, default=3000)
    parser.add_argument("--val-pairs", type=int, default=1000)
    parser.add_argument("--region-size", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--audit-images", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size != 8:
        raise ValueError("The paper-style v1 classifier is fixed to 8x8 patches")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = build_split(
        args.train_dir, args.output_dir, "train", args.train_pairs, args.seed,
        args.region_size, args.patch_size, args.audit_images,
    )
    val_summary = build_split(
        args.val_dir, args.output_dir, "val", args.val_pairs, args.seed + 1,
        args.region_size, args.patch_size, args.audit_images,
    )
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": train_summary, "validation": val_summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"train": train_summary, "validation": val_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()

