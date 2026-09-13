from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np


MASK_PATTERN = re.compile(r"^mask_(?P<index>\d+)_(?P<stem>.+)\.png$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TextureSAM masks with the DCT CNN soft texture map")
    parser.add_argument("--probability-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probability_paths = sorted(args.probability_root.glob("*/texture_probability.npy"))
    if not probability_paths:
        raise RuntimeError(f"No texture_probability.npy files under {args.probability_root}")
    records: list[dict] = []
    compatibility_errors: list[str] = []
    for probability_path in probability_paths:
        stem = probability_path.parent.name
        probability = np.load(probability_path).astype(np.float32)
        matching = []
        for mask_path in args.mask_dir.glob(f"mask_*_{stem}.png"):
            match = MASK_PATTERN.match(mask_path.name)
            if match and match.group("stem") == stem:
                matching.append((int(match.group("index")), mask_path))
        matching.sort()
        if not matching:
            compatibility_errors.append(f"No TextureSAM masks for {stem}")
            continue
        stem_dir = args.output_dir / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        for index, mask_path in matching:
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                compatibility_errors.append(f"Unreadable mask: {mask_path}")
                continue
            if mask_image.shape != probability.shape:
                compatibility_errors.append(
                    f"Shape mismatch for {mask_path.name}: mask={mask_image.shape}, probability={probability.shape}"
                )
                continue
            mask = mask_image > 0
            area = int(mask.sum())
            values = probability[mask]
            if area:
                mean = float(values.mean())
                median = float(np.median(values))
                p90 = float(np.percentile(values, 90))
                fraction_ge_050 = float((values >= 0.5).mean())
            else:
                mean = median = p90 = fraction_ge_050 = float("nan")
            weighted = np.zeros(probability.shape, dtype=np.uint16)
            weighted[mask] = np.rint(np.clip(probability[mask], 0, 1) * 65535).astype(np.uint16)
            weighted_path = stem_dir / f"weighted_mask_{index}.png"
            cv2.imwrite(str(weighted_path), weighted)
            records.append(
                {
                    "image": stem,
                    "mask_index": index,
                    "mask_path": str(mask_path.resolve()),
                    "shape_match": True,
                    "area_pixels": area,
                    "area_fraction": area / probability.size,
                    "mean_texture_probability": mean,
                    "median_texture_probability": median,
                    "p90_texture_probability": p90,
                    "fraction_ge_050_inside_mask": fraction_ge_050,
                    "weighted_mask_path": str(weighted_path.resolve()),
                }
            )

    if records:
        with (args.output_dir / "mask_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "probability_root": str(args.probability_root.resolve()),
        "texturesam_mask_dir": str(args.mask_dir.resolve()),
        "compatible_masks": len(records),
        "errors": compatibility_errors,
        "all_shapes_compatible": not compatibility_errors,
        "records": records,
    }
    with (args.output_dir / "compatibility.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if compatibility_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

