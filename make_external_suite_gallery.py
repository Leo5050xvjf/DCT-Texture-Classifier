from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack native comparison images into one review gallery")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3000)
    args = parser.parse_args()

    rows = []
    paths = sorted(args.input_dir.glob("*_comparison_native.jpg"))
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height = max(1, round(image.shape[0] * args.width / image.shape[1]))
        resized = cv2.resize(image, (args.width, height), interpolation=cv2.INTER_AREA)
        title_height = 44
        title = np.zeros((title_height, args.width, 3), dtype=np.uint8)
        cv2.putText(
            title, path.name.removesuffix("_comparison_native.jpg"),
            (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        rows.append(np.vstack((title, resized)))
    if not rows:
        raise RuntimeError(f"No native comparisons found under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gallery = np.vstack(rows)
    if not cv2.imwrite(str(args.output), gallery, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output.resolve()} ({gallery.shape[1]}x{gallery.shape[0]})")


if __name__ == "__main__":
    main()
