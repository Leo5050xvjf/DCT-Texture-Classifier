from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack inference comparisons into a compact review gallery")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--first-panels", type=int, default=3)
    parser.add_argument("--total-panels", type=int, default=5)
    parser.add_argument(
        "--pattern",
        default="comparison.jpg",
        help="Filename pattern inside each case directory (for example sigma_15_comparison.jpg)",
    )
    args = parser.parse_args()
    rows = []
    for comparison_path in sorted(args.input_root.glob(f"*/{args.pattern}")):
        image = cv2.imread(str(comparison_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        panel_width = image.shape[1] // args.total_panels
        image = image[:, : panel_width * args.first_panels]
        new_height = int(round(image.shape[0] * args.width / image.shape[1]))
        image = cv2.resize(image, (args.width, new_height), interpolation=cv2.INTER_AREA)
        title = np.zeros((42, args.width, 3), dtype=np.uint8)
        cv2.putText(title, comparison_path.parent.name, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([title, image]))
    if not rows:
        raise RuntimeError(f"No comparison images under {args.input_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
