from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dct_texture.core import image_files, labeled_panel, read_bgr, resize_to_height


def heatmap(probability: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    u8 = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.resize(color, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare texture maps at multiple synthetic noise levels")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sigma0-root", type=Path, required=True)
    parser.add_argument("--sigma15-root", type=Path, required=True)
    parser.add_argument("--sigma50-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="G")
    parser.add_argument("--width", type=int, default=1800)
    args = parser.parse_args()

    rows = []
    for path in image_files(args.input_dir):
        original = resize_to_height(read_bgr(path), 520)
        size = (original.shape[1], original.shape[0])
        panels = [labeled_panel(original, "original")]
        for sigma, root in ((0, args.sigma0_root), (15, args.sigma15_root), (50, args.sigma50_root)):
            probability = np.load(root / path.stem / "texture_probability.npy")
            fraction = float((probability >= 0.5).mean())
            panels.append(
                labeled_panel(
                    heatmap(probability, size),
                    f"{args.model_label} sigma={sigma}, P>=0.5: {fraction:.1%}",
                )
            )
        row = np.hstack(panels)
        new_height = int(round(row.shape[0] * args.width / row.shape[1]))
        row = cv2.resize(row, (args.width, new_height), interpolation=cv2.INTER_AREA)
        title = np.zeros((46, args.width, 3), dtype=np.uint8)
        cv2.putText(title, path.stem, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([title, row]))
    if not rows:
        raise RuntimeError("No demo images found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
