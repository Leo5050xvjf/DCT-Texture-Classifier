from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def save_gray(output: Path, name: str, image: np.ndarray) -> None:
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output / f"{name}.png"), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled texture/edge/noise diagnostics")
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics_v1"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = args.size
    yy, xx = np.mgrid[:n, :n]
    rng = np.random.default_rng(args.seed)

    save_gray(args.output_dir, "01_flat", np.full((n, n), 128.0))

    single_edge = np.full((n, n), 64.0)
    single_edge[:, n // 2 :] = 192.0
    save_gray(args.output_dir, "02_single_edge", single_edge)

    gradient = np.tile(np.linspace(32, 224, n, dtype=np.float32), (n, 1))
    save_gray(args.output_dir, "03_smooth_gradient", gradient)

    checker = (((xx // 4 + yy // 4) % 2) * 128 + 64).astype(np.float32)
    save_gray(args.output_dir, "04_checker_4px", checker)

    low_contrast = 128.0 + 8.0 * np.sin(2 * np.pi * xx / 12.0) + 8.0 * np.sin(2 * np.pi * yy / 17.0)
    save_gray(args.output_dir, "05_low_contrast_texture", low_contrast)

    noise15 = 128.0 + rng.normal(0.0, 15.0, size=(n, n))
    save_gray(args.output_dir, "06_gaussian_noise_sigma15", noise15)

    text_lines = np.full((n, n), 230, dtype=np.uint8)
    for y in range(35, n, 55):
        cv2.putText(text_lines, "EDGE 123 ABC", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 30, 2, cv2.LINE_AA)
    for x in range(10, n, 70):
        cv2.line(text_lines, (x, 0), (min(n - 1, x + 100), n - 1), 30, 1, cv2.LINE_AA)
    save_gray(args.output_dir, "07_text_and_lines", text_lines)

    # Same low-contrast texture with a smooth contrast ramp: useful for checking
    # whether the output measures texture structure or mostly its amplitude.
    amplitude = np.linspace(0.0, 20.0, n, dtype=np.float32)[None, :]
    ramp_texture = 128.0 + amplitude * np.sin(2 * np.pi * yy / 10.0)
    save_gray(args.output_dir, "08_texture_contrast_ramp", ramp_texture)

    print(f"wrote 8 diagnostic images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
