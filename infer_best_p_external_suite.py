from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import image_files, read_bgr, to_luma_u8
from dct_texture.robust_models import make_robust_model
from infer_stcnn_stride1 import infer_stride1


def annotate(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    scale = max(0.55, min(2.2, height / 850.0))
    thickness = max(1, round(scale * 2))
    banner_height = max(34, round(48 * scale))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, round(width * 0.82)), banner_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.82, output, 0.18, 0.0)
    cv2.putText(
        output, text, (max(8, round(12 * scale)), round(32 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return output


def heatmap_with_legend(probability: np.ndarray, title: str) -> np.ndarray:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    output = annotate(output, title + " | blue/purple=0 -> red=1")
    height, width = output.shape[:2]
    bar_width = max(160, min(round(width * 0.30), 900))
    bar_height = max(14, round(height * 0.018))
    margin = max(12, round(height * 0.018))
    x0 = margin
    y0 = height - margin - bar_height
    gradient = np.linspace(0, 255, bar_width, dtype=np.uint8)[None, :]
    gradient = cv2.resize(gradient, (bar_width, bar_height), interpolation=cv2.INTER_LINEAR)
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    pad = max(4, round(bar_height * 0.35))
    cv2.rectangle(
        output, (x0 - pad, y0 - pad),
        (min(width - 1, x0 + bar_width + pad), min(height - 1, y0 + bar_height + pad)),
        (0, 0, 0), -1,
    )
    output[y0 : y0 + bar_height, x0 : x0 + bar_width] = gradient
    font_scale = max(0.45, min(1.2, height / 1200.0))
    thickness = max(1, round(font_scale * 2))
    label_y = max(round(18 * font_scale), y0 - pad - 3)
    cv2.putText(output, "0", (x0, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    one_size = cv2.getTextSize("1", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.putText(output, "1", (x0 + bar_width - one_size[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


def save_probability(output_dir: Path, sigma: float, probability: np.ndarray, noisy: np.ndarray) -> None:
    suffix = f"sigma_{sigma:g}"
    np.save(output_dir / f"probability_{suffix}.npy", probability)
    cv2.imwrite(
        str(output_dir / f"probability_{suffix}_u16.png"),
        np.rint(np.clip(probability, 0.0, 1.0) * 65535.0).astype(np.uint16),
    )
    cv2.imwrite(
        str(output_dir / f"mask_{suffix}_t050.png"),
        (probability >= 0.5).astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(output_dir / f"input_{suffix}.png"),
        np.rint(np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Native-resolution best-P clean/noise suite")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/stcnn_limit/stcnn_diverse_conditioned_oracle.pt"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/stcnn_stride1/external_grass_best_p_suite"),
    )
    parser.add_argument(
        "--result-root", type=Path,
        default=Path("results/stcnn_stride1/external_grass_best_p_suite"),
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 15.0, 50.0])
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--model-label", default="best P")
    args = parser.parse_args()
    if 0.0 not in args.sigmas:
        raise ValueError("--sigmas must contain 0 for the clean overlay")

    files = image_files(args.input_dir)
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    kind = checkpoint["kind"]
    model = make_robust_model(kind).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    records = []
    suite_started = time.perf_counter()

    for image_index, path in enumerate(files, start=1):
        image_started = time.perf_counter()
        bgr = read_bgr(path)
        gray = to_luma_u8(bgr)
        height, width = gray.shape
        output_dir = args.output_root / path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        panels = [annotate(bgr, f"original | native {width}x{height}")]
        predictions = {}
        conditions = []
        for sigma in args.sigmas:
            condition_started = time.perf_counter()
            probability, noisy = infer_stride1(
                gray, model, device, args.batch_size, sigma,
                args.seed + image_index, kind=kind,
            )
            predictions[sigma] = probability
            save_probability(output_dir, sigma, probability, noisy)
            fraction = float((probability >= 0.5).mean())
            panels.append(
                heatmap_with_legend(
                    probability, f"{args.model_label} | input AWGN sigma={sigma:g} | >=0.5: {fraction:.1%}"
                )
            )
            conditions.append(
                {
                    "sigma": sigma, "fraction_ge_050": fraction,
                    "probability_mean": float(probability.mean()),
                    "elapsed_seconds": time.perf_counter() - condition_started,
                }
            )
        clean_heatmap = cv2.applyColorMap(
            np.rint(np.clip(predictions[0.0], 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        overlay = cv2.addWeighted(bgr, 0.58, clean_heatmap, 0.42, 0.0)
        overlay = annotate(overlay, "clean-result overlay | blue/purple=0 -> red=1")
        cv2.imwrite(str(output_dir / "overlay_clean.png"), overlay)
        panels.append(overlay)
        comparison = np.hstack(panels)
        result_path = args.result_root / f"{path.stem}_comparison_native.jpg"
        cv2.imwrite(str(result_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
        records.append(
            {
                "image": path.name, "source": str(path.resolve()),
                "native_height": height, "native_width": width,
                "comparison_height": int(comparison.shape[0]),
                "comparison_width": int(comparison.shape[1]),
                "conditions": conditions, "overlay_source_sigma": 0.0,
                "comparison": str(result_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": time.perf_counter() - image_started,
            }
        )
        print(
            f"[{image_index}/{len(files)}] {path.name} {width}x{height} "
            f"done in {records[-1]['elapsed_seconds']:.2f}s",
            flush=True,
        )

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "kind": kind,
        "parameters": checkpoint.get("parameters"), "device": str(device),
        "model_label": args.model_label,
        "model_receives_sigma": kind == "spatial32_conditional",
        "input_dir": str(args.input_dir.resolve()), "sigmas": args.sigmas,
        "resize": False,
        "panel_order": [
            "original", *[f"P map for AWGN sigma={sigma:g}" for sigma in args.sigmas],
            "overlay of clean sigma=0 result",
        ],
        "color_scale": "OpenCV TURBO: blue/purple=0, red=1",
        "elapsed_seconds": time.perf_counter() - suite_started,
        "images": records,
    }
    (args.result_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote {(args.result_root / 'manifest.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
