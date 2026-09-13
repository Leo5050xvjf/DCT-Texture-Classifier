from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


SIGMAS = [0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sigma_value(by_sigma: dict, sigma: float, field: str) -> float:
    return float(by_sigma[str(sigma)][field])


def draw_accuracy_plot(curves: dict[str, list[float]], output: Path) -> None:
    width, height = 1500, 900
    left, right, top, bottom = 105, 420, 70, 105
    plot_right = width - right
    plot_bottom = height - bottom
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def point(index: int, accuracy: float) -> tuple[int, int]:
        x = left + int(round(index * (plot_right - left) / (len(SIGMAS) - 1)))
        y = plot_bottom - int(round((accuracy - 0.45) / 0.55 * (plot_bottom - top)))
        return x, y

    for accuracy in np.arange(0.5, 1.001, 0.1):
        y = point(0, float(accuracy))[1]
        cv2.line(canvas, (left, y), (plot_right, y), (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{accuracy:.1f}", (38, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 1, cv2.LINE_AA)
    for index, sigma in enumerate(SIGMAS):
        x, _ = point(index, 0.5)
        cv2.line(canvas, (x, top), (x, plot_bottom), (238, 238, 238), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{sigma:g}", (x - 12, plot_bottom + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (plot_right, plot_bottom), (40, 40, 40), 2)

    colors = [
        (35, 35, 35),
        (40, 130, 230),
        (35, 180, 65),
        (190, 90, 30),
        (160, 35, 180),
        (0, 170, 200),
        (220, 50, 80),
        (100, 100, 230),
        (60, 160, 160),
        (115, 75, 35),
    ]
    legend_x = plot_right + 35
    legend_y = top + 10
    for curve_index, (name, values) in enumerate(curves.items()):
        color = colors[curve_index % len(colors)]
        points = np.asarray([point(i, value) for i, value in enumerate(values)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
        y = legend_y + curve_index * 53
        cv2.line(canvas, (legend_x, y + 8), (legend_x + 43, y + 8), color, 4, cv2.LINE_AA)
        cv2.circle(canvas, (legend_x + 21, y + 8), 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, name, (legend_x + 57, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Validation accuracy vs. input AWGN", (left, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Noise sigma (8-bit scale)", (390, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Accuracy", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact robust-v2 result artifacts")
    parser.add_argument("--root", type=Path, default=Path("outputs/robust_v2"))
    args = parser.parse_args()
    root = args.root

    patch = load_json(root / "patch_models" / "summary.json")
    conditional = load_json(root / "conditional_accuracy_selected" / "summary.json")
    dense = load_json(root / "map_generator" / "metrics.json")
    g_patch = load_json(root / "map_generator" / "patch_label_metrics.json")
    diagnostics = load_json(root / "g_diagnostics" / "metrics.json")
    texturesam = load_json(root / "map_generator_texturesam_compatibility" / "compatibility.json")
    external_manifest = load_json(root / "map_generator_external_grass" / "manifest.json")

    wanted_patch = [
        "dct8_clean_only_v1",
        "dct8_fixed15",
        "dct8_fixed25",
        "dct8_fixed50",
        "dct8_mixed",
        "spatial8_mixed",
        "spatial32_mixed",
        "hybrid32_mixed",
    ]
    curves: dict[str, list[float]] = {}
    compact_models: dict[str, dict] = {}
    for name in wanted_patch:
        result = patch["models"][name]
        curves[name] = [sigma_value(result["by_sigma"], sigma, "accuracy") for sigma in SIGMAS]
        compact_models[name] = {
            "parameters": result["parameters"],
            "accuracy_by_sigma": dict(zip([str(s) for s in SIGMAS], curves[name])),
            "mean_accuracy": float(np.mean(curves[name])),
        }

    cond_result = conditional["models"]["dct8_mixed_conditional"]
    cond_name = "dct8_mixed_conditional_selected_by_accuracy"
    curves[cond_name] = [sigma_value(cond_result["by_sigma"], sigma, "accuracy") for sigma in SIGMAS]
    compact_models[cond_name] = {
        "parameters": cond_result["parameters"],
        "accuracy_by_sigma": dict(zip([str(s) for s in SIGMAS], curves[cond_name])),
        "mean_accuracy": float(np.mean(curves[cond_name])),
    }

    curves["map_generator_G_central8"] = [sigma_value(g_patch["by_sigma"], sigma, "accuracy") for sigma in SIGMAS]
    compact_models["map_generator_G_central8"] = {
        "parameters": dense["model"]["parameters"],
        "accuracy_by_sigma": dict(zip([str(s) for s in SIGMAS], curves["map_generator_G_central8"])),
        "mean_accuracy": float(np.mean(curves["map_generator_G_central8"])),
        "note": "Central 8x8 mean of a dense map; different training objective from patch classifiers.",
    }

    dense_compact = {}
    for sigma in SIGMAS:
        record = dense["by_sigma"][str(sigma)]
        dense_compact[str(sigma)] = {
            "G": {key: record["map_generator_g"][key] for key in ("rmse", "pearson", "binary_accuracy", "prediction_mean")},
            "direct_noisy_DCT_teacher": {
                key: record["clean_teacher_applied_directly_to_noisy"][key]
                for key in ("rmse", "pearson", "binary_accuracy", "prediction_mean")
            },
        }

    diagnostic_rows = []
    for record in diagnostics["records"]:
        if float(record["sigma"]) not in (0.0, 15.0, 50.0):
            continue
        diagnostic_rows.append(
            {
                "image": record["image"],
                "sigma": float(record["sigma"]),
                "target_mean": float(record["target"]["mean"]),
                "target_fraction_high": float(record["target"]["fraction_ge_050"]),
                "direct_mean": float(record["direct_noisy_teacher"]["mean"]),
                "G_mean": float(record["map_generator_g"]["mean"]),
                "G_fraction_high": float(record["map_generator_g"]["fraction_ge_050"]),
                "G_binary_accuracy": float(record["map_generator_g"]["against_target"]["binary_accuracy"]),
                "G_pearson": float(record["map_generator_g"]["against_target"]["pearson"]),
            }
        )

    summary = {
        "protocol": {
            "patch_train_samples": patch["protocol"]["train_samples"],
            "patch_validation_samples": patch["protocol"]["validation_samples"],
            "patch_noise": patch["protocol"]["noise"],
            "map_train_crops": dense["protocol"]["train_crops"],
            "map_validation_crops": dense["protocol"]["validation_crops"],
            "map_crop_size": dense["protocol"]["crop_size"],
            "map_target": dense["protocol"]["target"],
        },
        "patch_screen": compact_models,
        "dense_map": dense_compact,
        "controlled_diagnostics": diagnostic_rows,
        "full_resolution_inference": {
            "resize": external_manifest["resize"],
            "images": len(external_manifest["images"]),
            "tile_size": external_manifest["tile_size"],
            "overlap": external_manifest["overlap"],
            "total_seconds": external_manifest["total_seconds"],
        },
        "texturesam_compatibility": {
            "compatible_masks": texturesam["compatible_masks"],
            "all_shapes_compatible": texturesam["all_shapes_compatible"],
            "errors": texturesam["errors"],
        },
    }
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    fieldnames = list(diagnostic_rows[0])
    with (root / "controlled_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    draw_accuracy_plot(curves, root / "accuracy_vs_noise.png")
    print(f"wrote {(root / 'summary.json').resolve()}")
    print(f"wrote {(root / 'controlled_diagnostics.csv').resolve()}")
    print(f"wrote {(root / 'accuracy_vs_noise.png').resolve()}")


if __name__ == "__main__":
    main()
