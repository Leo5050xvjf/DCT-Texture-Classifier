from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import dct_basis, image_files, labeled_panel, read_bgr, to_luma_u8
from dct_texture.map_model import MapGeneratorUNet
from dct_texture.model import DCTTextureClassifier
from dct_texture.teacher_map import teacher_maps_for_crops
from train_map_generator import map_metrics


def color_map(probability: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def summary(probability: np.ndarray) -> dict:
    return {
        "mean": float(probability.mean()),
        "std": float(probability.std()),
        "fraction_ge_050": float((probability >= 0.5).mean()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled clean/noisy comparison of direct teacher and G")
    parser.add_argument("--input-dir", type=Path, default=Path("data/diagnostics_v1"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--g-checkpoint", type=Path, default=Path("outputs/robust_v2/map_generator/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robust_v2/g_diagnostics"))
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher = DCTTextureClassifier(**teacher_checkpoint["model_config"]).to(device)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    g_checkpoint = torch.load(args.g_checkpoint, map_location=device, weights_only=False)
    generator_model = MapGeneratorUNet(g_checkpoint["base_channels"]).to(device)
    generator_model.load_state_dict(g_checkpoint["model"])
    generator_model.eval()
    basis = torch.from_numpy(dct_basis()).to(device)
    mean = torch.from_numpy(teacher_checkpoint["dct_mean"]).to(device)
    std = torch.from_numpy(teacher_checkpoint["dct_std"]).to(device)
    rng = np.random.default_rng(args.seed)
    records = []
    for path in image_files(args.input_dir):
        if "gaussian_noise" in path.stem:
            continue
        gray_u8 = to_luma_u8(read_bgr(path))
        clean = torch.from_numpy(gray_u8.astype(np.float32) / 255.0).to(device)
        target = teacher_maps_for_crops(clean[None], teacher, basis, mean, std)[0].cpu().numpy()
        image_dir = args.output_dir / path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        for sigma in (0.0, 15.0, 50.0):
            noise = rng.normal(0.0, sigma / 255.0, clean.shape).astype(np.float32)
            noisy_np = np.clip(clean.cpu().numpy() + noise, 0.0, 1.0)
            noisy = torch.from_numpy(noisy_np).to(device)
            direct = teacher_maps_for_crops(noisy[None], teacher, basis, mean, std)[0].cpu().numpy()
            prediction = torch.sigmoid(generator_model(noisy[None, None]))[0].cpu().numpy()
            record = {
                "image": path.stem,
                "sigma": sigma,
                "target": summary(target),
                "direct_noisy_teacher": summary(direct) | {"against_target": map_metrics(target, direct)},
                "map_generator_g": summary(prediction) | {"against_target": map_metrics(target, prediction)},
            }
            records.append(record)
            if sigma in (15.0, 50.0):
                original_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
                noisy_bgr = cv2.cvtColor(np.rint(noisy_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                panels = [
                    labeled_panel(original_bgr, "clean input"),
                    labeled_panel(noisy_bgr, f"noisy sigma={sigma:.0f}"),
                    labeled_panel(color_map(target), "clean teacher target"),
                    labeled_panel(color_map(direct), "clean teacher on noisy"),
                    labeled_panel(color_map(prediction), "map generator G"),
                ]
                cv2.imwrite(str(image_dir / f"sigma_{int(sigma):02d}_comparison.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
    result = {"device": str(device), "records": records}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

