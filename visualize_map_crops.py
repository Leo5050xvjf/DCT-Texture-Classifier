from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dct_texture.core import dct_basis, labeled_panel
from dct_texture.map_model import MapGeneratorUNet
from dct_texture.model import DCTTextureClassifier
from dct_texture.teacher_map import teacher_maps_for_crops


def color_map(probability: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clean target, direct noisy teacher, and G on held-out crops")
    parser.add_argument("--data", type=Path, default=Path("data/map_generator_v2/val.npz"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--g-checkpoint", type=Path, default=Path("outputs/robust_v2/map_generator/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robust_v2/map_generator/visuals"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    indices = np.linspace(0, len(data["clean"]) - 1, args.count, dtype=int)
    clean_u8 = data["clean"][indices]
    target = data["target_u16"][indices].astype(np.float32) / 65535.0
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
    clean = torch.from_numpy(clean_u8.astype(np.float32) / 255.0).to(device)
    rng = np.random.default_rng(args.seed)
    for sigma in (15.0, 50.0):
        noise = torch.from_numpy(rng.normal(0.0, sigma / 255.0, clean.shape).astype(np.float32)).to(device)
        noisy = (clean + noise).clamp(0.0, 1.0)
        direct = teacher_maps_for_crops(noisy, teacher, basis, mean, std).cpu().numpy()
        prediction = torch.sigmoid(generator_model(noisy[:, None])).cpu().numpy()
        rows = []
        for row_index, source_index in enumerate(indices):
            original = cv2.cvtColor(clean_u8[row_index], cv2.COLOR_GRAY2BGR)
            noisy_bgr = cv2.cvtColor(np.rint(noisy[row_index].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            panels = [
                labeled_panel(original, f"val crop {source_index}"),
                labeled_panel(noisy_bgr, f"noisy sigma={sigma:.0f}"),
                labeled_panel(color_map(target[row_index]), "clean target"),
                labeled_panel(color_map(direct[row_index]), "direct noisy DCT"),
                labeled_panel(color_map(prediction[row_index]), "G prediction"),
            ]
            row = np.hstack(panels)
            row = cv2.resize(row, (1600, int(round(row.shape[0] * 1600 / row.shape[1]))), interpolation=cv2.INTER_AREA)
            rows.append(row)
        cv2.imwrite(str(args.output_dir / f"validation_sigma_{int(sigma):02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote sigma {sigma:.0f} gallery")


if __name__ == "__main__":
    main()
