from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dct_texture.core import dct_basis, image_files, read_bgr, to_luma_u8
from dct_texture.model import DCTTextureClassifier
from dct_texture.teacher_map import teacher_maps_for_crops


def sample_crops(image_dir: Path, crops_per_image: int, crop_size: int, seed: int):
    rng = np.random.default_rng(seed)
    crops = []
    names = []
    xs = []
    ys = []
    files = image_files(image_dir)
    for index, path in enumerate(files, start=1):
        gray = to_luma_u8(read_bgr(path))
        if min(gray.shape) < crop_size:
            raise RuntimeError(f"{path} is smaller than {crop_size}")
        for _ in range(crops_per_image):
            y = int(rng.integers(0, gray.shape[0] - crop_size + 1))
            x = int(rng.integers(0, gray.shape[1] - crop_size + 1))
            crops.append(gray[y : y + crop_size, x : x + crop_size].copy())
            names.append(path.name)
            xs.append(x)
            ys.append(y)
        if index % 100 == 0 or index == len(files):
            print(f"sampled {index}/{len(files)} images from {image_dir.name}", flush=True)
    return np.stack(crops).astype(np.uint8), np.asarray(names), np.asarray(xs), np.asarray(ys)


def build_split(
    image_dir: Path,
    output: Path,
    crops_per_image: int,
    crop_size: int,
    seed: int,
    teacher: DCTTextureClassifier,
    basis: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    map_batch_size: int,
) -> dict:
    crops, names, xs, ys = sample_crops(image_dir, crops_per_image, crop_size, seed)
    target_u16 = np.empty(crops.shape, dtype=np.uint16)
    for start in range(0, len(crops), map_batch_size):
        clean = torch.from_numpy(crops[start : start + map_batch_size].astype(np.float32) / 255.0).to(device)
        targets = teacher_maps_for_crops(clean, teacher, basis, mean, std)
        target_u16[start : start + len(clean)] = np.rint(targets.cpu().numpy() * 65535).astype(np.uint16)
        print(f"generated teacher maps {min(start + map_batch_size, len(crops))}/{len(crops)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, clean=crops, target_u16=target_u16, image_names=names, x=xs, y=ys)
    return {
        "image_dir": str(image_dir.resolve()),
        "images": len(image_files(image_dir)),
        "crops_per_image": crops_per_image,
        "crops": len(crops),
        "crop_size": crop_size,
        "target": "clean-image v1 DCT classifier, stride 1, overlap-averaged soft map",
        "target_probability_mean": float(target_u16.mean() / 65535.0),
        "target_fraction_ge_050": float((target_u16 >= 32768).mean()),
        "output": str(output.resolve()),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean-crop and dense clean-teacher-map data for G")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/paper_v1/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/map_generator_v2"))
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--train-crops-per-image", type=int, default=2)
    parser.add_argument("--val-crops-per-image", type=int, default=2)
    parser.add_argument("--map-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher = DCTTextureClassifier(**checkpoint["model_config"]).to(device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    basis = torch.from_numpy(dct_basis()).to(device)
    mean = torch.from_numpy(checkpoint["dct_mean"]).to(device)
    std = torch.from_numpy(checkpoint["dct_std"]).to(device)
    train = build_split(
        args.train_images, args.output_dir / "train.npz", args.train_crops_per_image,
        args.crop_size, args.seed, teacher, basis, mean, std, device, args.map_batch_size,
    )
    validation = build_split(
        args.val_images, args.output_dir / "val.npz", args.val_crops_per_image,
        args.crop_size, args.seed + 1, teacher, basis, mean, std, device, args.map_batch_size,
    )
    summary = {"train": train, "validation": validation}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

