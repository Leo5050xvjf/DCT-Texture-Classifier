from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dct_texture.robust_models import make_robust_model
from infer_stcnn_stride1 import infer_stride1


def build_split(
    source: Path,
    output: Path,
    model: torch.nn.Module,
    kind: str,
    device: torch.device,
    batch_size: int,
) -> dict:
    data = np.load(source)
    clean = data["clean"]
    targets = np.empty(clean.shape, dtype=np.uint16)
    started = time.perf_counter()
    for index, image in enumerate(clean):
        probability, _ = infer_stride1(
            image, model, device, batch_size, sigma=0.0, seed=0, kind=kind
        )
        targets[index] = np.rint(probability * 65535.0).astype(np.uint16)
        if (index + 1) % 100 == 0 or index + 1 == len(clean):
            print(f"{source.name}: {index + 1}/{len(clean)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        clean=clean,
        target_u16=targets,
        image_names=data["image_names"],
        x=data["x"],
        y=data["y"],
    )
    return {
        "source": str(source), "output": str(output), "crops": len(clean),
        "crop_size": int(clean.shape[-1]),
        "target_probability_mean": float(targets.mean() / 65535.0),
        "target_fraction_ge_050": float((targets >= 32768).mean()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace old DCT dense targets with clean stride-1 STCNN targets")
    parser.add_argument("--source-dir", type=Path, default=Path("data/map_generator_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/map_generator_stcnn_v1"))
    parser.add_argument(
        "--teacher", type=Path,
        default=Path("checkpoints/stcnn_limit/stcnn_diverse_conditioned_oracle.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.teacher, map_location=device, weights_only=False)
    kind = checkpoint["kind"]
    model = make_robust_model(kind).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    started = time.perf_counter()
    splits = {
        split: build_split(
            args.source_dir / f"{split}.npz", args.output_dir / f"{split}.npz",
            model, kind, device, args.batch_size,
        )
        for split in ("train", "val")
    }
    payload = {
        "teacher_checkpoint": str(args.teacher), "teacher_kind": kind,
        "teacher_parameters": checkpoint.get("parameters"),
        "teacher_input": "clean luminance with exact sigma=0",
        "dense_method": "32x32 context at stride 1; central-8x8 scalar predictions overlap-averaged",
        "source_crops": "same image-disjoint clean crops as map_generator_v2; only targets replaced",
        "old_DCT_role": "none in these targets",
        "device": str(device), "elapsed_seconds": time.perf_counter() - started,
        "splits": splits,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
