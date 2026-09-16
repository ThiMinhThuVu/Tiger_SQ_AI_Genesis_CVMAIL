#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models import MultitaskSAM
from tiger_models.data import STATIONS, LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.runtime import atomic_json, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("Prediction requires a CUDA execution node unless --allow-cpu is explicit")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    precision = args.mixed_precision or config.get("precision_selected", "bf16")
    if device.type != "cuda":
        precision = "fp32"
    model = MultitaskSAM(
        checkpoint_path=config["checkpoint"],
        finetune=config["finetune"],
        gradient_checkpointing=config["gradient_checkpointing"],
        coarse_head=False,
        config_name=config.get("sam2_config", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    records = records_for_fold(args.data_root, args.fold, args.split)
    dataset = TigerMultitaskDataset(
        records, labelmap, int(config["height"]), int(config["width"]), training=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    task1 = args.output_dir / "task1"
    task2 = args.output_dir / "task2"
    entropy_dir = args.output_dir / "entropy"
    for directory in (task1, task2, entropy_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if any(task1.iterdir()) or any(task2.iterdir()):
        raise FileExistsError("Prediction directories are non-empty; refusing to overwrite")

    rows: list[list[object]] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            context = nullcontext()
            if device.type == "cuda" and precision in {"bf16", "fp16"}:
                context = torch.autocast(
                    "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with context:
                output = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(1000 * (time.perf_counter() - start))
            fine_probability = output["fine_logits"].float().softmax(1)
            visibility_probability = output["visibility_logits"].float().sigmoid()
            if not torch.isfinite(fine_probability).all() or not torch.isfinite(visibility_probability).all():
                raise FloatingPointError("Non-finite prediction probability")
            fine_id = fine_probability.argmax(1)[0].cpu().numpy().astype(np.uint8)
            entropy = (
                -(fine_probability * fine_probability.clamp_min(1e-8).log()).sum(1)
                / np.log(31)
            )[0].cpu().numpy().astype(np.float16)
            name = batch["name"][0]
            original_height, original_width = (int(x) for x in batch["original_size"][0])
            fine_native = np.asarray(
                Image.fromarray(fine_id, mode="L").resize(
                    (original_width, original_height), Image.Resampling.NEAREST
                )
            )
            coarse_native = labelmap.fine_to_coarse[fine_native]
            Image.fromarray(labelmap.encode_fine(fine_native), mode="RGB").save(task1 / name)
            Image.fromarray(labelmap.encode_coarse(coarse_native), mode="RGB").save(task2 / name)
            np.save(entropy_dir / f"{Path(name).stem}.npy", entropy)
            rows.append([Path(name).stem, *visibility_probability[0].cpu().tolist()])

    with (args.output_dir / "task3.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", *STATIONS])
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint), "fold": args.fold, "split": args.split,
        "frames": len(rows),
        "oracle_prompt": False, "coarse_strategy": "derived_from_fine",
        "mixed_precision": precision,
        "inference_ms_per_image_mean": float(np.mean(latencies)),
        "inference_ms_per_image_median": float(np.median(latencies)),
    }
    atomic_json(args.output_dir / "prediction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
