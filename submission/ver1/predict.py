#!/usr/bin/env python3
"""Combined Task 1/2/3 entrypoint for center-stratified ver1."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
from PIL import Image

from contract import validate_output
from model import STATIONS, TASK1_PALETTE, TASK2_PALETTE, load_ensemble, predict_ensemble

INPUT_DIR = Path(os.environ.get("TIGERSQAI_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("TIGERSQAI_OUTPUT_DIR", "/output"))
CHECKPOINT_DIR = Path(os.environ.get("TIGERSQAI_CHECKPOINT_DIR", "/opt/checkpoints"))


def main() -> int:
    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No PNG frames found in {INPUT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    (OUTPUT_DIR / "task1").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "task2").mkdir(parents=True, exist_ok=True)
    task1_models, task2_models, heads = load_ensemble(CHECKPOINT_DIR, device)
    started = time.monotonic()
    with (OUTPUT_DIR / "task3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["case_id", *STATIONS])
        for index, frame_path in enumerate(frames, 1):
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
            task1, task2, visibility = predict_ensemble(image, task1_models, task2_models, heads, device)
            Image.fromarray(TASK1_PALETTE[task1]).save(OUTPUT_DIR / "task1" / frame_path.name)
            Image.fromarray(TASK2_PALETTE[task2]).save(OUTPUT_DIR / "task2" / frame_path.name)
            writer.writerow([frame_path.stem, *[f"{float(value):.8f}" for value in visibility]])
            handle.flush()
            print(f"[{index}/{len(frames)}] {frame_path.name}", flush=True)
    validate_output(INPUT_DIR, OUTPUT_DIR)
    print(f"ver1 completed in {(time.monotonic()-started)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
