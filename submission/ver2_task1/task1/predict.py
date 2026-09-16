#!/usr/bin/env python3
"""Standalone Task 1 entrypoint using the corrected D517 ensemble."""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

from model import TASK1_PALETTE, load_task1_ensemble, predict_task1
from validate import validate_output

INPUT_DIR = Path(os.environ.get("TIGERSQAI_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("TIGERSQAI_OUTPUT_DIR", "/output"))
CHECKPOINT_DIR = Path(os.environ.get("TIGERSQAI_CHECKPOINT_DIR", "/opt/checkpoints/task1"))


def main() -> int:
    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No PNG frames found in {INPUT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    task_dir = OUTPUT_DIR / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    models = load_task1_ensemble(CHECKPOINT_DIR, device)
    started = time.monotonic()
    for index, frame_path in enumerate(frames, 1):
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
        ids = predict_task1(image, models, device)
        Image.fromarray(TASK1_PALETTE[ids]).save(task_dir / frame_path.name)
        print(f"[{index}/{len(frames)}] {frame_path.name}", flush=True)
    validate_output(INPUT_DIR, OUTPUT_DIR)
    print(f"Task 1 D517 completed in {(time.monotonic()-started)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
