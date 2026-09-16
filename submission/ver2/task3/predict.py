#!/usr/bin/env python3
"""Standalone Task 3 P0/new-CSV entrypoint."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
from PIL import Image

from model import STATIONS, load_task3_ensemble, predict_task3
from validate import validate_output

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
    models, heads = load_task3_ensemble(
        CHECKPOINT_DIR / "encoder", CHECKPOINT_DIR / "head", device
    )
    started = time.monotonic()
    with (OUTPUT_DIR / "task3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", *STATIONS])
        for index, frame_path in enumerate(frames, 1):
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
            scores = predict_task3(image, models, heads, device)
            writer.writerow([frame_path.stem, *[f"{float(value):.8f}" for value in scores]])
            handle.flush()
            print(f"[{index}/{len(frames)}] {frame_path.name}", flush=True)
    validate_output(INPUT_DIR, OUTPUT_DIR)
    print(f"Task 3 P0 ver2 completed in {(time.monotonic()-started)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
