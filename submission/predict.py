#!/usr/bin/env python3
"""Combined Task 1/2/3 entrypoint for TIGER SQ-AI 2026."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
from PIL import Image

from contract import validate_output
from model import (
    STATIONS,
    TASK1_PALETTE,
    TASK2_PALETTE,
    load_ensemble,
    predict_ensemble,
)


INPUT_DIR = Path(os.environ.get("TIGERSQAI_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("TIGERSQAI_OUTPUT_DIR", "/output"))
CHECKPOINT_DIR = Path(
    os.environ.get("TIGERSQAI_CHECKPOINT_DIR", "/opt/checkpoints")
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if os.environ.get("TIGERSQAI_REQUIRE_CUDA", "0") == "1":
        raise RuntimeError("CUDA is required but no NVIDIA GPU is available")
    print("warning: CUDA unavailable; falling back to slow CPU inference", flush=True)
    return torch.device("cpu")


def main() -> int:
    started = time.monotonic()
    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No PNG frames found in {INPUT_DIR}")

    task1_dir = OUTPUT_DIR / "task1"
    task2_dir = OUTPUT_DIR / "task2"
    task1_dir.mkdir(parents=True, exist_ok=True)
    task2_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "task3.csv"

    device = select_device()
    print(f"found {len(frames)} frames; inference device={device}", flush=True)
    models, heads = load_ensemble(CHECKPOINT_DIR, device)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case_id", *STATIONS])
        csv_file.flush()

        for index, frame_path in enumerate(frames, start=1):
            frame_started = time.monotonic()
            with Image.open(frame_path) as handle:
                image = handle.convert("RGB")
            task1_ids, task2_ids, station_scores = predict_ensemble(
                image, models, heads, device
            )

            Image.fromarray(TASK1_PALETTE[task1_ids]).save(
                task1_dir / frame_path.name, format="PNG"
            )
            Image.fromarray(TASK2_PALETTE[task2_ids]).save(
                task2_dir / frame_path.name, format="PNG"
            )
            writer.writerow(
                [frame_path.stem, *[f"{float(score):.8f}" for score in station_scores]]
            )
            csv_file.flush()
            print(
                f"[{index}/{len(frames)}] {frame_path.name} "
                f"({time.monotonic() - frame_started:.2f}s)",
                flush=True,
            )

    print("validating complete output contract", flush=True)
    validate_output(INPUT_DIR, OUTPUT_DIR)
    print(
        f"completed {len(frames)} frames in {(time.monotonic() - started) / 60:.2f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
