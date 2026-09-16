#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import STATIONS, LabelMap
from tiger_models.runtime import atomic_json


def main() -> int:
    directory = ROOT / "artifacts/new_baselines/sam2_hiera_tiny_partial/engineering/stability/fold_0"
    history = json.loads((directory / "history.json").read_text())
    if len(history) != 10 or [row["epoch"] for row in history] != list(range(10)):
        raise RuntimeError("Stability history does not contain resumed epochs 0..9 exactly once")
    numeric_values = [
        value for row in history for value in row.values()
        if isinstance(value, (int, float))
    ]
    finite = all(math.isfinite(value) for value in numeric_values)
    loss_decreased = min(row["train_loss"] for row in history[-3:]) < history[0]["train_loss"]
    gradients_stable = max(row["gradient_norm_max"] for row in history) < 100.0
    if not (directory / "partial_run.json").is_file():
        raise RuntimeError("No intentional early-stop artifact; checkpoint resume was not exercised")

    labels = LabelMap.load(ROOT / "data/labelmap.csv")
    predicted_pixels = foreground_pixels = 0
    for path in (directory / "predictions/task1").glob("*.png"):
        prediction = labels.decode_fine(np.asarray(Image.open(path).convert("RGB")))
        predicted_pixels += prediction.size
        foreground_pixels += int((prediction != 0).sum())
    foreground_fraction = foreground_pixels / predicted_pixels
    segmentation_not_background = 0.001 < foreground_fraction < 0.999
    with (directory / "predictions/task3.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    probabilities = np.asarray([[float(row[station]) for station in STATIONS] for row in rows])
    visibility_not_constant = float(probabilities.std()) > 1e-4 and float(probabilities.std(axis=0).mean()) > 1e-5
    result = {
        "status": "PASS" if all((finite, loss_decreased, gradients_stable,
                                      segmentation_not_background, visibility_not_constant)) else "FAIL",
        "epochs": 10, "checkpoint_resume_exercised": True,
        "finite_metrics_and_losses": finite, "loss_decreased": loss_decreased,
        "first_train_loss": history[0]["train_loss"],
        "minimum_last_three_train_loss": min(row["train_loss"] for row in history[-3:]),
        "maximum_gradient_norm": max(row["gradient_norm_max"] for row in history),
        "segmentation_foreground_fraction": foreground_fraction,
        "segmentation_not_collapsed_to_background": segmentation_not_background,
        "visibility_probability_std": float(probabilities.std()),
        "visibility_not_constant": visibility_not_constant,
        "benchmark_eligible": False,
    }
    atomic_json(directory / "stability_check.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Stability gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
