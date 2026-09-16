#!/usr/bin/env python3
"""Finalize official Task 1 from frozen full40 fine OOF predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tiger_models.data import LabelMap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fine-prediction-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/p0_baseline_lock_v2/oof_predictions_native_ids",
    )
    parser.add_argument(
        "--fine-metrics", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/seed_2026/test_oof/oof_test_metrics.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "full_version/task1_merged_16cls/artifacts/full40_seed2026",
    )
    args = parser.parse_args()

    labelmap = LabelMap.load(ROOT / "data/labelmap.csv")
    expected = sorted(path.name for path in (ROOT / "data/images").glob("*.png"))
    available = sorted(path.name for path in args.fine_prediction_dir.glob("*.png"))
    if expected != available or len(expected) != 524:
        raise ValueError("Expected exactly 524 native fine OOF predictions matching the dataset")

    prediction_dir = args.output_dir / "oof_predictions_native_rgb"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name in expected:
        if (prediction_dir / name).is_file():
            continue
        with Image.open(args.fine_prediction_dir / name) as handle:
            fine_ids = np.asarray(handle.convert("L"), dtype=np.uint8)
        if int(fine_ids.max(initial=0)) >= len(labelmap.fine_to_coarse):
            raise ValueError(f"Fine prediction has an invalid ID: {name}")
        coarse_ids = labelmap.fine_to_coarse[fine_ids]
        Image.fromarray(labelmap.encode_coarse(coarse_ids), mode="RGB").save(prediction_dir / name)

    source = json.loads(args.fine_metrics.read_text())
    folds = []
    for row in source["folds"]:
        folds.append({
            "fold": int(row["fold"]),
            "cases": len(row["test_cases"]),
            "frames": int(row["test_frames"]),
            "dice": float(row["coarse_dice"]),
            "nhd": float(row["coarse_nhd"]),
            "task_score": float((row["coarse_dice"] + 1.0 - row["coarse_nhd"]) / 2.0),
        })
    overall_source = source["oof_case_aggregate"]
    overall = {
        "cases": int(overall_source["cases"]),
        "frames": int(overall_source["frames"]),
        "dice": float(overall_source["coarse_dice"]),
        "nhd": float(overall_source["coarse_nhd"]),
        "task_score": float((overall_source["coarse_dice"] + 1.0 - overall_source["coarse_nhd"]) / 2.0),
    }
    payload = {
        "status": "completed",
        "official_task": "task1_merged_segmentation_16_ids",
        "derivation": "deterministic fine-ID to merged-ID mapping from frozen Task-2 OOF predictions",
        "evaluation_grid": "512x896",
        "native_output_masks": len(expected),
        "source_metrics": str(args.fine_metrics),
        "folds": folds,
        "overall": overall,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oof_test_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "completion.json").write_text(json.dumps({
        "status": "completed", "folds": 5, "cases": 40, "frames": 524,
        "native_output_masks": 524,
    }, indent=2) + "\n")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
