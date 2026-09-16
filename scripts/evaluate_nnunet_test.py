#!/usr/bin/env python3
"""Evaluate nnU-Net test predictions with weighted TIGER segmentation metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiger_models.data import LabelMap, records_for_fold
from tiger_models.metrics import segmentation_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def load_prediction(path: Path) -> np.ndarray:
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected singleton 2D NIfTI at {path}, got {array.shape}")
    return array.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    folds = []
    for fold in range(5):
        prediction_dir = args.prediction_root / f"fold_{fold}" / "test_predictions"
        if not prediction_dir.is_dir():
            continue
        predictions, targets, cases = [], [], []
        for record in records_for_fold(args.data_root, fold, "test"):
            prediction_path = prediction_dir / f"{Path(record.name).stem}.nii.gz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Missing prediction for fold {fold}: {prediction_path}")
            prediction = load_prediction(prediction_path)
            target = labelmap.decode_fine(np.asarray(Image.open(record.fine_mask).convert("RGB")))
            if prediction.shape != target.shape:
                raise ValueError(f"Shape mismatch for {record.name}: {prediction.shape} vs {target.shape}")
            if prediction.min(initial=0) < 0 or prediction.max(initial=0) >= 31:
                raise ValueError(f"Invalid label IDs in {prediction_path}")
            predictions.append(prediction)
            targets.append(target)
            cases.append(record.case_id)
        fine = segmentation_metrics(predictions, targets, cases, labelmap.fine_weights, labelmap.fine_names)
        coarse_predictions = [labelmap.fine_to_coarse[item] for item in predictions]
        coarse_targets = [labelmap.fine_to_coarse[item] for item in targets]
        coarse = segmentation_metrics(
            coarse_predictions, coarse_targets, cases,
            labelmap.coarse_weights, labelmap.coarse_names,
        )
        folds.append({
            "fold": fold,
            "prediction_dir": str(prediction_dir),
            "fine_dice": fine["dice"], "fine_nhd": fine["nhd"],
            "coarse_dice": coarse["dice"], "coarse_nhd": coarse["nhd"],
            "fine_pixel_accuracy": float(np.mean([
                (prediction == target).mean() for prediction, target in zip(predictions, targets)
            ])),
            "selection_score": None,
        })
    if len(folds) < 5 and not args.allow_incomplete:
        raise RuntimeError(f"Only {len(folds)}/5 nnU-Net test folds are available")
    names = ("fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd", "fine_pixel_accuracy")
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in folds])),
            "sample_standard_deviation": float(np.std([row[name] for row in folds], ddof=1))
            if len(folds) > 1 else 0.0,
        }
        for name in names
    }
    output = {
        "experiment": "nnunet_v2_2d",
        "model_family": "nnunet",
        "completed_test_folds": len(folds),
        "selection_score": None,
        "folds": folds,
        "summary": summary,
        "note": "Segmentation-only weighted TIGER test evaluation; visibility and selection_score are unavailable.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
