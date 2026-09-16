#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import (
    STATIONS, LabelMap, records_for_fold, write_visibility_ground_truth,
)
from tiger_models.runtime import atomic_json
from tiger_models.metrics import segmentation_metrics, selection_score, visibility_metrics


def validate(prediction_root: Path, data_root: Path, fold: int, split: str = "test") -> dict:
    labelmap = LabelMap.load(data_root / "labelmap.csv")
    records = records_for_fold(data_root, fold, split)
    expected = {record.name for record in records}
    for task in ("task1", "task2"):
        directory = prediction_root / task
        actual = {path.name for path in directory.glob("*.png")}
        if actual != expected:
            raise ValueError(f"{task} filename mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    fine_predictions = []
    fine_targets = []
    for record in records:
        with Image.open(record.image) as source:
            expected_size = source.size
        with Image.open(prediction_root / "task1" / record.name) as image:
            if image.mode != "RGB" or image.format != "PNG" or image.size != expected_size:
                raise ValueError(f"Invalid Task 1 PNG metadata for {record.name}")
            fine = labelmap.decode_fine(np.asarray(image))
        with Image.open(prediction_root / "task2" / record.name) as image:
            if image.mode != "RGB" or image.format != "PNG" or image.size != expected_size:
                raise ValueError(f"Invalid Task 2 PNG metadata for {record.name}")
            coarse = labelmap.decode_coarse(np.asarray(image))
        if not np.array_equal(labelmap.fine_to_coarse[fine], coarse):
            raise ValueError(f"Task 2 is not derived from Task 1 for {record.name}")
        target = labelmap.decode_fine(np.asarray(Image.open(record.fine_mask).convert("RGB")))
        evaluation_size = (896, 512)
        fine_predictions.append(np.asarray(
            Image.fromarray(fine).resize(evaluation_size, Image.Resampling.NEAREST)
        ))
        fine_targets.append(np.asarray(
            Image.fromarray(target).resize(evaluation_size, Image.Resampling.NEAREST)
        ))

    with (prediction_root / "task3.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["case_id", *STATIONS]:
            raise ValueError(f"Task 3 columns/order are incorrect: {reader.fieldnames}")
        rows = list(reader)
    ids = {row["case_id"] for row in rows}
    expected_ids = {Path(name).stem for name in expected}
    if ids != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("Task 3 case_id rows do not match validation frames")
    values = np.asarray([[float(row[station]) for station in STATIONS] for row in rows])
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Task 3 values must be finite probabilities in [0,1]")
    probability_by_name = {f"{row['case_id']}.png": values[index] for index, row in enumerate(rows)}
    ordered_probabilities = np.stack([probability_by_name[record.name] for record in records])
    visibility_target = np.stack([record.visibility for record in records])
    case_ids = [record.case_id for record in records]
    fine_metrics = segmentation_metrics(
        fine_predictions, fine_targets, case_ids, labelmap.fine_weights, labelmap.fine_names
    )
    coarse_predictions = [labelmap.fine_to_coarse[item] for item in fine_predictions]
    coarse_targets = [labelmap.fine_to_coarse[item] for item in fine_targets]
    coarse_metrics = segmentation_metrics(
        coarse_predictions, coarse_targets, case_ids, labelmap.coarse_weights, labelmap.coarse_names
    )
    visibility = visibility_metrics(visibility_target, ordered_probabilities, STATIONS)
    quantitative = {
        "evaluation_resolution": [512, 896],
        "fine_dice": fine_metrics["dice"], "fine_nhd": fine_metrics["nhd"],
        "coarse_dice": coarse_metrics["dice"], "coarse_nhd": coarse_metrics["nhd"],
        "fine_pixel_accuracy": float(np.mean([
            (prediction == target).mean()
            for prediction, target in zip(fine_predictions, fine_targets)
        ])),
        "coarse_pixel_accuracy": float(np.mean([
            (prediction == target).mean()
            for prediction, target in zip(coarse_predictions, coarse_targets)
        ])),
        "visibility_accuracy": float(
            ((ordered_probabilities >= 0.5) == (visibility_target >= 0.5)).mean()
        ),
        "visibility_macro_f1": visibility["macro_f1"],
        "visibility_macro_auroc": visibility["macro_auroc"],
    }
    quantitative["selection_score"] = selection_score(
        quantitative["fine_dice"], quantitative["fine_nhd"],
        quantitative["coarse_dice"], quantitative["coarse_nhd"],
        quantitative["visibility_macro_f1"], quantitative["visibility_macro_auroc"],
    )
    return {
        "status": "PASS", "fold": fold, "split": split, "frames": len(expected),
        "task1_rgb_png": True, "task2_rgb_png": True,
        "task2_derived_from_task1": True, "task3_raw_probabilities": True,
        "oracle_prompt": False, "quantitative_metrics": quantitative,
    }


def prepare_official_input(
    prediction_root: Path,
    data_root: Path,
    fold: int,
    destination: Path,
    fixture: bool,
    split: str = "test",
) -> tuple[Path, Path]:
    records = records_for_fold(data_root, fold, split)
    gt_root = destination / "gt"
    method_name = next(
        (parent.name for parent in prediction_root.parents if parent.name.endswith("_partial")),
        "sam2_hiera_tiny_partial",
    )
    pred_root = destination / "predictions" / method_name
    if destination.exists():
        shutil.rmtree(destination)
    for base in (gt_root, pred_root):
        (base / "task1").mkdir(parents=True)
        (base / "task2").mkdir(parents=True)
    for record in records:
        for task, source_directory in (
            ("task1", data_root / "masks_fine"),
            ("task2", data_root / "masks_coarse"),
        ):
            pairs = [
                (source_directory / record.name, gt_root / task / record.name),
                (prediction_root / task / record.name, pred_root / task / record.name),
            ]
            for source, target in pairs:
                if fixture:
                    Image.open(source).convert("RGB").resize((28, 16), Image.Resampling.NEAREST).save(target)
                else:
                    target.symlink_to(source.resolve())
    write_visibility_ground_truth(records, gt_root / "task3.csv")
    shutil.copy2(prediction_root / "task3.csv", pred_root / "task3.csv")
    return gt_root, destination / "predictions"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-official", action="store_true")
    parser.add_argument("--official-fixture", action="store_true", help="format/API smoke only; not benchmark metrics")
    parser.add_argument("--official-evaluator", type=Path, default=ROOT / "external/tigersqai_evaluation")
    parser.add_argument("--official-timeout-seconds", type=int, default=None)
    args = parser.parse_args()
    result = validate(args.prediction_dir, args.data_root, args.fold, args.split)
    atomic_json(args.prediction_dir.parent / "quantitative_metrics.json", result["quantitative_metrics"])
    output = args.output or args.prediction_dir.parent / "prediction_validation.json"
    if args.run_official:
        official_root = args.prediction_dir.parent / (
            "official_fixture_input" if args.official_fixture else "official_native_input"
        )
        gt_root, pred_root = prepare_official_input(
            args.prediction_dir, args.data_root, args.fold, official_root,
            args.official_fixture, args.split,
        )
        report = args.prediction_dir.parent / (
            "official_fixture_report.md" if args.official_fixture else "official_report.md"
        )
        command = [
            sys.executable,
            str(args.official_evaluator / "metrics/01_evaluate_challenge.py"),
            "--gt", str(gt_root.resolve()), "--pred", str(pred_root.resolve()),
            "--out", str(report.resolve()),
            "--figures-dir", "none", "--save-json",
        ]
        timed_out = False
        try:
            process = subprocess.run(
                command, cwd=args.official_evaluator, text=True, capture_output=True,
                timeout=args.official_timeout_seconds,
            )
            output_text = process.stdout + process.stderr
            returncode = process.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            output_text = stdout + stderr + f"\nTIMEOUT after {args.official_timeout_seconds} seconds\n"
            returncode = 124
        (report.parent / f"{report.stem}.log").write_text(output_text)
        result.update({
            "official_evaluator_returncode": returncode,
            "official_evaluator_passed": returncode == 0 and report.is_file(),
            "official_evaluator_mode": "downscaled_format_fixture" if args.official_fixture else "native_benchmark",
            "official_evaluator_timed_out": timed_out,
            "official_report": str(report),
        })
        if returncode != 0:
            atomic_json(output, result)
            raise RuntimeError(
                f"Official evaluator failed with code {returncode}; see {report.with_suffix('.log')}"
            )
    atomic_json(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
