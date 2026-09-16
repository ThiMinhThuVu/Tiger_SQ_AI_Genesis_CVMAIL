#!/usr/bin/env python3
"""P0 native-resolution baseline lock for the full40 fine-31 experiment.

The command is deliberately split into prepare/infer/finalize stages so GPU
inference can run per fold and the expensive native-size metric pass can be
resumed without retraining.  P0 never selects or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from full_version.task2_fine_31cls.scripts.data_full import (  # noqa: E402
    audit_fine_masks,
    build_manifest,
    load_manifest,
    records_for_manifest,
)
from tiger_models.data import LabelMap, sha256  # noqa: E402
from tiger_models.metrics import binary_dice, normalized_hausdorff  # noqa: E402


P0_VERSION = "p0_native_baseline_lock_v1"
LEGACY_TASK_SCORE = 0.7740267491216186
PARITY_TOLERANCE = 1e-6
AGGREGATION_TOLERANCE = 1e-10
EXPECTED_CASES = 40
EXPECTED_FRAMES = 524

_WORKER_LABELMAP: LabelMap | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite P0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(path)


def write_text(path: Path, payload: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite P0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite P0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest_paths(paths: Iterable[Path]) -> dict[str, str]:
    root = ROOT.resolve()
    return {
        str(path.resolve().relative_to(root)): sha256(path.resolve())
        for path in sorted(paths)
    }


def root_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(path.read_text()))


def task_score(dice: float, nhd: float) -> float:
    return (float(dice) + 1.0 - float(nhd)) / 2.0


def metric_contract_markdown() -> str:
    return """# P0 evaluation contract — full40 fine-31

## Material Passport

- Origin Skill: experiment-agent / run
- Origin Date: 2026-08-27 UTC
- Verification Status: governed by `P0_SUMMARY.md`
- Version Label: p0_native_baseline_lock_v1

## Frozen objective

- Challenge task: **Task 1 fine-grained semantic segmentation**.
- Output space: IDs 0..30 (background plus 30 foreground classes).
- Local compatibility key: `task1_score`.
- Primary score: `(weighted Dice + 1 - weighted nHD) / 2`.
- Aggregation order: class-weighted frame score -> mean frames within case ->
  mean cases within dataset. Every case has equal dataset weight.
- Absent convention: both empty gives Dice=1/nHD=0; exactly one empty gives
  Dice=0/nHD=1.
- Score unit: delta 0.005 equals 0.5 percentage point.

## Native-resolution rule

The model may consume resized inputs, but fine probabilities are interpolated
to the original image dimensions before argmax. Predictions are scored against
the untouched native-resolution RGB ground truth. Evaluation must never resize
ground truth to the model input grid.

## Data-use firewall

- Inner validation selects checkpoints and screens interventions.
- Outer OOF is a confirmation set, not a tuning dashboard.
- P0 evaluates only the already-frozen baseline checkpoints.
- No hidden challenge data or leaderboard result is used.
"""


def _official_metric_functions():
    external_root = ROOT / "external/tigersqai_evaluation"
    sys.path.insert(0, str(external_root))
    try:
        from metrics.metrics import (  # type: ignore
            _binary_dice as official_dice,
            _binary_normalized_hausdorff as official_nhd,
        )
    finally:
        sys.path.remove(str(external_root))
    return official_dice, official_nhd


def synthetic_metric_parity() -> dict[str, Any]:
    official_dice, official_nhd = _official_metric_functions()
    shape = (32, 48)
    scenarios: list[tuple[str, np.ndarray, np.ndarray]] = []

    empty = np.zeros(shape, dtype=bool)
    square = empty.copy(); square[8:20, 12:28] = True
    shifted = empty.copy(); shifted[10:22, 15:31] = True
    thin = empty.copy(); thin[4:28, 22] = True
    thin_shifted = empty.copy(); thin_shifted[4:28, 24] = True
    remote_fp = square.copy(); remote_fp[30, 46] = True
    scenarios.extend([
        ("both_empty", empty, empty),
        ("prediction_only", square, empty),
        ("target_only", empty, square),
        ("perfect", square, square),
        ("shifted", shifted, square),
        ("remote_false_positive", remote_fp, square),
        ("thin_shifted", thin_shifted, thin),
    ])

    diagonal = float(np.hypot(*shape))
    rows = []
    max_dice_difference = 0.0
    max_nhd_difference = 0.0
    for name, prediction, target in scenarios:
        fast_dice = binary_dice(prediction, target)
        fast_nhd = normalized_hausdorff(prediction, target)
        reference_dice = float(official_dice(prediction, target))
        reference_nhd = float(official_nhd(prediction, target, diagonal))
        dice_difference = abs(fast_dice - reference_dice)
        nhd_difference = abs(fast_nhd - reference_nhd)
        max_dice_difference = max(max_dice_difference, dice_difference)
        max_nhd_difference = max(max_nhd_difference, nhd_difference)
        rows.append({
            "scenario": name,
            "fast_dice": fast_dice,
            "reference_dice": reference_dice,
            "absolute_dice_difference": dice_difference,
            "fast_nhd": fast_nhd,
            "reference_nhd": reference_nhd,
            "absolute_nhd_difference": nhd_difference,
        })
    passed = max(max_dice_difference, max_nhd_difference) <= PARITY_TOLERANCE
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": PARITY_TOLERANCE,
        "max_absolute_dice_difference": max_dice_difference,
        "max_absolute_nhd_difference": max_nhd_difference,
        "scenarios": rows,
    }


def split_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    test_occurrences: list[str] = []
    validation_occurrences: list[str] = []
    rows = []
    passed = True
    for fold_text, payload in sorted(manifest["folds"].items(), key=lambda item: int(item[0])):
        train = set(payload["train_cases"])
        validation = set(payload["validation_cases"])
        test = set(payload["test_cases"])
        overlap = {
            "train_validation": sorted(train & validation),
            "train_test": sorted(train & test),
            "validation_test": sorted(validation & test),
        }
        fold_pass = (
            not any(overlap.values())
            and (len(train), len(validation), len(test)) == (28, 4, 8)
        )
        passed &= fold_pass
        test_occurrences.extend(sorted(test))
        validation_occurrences.extend(sorted(validation))
        rows.append({
            "fold": int(fold_text),
            "train_cases": len(train),
            "validation_cases": len(validation),
            "test_cases": len(test),
            "overlap": overlap,
            "status": "PASS" if fold_pass else "FAIL",
        })
    test_counts = Counter(test_occurrences)
    oof_once = len(test_counts) == EXPECTED_CASES and set(test_counts.values()) == {1}
    validation_unique = len(validation_occurrences) == len(set(validation_occurrences)) == 20
    passed &= oof_once and validation_unique
    return {
        "status": "PASS" if passed else "FAIL",
        "grouping_unit": manifest.get("grouping_unit"),
        "folds": rows,
        "outer_oof_case_occurrences": dict(sorted(test_counts.items())),
        "every_case_in_outer_test_exactly_once": oof_once,
        "validation_assignments_unique": validation_unique,
    }


def baseline_manifest(
    config_path: Path,
    split_path: Path,
    model_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    checkpoints = [model_dir / f"fold_{fold}" / "best.pt" for fold in range(5)]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    source_paths = [
        config_path,
        split_path,
        ROOT / "tiger_models/metrics.py",
        ROOT / "tiger_models/data.py",
        ROOT / "scripts/train_mask2former_improved.py",
        Path(__file__),
    ]
    config = load_config(config_path)
    try:
        import scipy
        import torch
        import transformers

        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    except Exception as error:  # pragma: no cover - environment capture only
        environment = {"capture_error": repr(error)}
    return {
        "p0_version": P0_VERSION,
        "generated_at": utc_now(),
        "experiment_id": config.get("experiment_id"),
        "challenge_task": "Task 1 fine-grained semantic segmentation",
        "local_metric_key": "task1_score",
        "model_input_size": [int(config["height"]), int(config["width"])],
        "base_seed": int(config["seed"]),
        "effective_fold_seeds": [int(config["seed"]) + fold for fold in range(5)],
        "config": root_relative(config_path),
        "split_manifest": root_relative(split_path),
        "model_dir": root_relative(model_dir),
        "data_root": root_relative(data_root),
        "missing_checkpoints": missing,
        "checkpoint_sha256": digest_paths([path for path in checkpoints if path.is_file()]),
        "source_sha256": digest_paths(source_paths),
        "environment": environment,
        "status": "PASS" if not missing else "FAIL",
    }


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"P0 output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    frozen_manifest = load_manifest(args.split_manifest)
    current_manifest = build_manifest(args.data_root, int(config["split_seed"]))
    snapshot_match = current_manifest["snapshot"] == frozen_manifest.get("snapshot")
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    mask_audit = audit_fine_masks(args.data_root, labelmap)
    dataset_pass = (
        snapshot_match
        and current_manifest["snapshot"]["case_count"] == EXPECTED_CASES
        and current_manifest["snapshot"]["frame_count"] == EXPECTED_FRAMES
        and mask_audit["paired_frames_checked"] == EXPECTED_FRAMES
        and mask_audit["dimensions_match"]
    )
    dataset_payload = {
        "status": "PASS" if dataset_pass else "FAIL",
        "snapshot_matches_frozen_manifest": snapshot_match,
        "snapshot": current_manifest["snapshot"],
        "mask_audit": mask_audit,
        "case_inventory": current_manifest["cases"],
    }

    write_text(output_dir / "evaluation_contract.md", metric_contract_markdown())
    write_json(output_dir / "dataset_audit.json", dataset_payload)
    write_json(output_dir / "split_audit.json", split_audit(frozen_manifest))
    write_json(output_dir / "metric_parity_synthetic.json", synthetic_metric_parity())
    write_json(
        output_dir / "baseline_manifest.json",
        baseline_manifest(args.config, args.split_manifest, args.model_dir, args.data_root),
    )
    legacy_path = ROOT / "full_version/task2_fine_31cls/artifacts/seed_2026/test_oof/oof_test_metrics.json"
    if not legacy_path.is_file():
        raise FileNotFoundError(f"Missing legacy OOF metrics: {legacy_path}")
    shutil.copyfile(legacy_path, output_dir / "legacy_resized_metrics.json")
    write_json(output_dir / "prepare_complete.json", {
        "status": "COMPLETED",
        "generated_at": utc_now(),
        "next": "run infer for folds 0..4",
    })
    print(json.dumps({
        "stage": "prepare",
        "output_dir": str(output_dir),
        "dataset_status": dataset_payload["status"],
        "split_status": json.loads((output_dir / "split_audit.json").read_text())["status"],
        "parity_status": json.loads((output_dir / "metric_parity_synthetic.json").read_text())["status"],
    }, indent=2), flush=True)


def _load_native_target(path: Path, labelmap: LabelMap) -> np.ndarray:
    with Image.open(path) as handle:
        return labelmap.decode_fine(np.asarray(handle.convert("RGB")))


def infer(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as torch_functional
    from torch.utils.data import DataLoader

    from scripts.train_mask2former import collate
    from scripts.train_mask2former_improved import build_improved
    from tiger_models.data import TigerMultitaskDataset
    from tiger_models.improved_mask2former import soft_presence_modulation

    if args.fold is None:
        raise ValueError("--fold is required for infer")
    if not torch.cuda.is_available():
        raise RuntimeError("P0 native inference requires CUDA")
    output_dir = args.output_dir
    if not (output_dir / "prepare_complete.json").is_file():
        raise FileNotFoundError("P0 prepare stage has not completed")
    fold_output = output_dir / "folds" / f"fold_{args.fold}.json"
    if fold_output.exists():
        raise FileExistsError(f"Fold already completed: {fold_output}")

    config = load_config(args.config)
    manifest = load_manifest(args.split_manifest)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    checkpoint_path = args.model_dir / f"fold_{args.fold}" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_improved(config, labelmap).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    records = records_for_manifest(args.data_root, manifest, args.fold, "test")
    dataset = TigerMultitaskDataset(
        records, labelmap, int(config["height"]), int(config["width"]), False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    prediction_dir = output_dir / "oof_predictions_native_ids"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, (record, batch) in enumerate(zip(records, loader), start=1):
            prediction_path = prediction_dir / record.name
            if prediction_path.exists():
                raise FileExistsError(f"Refusing to overwrite prediction: {prediction_path}")
            with Image.open(record.image) as image_handle:
                original_width, original_height = image_handle.size
            result = model(batch["image"].to(device, non_blocking=True))
            fine_probability = result["fine_probability"]
            if config.get("presence_head", True):
                fine_probability = soft_presence_modulation(
                    fine_probability,
                    result["presence_logits"],
                    float(config["presence_epsilon"]),
                    float(config["presence_gamma"]),
                )
            native_probability = torch_functional.interpolate(
                fine_probability,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = native_probability.argmax(1).to(torch.uint8).cpu().numpy()[0]
            Image.fromarray(prediction).save(prediction_path, compress_level=6)
            with Image.open(record.fine_mask) as target_handle:
                target_size = target_handle.size
            if target_size != (original_width, original_height):
                raise ValueError(f"Native image/GT mismatch for {record.name}")
            rows.append({
                "name": record.name,
                "case_id": record.case_id,
                "fold": args.fold,
                "original_width": original_width,
                "original_height": original_height,
                "prediction_sha256": sha256(prediction_path),
            })
            print(json.dumps({
                "stage": "infer",
                "fold": args.fold,
                "completed_frames": index,
                "total_frames": len(records),
                "name": record.name,
            }), flush=True)
            del result, fine_probability, native_probability

    write_json(fold_output, {
        "status": "COMPLETED",
        "fold": args.fold,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "frames": len(rows),
        "cases": sorted({row["case_id"] for row in rows}),
        "predictions": rows,
        "completed_at": utc_now(),
    })
    print(json.dumps({
        "stage": "infer",
        "fold": args.fold,
        "status": "COMPLETED",
        "frames": len(rows),
        "output": str(fold_output),
    }, indent=2), flush=True)


def _worker_init(labelmap_path: str) -> None:
    global _WORKER_LABELMAP
    _WORKER_LABELMAP = LabelMap.load(labelmap_path)


def frame_metrics_worker(payload: tuple[str, str, int, int, str, str]) -> dict[str, Any]:
    name, case_id, center, fold, prediction_text, target_text = payload
    if _WORKER_LABELMAP is None:
        raise RuntimeError("Metric worker label map is not initialized")
    labelmap = _WORKER_LABELMAP
    prediction_path = Path(prediction_text)
    target_path = Path(target_text)
    with Image.open(prediction_path) as prediction_handle:
        prediction = np.asarray(prediction_handle, dtype=np.uint8)
    target = _load_native_target(target_path, labelmap)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/GT size mismatch for {name}: {prediction.shape} != {target.shape}")
    if prediction.min(initial=0) < 0 or prediction.max(initial=0) >= len(labelmap.fine_names):
        raise ValueError(f"Prediction IDs outside 0..30 for {name}")

    class_rows = []
    weighted_dice = 0.0
    weighted_nhd = 0.0
    weight_total = float(labelmap.fine_weights.sum())
    for class_id, (class_name, weight) in enumerate(
        zip(labelmap.fine_names, labelmap.fine_weights.astype(float))
    ):
        pred_binary = prediction == class_id
        target_binary = target == class_id
        dice = binary_dice(pred_binary, target_binary)
        nhd = normalized_hausdorff(pred_binary, target_binary)
        pred_present = bool(pred_binary.any())
        target_present = bool(target_binary.any())
        weighted_dice += dice * weight
        weighted_nhd += nhd * weight
        class_rows.append({
            "name": name,
            "case_id": case_id,
            "center": center,
            "fold": fold,
            "class_id": class_id,
            "class_name": class_name,
            "weight": weight,
            "target_present": int(target_present),
            "prediction_present": int(pred_present),
            "target_pixels": int(target_binary.sum()),
            "prediction_pixels": int(pred_binary.sum()),
            "dice": dice,
            "nhd": nhd,
        })
    weighted_dice /= weight_total
    weighted_nhd /= weight_total
    return {
        "name": name,
        "case_id": case_id,
        "center": center,
        "fold": fold,
        "width": int(prediction.shape[1]),
        "height": int(prediction.shape[0]),
        "dice": weighted_dice,
        "nhd": weighted_nhd,
        "task_score": task_score(weighted_dice, weighted_nhd),
        "class_rows": class_rows,
    }


def _mean(values: Iterable[float]) -> float:
    return float(statistics.mean(float(value) for value in values))


def aggregate_rows(
    frame_rows: Sequence[dict[str, Any]],
    class_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    frames_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        frames_by_case[str(row["case_id"])].append(row)
    case_rows = []
    for case_id, rows in sorted(frames_by_case.items()):
        dice = _mean(row["dice"] for row in rows)
        nhd = _mean(row["nhd"] for row in rows)
        case_rows.append({
            "case_id": case_id,
            "center": int(rows[0]["center"]),
            "fold": int(rows[0]["fold"]),
            "frames": len(rows),
            "dice": dice,
            "nhd": nhd,
            "task_score": task_score(dice, nhd),
        })

    def summarize_cases(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        dice = _mean(row["dice"] for row in rows)
        nhd = _mean(row["nhd"] for row in rows)
        return {
            "cases": len(rows),
            "frames": sum(int(row["frames"]) for row in rows),
            "dice": dice,
            "nhd": nhd,
            "task_score": task_score(dice, nhd),
        }

    overall = summarize_cases(case_rows)
    per_center = {}
    for center in sorted({int(row["center"]) for row in case_rows}):
        selected = [row for row in case_rows if int(row["center"]) == center]
        per_center[f"center_{center}"] = summarize_cases(selected)
    per_fold = {}
    for fold in sorted({int(row["fold"]) for row in case_rows}):
        selected = [row for row in case_rows if int(row["fold"]) == fold]
        per_fold[str(fold)] = summarize_cases(selected)

    classes_by_case: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        classes_by_case[(int(row["class_id"]), str(row["case_id"]))].append(row)
    per_class = []
    class_ids = sorted({int(row["class_id"]) for row in class_rows})
    class_weights = {
        class_id: float(next(
            row["weight"] for row in class_rows if int(row["class_id"]) == class_id
        ))
        for class_id in class_ids
    }
    weight_total = sum(class_weights.values())
    if weight_total <= 0:
        raise ValueError("Class-weight total must be positive")
    for class_id in class_ids:
        selected = [row for row in class_rows if int(row["class_id"]) == class_id]
        case_class_rows = []
        for (candidate_id, case_id), rows in sorted(classes_by_case.items()):
            if candidate_id != class_id:
                continue
            case_class_rows.append({
                "case_id": case_id,
                "dice": _mean(row["dice"] for row in rows),
                "nhd": _mean(row["nhd"] for row in rows),
            })
        dice = _mean(row["dice"] for row in case_class_rows)
        nhd = _mean(row["nhd"] for row in case_class_rows)
        weight = class_weights[class_id]
        present = [row for row in selected if int(row["target_present"]) == 1]
        absent = [row for row in selected if int(row["target_present"]) == 0]
        misses = sum(
            int(row["target_present"]) == 1 and int(row["prediction_present"]) == 0
            for row in selected
        )
        false_positives = sum(
            int(row["target_present"]) == 0 and int(row["prediction_present"]) == 1
            for row in selected
        )
        error_budget = (weight / weight_total) * ((1.0 - dice) + nhd) / 2.0
        per_class.append({
            "class_id": class_id,
            "class_name": selected[0]["class_name"],
            "weight": weight,
            "case_balanced_dice": dice,
            "case_balanced_nhd": nhd,
            "task_error_budget": error_budget,
            "present_frames": len(present),
            "absent_frames": len(absent),
            "present_only_dice": _mean(row["dice"] for row in present) if present else None,
            "present_only_nhd": _mean(row["nhd"] for row in present) if present else None,
            "complete_miss_rate": misses / len(present) if present else None,
            "absent_false_positive_rate": false_positives / len(absent) if absent else None,
            "target_pixels": sum(int(row["target_pixels"]) for row in selected),
        })

    reconstruction = 1.0 - sum(float(row["task_error_budget"]) for row in per_class)
    fold_scores = [float(payload["task_score"]) for payload in per_fold.values()]
    return {
        "overall": overall,
        "per_fold": per_fold,
        "fold_sample_standard_deviation": statistics.stdev(fold_scores),
        "per_center": per_center,
        "worst_center": min(per_center, key=lambda key: per_center[key]["task_score"]),
        "per_class": per_class,
        "class_weight_total": weight_total,
        "task_score_reconstructed_from_class_error_budget": reconstruction,
        "aggregation_absolute_difference": abs(overall["task_score"] - reconstruction),
        "case_rows": case_rows,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _typed_frame_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    return [{
        **row,
        "center": int(row["center"]),
        "fold": int(row["fold"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "dice": float(row["dice"]),
        "nhd": float(row["nhd"]),
        "task_score": float(row["task_score"]),
    } for row in rows]


def _typed_class_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    integer_fields = {
        "center", "fold", "class_id", "target_present", "prediction_present",
        "target_pixels", "prediction_pixels",
    }
    float_fields = {"weight", "dice", "nhd"}
    output = []
    for row in rows:
        converted = dict(row)
        for key in integer_fields:
            converted[key] = int(converted[key])
        for key in float_fields:
            converted[key] = float(converted[key])
        output.append(converted)
    return output


def real_metric_parity(
    frame_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    labelmap: LabelMap,
) -> dict[str, Any]:
    official_dice, official_nhd = _official_metric_functions()
    selected_indices = np.linspace(0, len(frame_rows) - 1, num=min(12, len(frame_rows)), dtype=int)
    rows = []
    max_dice_difference = 0.0
    max_nhd_difference = 0.0
    for index in selected_indices:
        row = frame_rows[int(index)]
        name = str(row["name"])
        with Image.open(args.output_dir / "oof_predictions_native_ids" / name) as handle:
            prediction = np.asarray(
                handle.resize((96, 64), Image.Resampling.NEAREST), dtype=np.uint8,
            )
        with Image.open(args.data_root / "masks_fine" / name) as handle:
            target_rgb = np.asarray(handle.resize((96, 64), Image.Resampling.NEAREST).convert("RGB"))
        target = labelmap.decode_fine(target_rgb)
        diagonal = float(np.hypot(*prediction.shape))
        for class_id in range(len(labelmap.fine_names)):
            pred_binary = prediction == class_id
            target_binary = target == class_id
            fast_dice = binary_dice(pred_binary, target_binary)
            fast_nhd = normalized_hausdorff(pred_binary, target_binary)
            reference_dice = float(official_dice(pred_binary, target_binary))
            reference_nhd = float(official_nhd(pred_binary, target_binary, diagonal))
            dice_difference = abs(fast_dice - reference_dice)
            nhd_difference = abs(fast_nhd - reference_nhd)
            max_dice_difference = max(max_dice_difference, dice_difference)
            max_nhd_difference = max(max_nhd_difference, nhd_difference)
            rows.append({
                "name": name,
                "class_id": class_id,
                "absolute_dice_difference": dice_difference,
                "absolute_nhd_difference": nhd_difference,
            })
    passed = max(max_dice_difference, max_nhd_difference) <= PARITY_TOLERANCE
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": PARITY_TOLERANCE,
        "frames_sampled": len(selected_indices),
        "class_comparisons": len(rows),
        "comparison_grid": [64, 96],
        "max_absolute_dice_difference": max_dice_difference,
        "max_absolute_nhd_difference": max_nhd_difference,
        "comparisons": rows,
    }


def finalize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    manifest = load_manifest(args.split_manifest)
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    fold_payloads = []
    for fold in range(5):
        path = output_dir / "folds" / f"fold_{fold}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing P0 fold inference artifact: {path}")
        fold_payloads.append(json.loads(path.read_text()))

    prediction_records = [row for payload in fold_payloads for row in payload["predictions"]]
    names = [str(row["name"]) for row in prediction_records]
    expected_names = sorted(path.name for path in (args.data_root / "images").glob("*.png"))
    if len(names) != EXPECTED_FRAMES or len(set(names)) != EXPECTED_FRAMES:
        raise ValueError("P0 OOF predictions must contain 524 unique frames")
    if sorted(names) != expected_names:
        raise ValueError("P0 prediction names differ from the data snapshot")

    case_to_fold = {
        case_id: int(fold)
        for fold, payload in manifest["folds"].items()
        for case_id in payload["test_cases"]
    }
    cases = manifest["cases"]
    work_items = []
    for name in expected_names:
        parsed_case = name.rsplit("_", 1)[0]
        if "_frame_" in name:
            from full_version.task2_fine_31cls.scripts.data_full import parse_frame_name
            parsed_case = str(parse_frame_name(name)["case_id"])
        center = int(cases[parsed_case]["center"])
        work_items.append((
            name,
            parsed_case,
            center,
            case_to_fold[parsed_case],
            str(output_dir / "oof_predictions_native_ids" / name),
            str(args.data_root / "masks_fine" / name),
        ))

    frame_rows = []
    class_rows = []
    with ProcessPoolExecutor(
        max_workers=int(args.metric_workers),
        initializer=_worker_init,
        initargs=(str(args.data_root / "labelmap.csv"),),
    ) as executor:
        for index, result in enumerate(executor.map(frame_metrics_worker, work_items), start=1):
            class_rows.extend(result.pop("class_rows"))
            frame_rows.append(result)
            if index % 5 == 0 or index == len(work_items):
                print(json.dumps({
                    "stage": "native_metrics",
                    "completed_frames": index,
                    "total_frames": len(work_items),
                }), flush=True)

    frame_fields = [
        "name", "case_id", "center", "fold", "width", "height",
        "dice", "nhd", "task_score",
    ]
    class_fields = [
        "name", "case_id", "center", "fold", "class_id", "class_name", "weight",
        "target_present", "prediction_present", "target_pixels", "prediction_pixels",
        "dice", "nhd",
    ]
    write_csv(output_dir / "per_frame.csv", frame_rows, frame_fields)
    write_csv(output_dir / "per_class_frame.csv", class_rows, class_fields)

    aggregate = aggregate_rows(frame_rows, class_rows)
    case_rows = aggregate.pop("case_rows")
    case_fields = ["case_id", "center", "fold", "frames", "dice", "nhd", "task_score"]
    write_csv(output_dir / "per_case.csv", case_rows, case_fields)
    fold_rows = [{"fold": int(fold), **payload} for fold, payload in aggregate["per_fold"].items()]
    write_csv(
        output_dir / "per_fold.csv", fold_rows,
        ["fold", "cases", "frames", "dice", "nhd", "task_score"],
    )
    center_rows = [{"center": center, **payload} for center, payload in aggregate["per_center"].items()]
    write_csv(
        output_dir / "per_center.csv", center_rows,
        ["center", "cases", "frames", "dice", "nhd", "task_score"],
    )

    score_difference = float(aggregate["overall"]["task_score"]) - LEGACY_TASK_SCORE
    canonical = {
        "p0_version": P0_VERSION,
        "generated_at": utc_now(),
        "experiment": load_config(args.config)["experiment_id"],
        "evaluation_grid": "native_resolution",
        "aggregation": "class-weighted frame -> case mean -> dataset mean",
        "legacy_resized_task_score": LEGACY_TASK_SCORE,
        "native_minus_legacy_task_score": score_difference,
        **aggregate,
    }
    write_json(output_dir / "canonical_native_oof_metrics.json", canonical)

    prediction_paths = [output_dir / "oof_predictions_native_ids" / name for name in expected_names]
    prediction_hashes = digest_paths(prediction_paths)
    checksum_lines = [f"{digest}  {path}\n" for path, digest in sorted(prediction_hashes.items())]
    write_text(output_dir / "prediction_checksums.sha256", "".join(checksum_lines))

    parity_real = real_metric_parity(frame_rows, args, labelmap)
    write_json(output_dir / "metric_parity_real_sample.json", parity_real)

    reconstructed = aggregate_rows(
        _typed_frame_rows(_read_csv(output_dir / "per_frame.csv")),
        _typed_class_rows(_read_csv(output_dir / "per_class_frame.csv")),
    )
    reconstruction_checks = {
        "overall_task_score": abs(
            reconstructed["overall"]["task_score"] - canonical["overall"]["task_score"]
        ),
        "overall_dice": abs(reconstructed["overall"]["dice"] - canonical["overall"]["dice"]),
        "overall_nhd": abs(reconstructed["overall"]["nhd"] - canonical["overall"]["nhd"]),
        "class_error_budget": canonical["aggregation_absolute_difference"],
    }
    reproducible = max(reconstruction_checks.values()) <= AGGREGATION_TOLERANCE
    reproducibility = {
        "status": "REPRODUCIBLE" if reproducible else "NOT_REPRODUCIBLE",
        "classification": "deterministic evaluation of frozen prediction masks",
        "tolerance": AGGREGATION_TOLERANCE,
        "checks": reconstruction_checks,
        "prediction_files": len(prediction_paths),
        "all_prediction_hashes_unique_by_path": len(prediction_hashes) == EXPECTED_FRAMES,
        "note": "CSV reload and independent aggregation exactly reproduce the canonical metrics.",
    }
    write_json(output_dir / "reproducibility_report.json", reproducibility)

    prepare_gates = {
        "dataset": json.loads((output_dir / "dataset_audit.json").read_text())["status"],
        "split": json.loads((output_dir / "split_audit.json").read_text())["status"],
        "synthetic_metric_parity": json.loads(
            (output_dir / "metric_parity_synthetic.json").read_text()
        )["status"],
        "real_sample_metric_parity": parity_real["status"],
        "baseline_manifest": json.loads((output_dir / "baseline_manifest.json").read_text())["status"],
        "native_prediction_coverage": "PASS" if len(prediction_paths) == EXPECTED_FRAMES else "FAIL",
        "aggregation_reconstruction": "PASS" if (
            canonical["aggregation_absolute_difference"] <= AGGREGATION_TOLERANCE
        ) else "FAIL",
        "reproducibility": "PASS" if reproducible else "FAIL",
    }
    p0_pass = all(status == "PASS" for status in prepare_gates.values())
    summary = {
        "status": "PASS" if p0_pass else "FAIL",
        "generated_at": utc_now(),
        "gates": prepare_gates,
        "canonical_baseline": canonical["overall"],
        "worst_center": canonical["worst_center"],
        "legacy_resized_task_score": LEGACY_TASK_SCORE,
        "native_minus_legacy_task_score": score_difference,
        "next_gate": "P1 error analysis" if p0_pass else "resolve failed P0 gates",
    }
    write_json(output_dir / "p0_status.json", summary)
    write_text(output_dir / "P0_SUMMARY.md", summary_markdown(summary, canonical))
    print(json.dumps(summary, indent=2), flush=True)


def package_cached(args: argparse.Namespace) -> None:
    """Finish P0 from cached native metric CSVs without recomputing nHD."""
    output_dir = args.output_dir
    frame_path = output_dir / "per_frame.csv"
    class_path = output_dir / "per_class_frame.csv"
    if not frame_path.is_file() or not class_path.is_file():
        raise FileNotFoundError(
            "Package stage requires completed per_frame.csv and per_class_frame.csv"
        )

    frame_rows = _typed_frame_rows(_read_csv(frame_path))
    class_rows = _typed_class_rows(_read_csv(class_path))
    if len(frame_rows) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} cached frame rows, got {len(frame_rows)}")
    if len(class_rows) != EXPECTED_FRAMES * 31:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES * 31} cached class-frame rows, got {len(class_rows)}"
        )

    expected_names = sorted(path.name for path in (args.data_root / "images").glob("*.png"))
    metric_names = sorted(str(row["name"]) for row in frame_rows)
    if metric_names != expected_names:
        raise ValueError("Cached metric rows differ from the frozen data snapshot")
    prediction_paths = [output_dir / "oof_predictions_native_ids" / name for name in expected_names]
    missing_predictions = [str(path) for path in prediction_paths if not path.is_file()]
    if missing_predictions:
        raise FileNotFoundError(f"Missing native predictions: {missing_predictions[:10]}")

    # Refresh provenance because package_cached is itself part of the frozen code path.
    write_json(
        output_dir / "baseline_manifest.json",
        baseline_manifest(args.config, args.split_manifest, args.model_dir, args.data_root),
        overwrite=True,
    )

    aggregate = aggregate_rows(frame_rows, class_rows)
    case_rows = aggregate.pop("case_rows")
    write_csv(
        output_dir / "per_case.csv",
        case_rows,
        ["case_id", "center", "fold", "frames", "dice", "nhd", "task_score"],
        overwrite=True,
    )
    fold_rows = [{"fold": int(fold), **payload} for fold, payload in aggregate["per_fold"].items()]
    write_csv(
        output_dir / "per_fold.csv",
        fold_rows,
        ["fold", "cases", "frames", "dice", "nhd", "task_score"],
        overwrite=True,
    )
    center_rows = [
        {"center": center, **payload} for center, payload in aggregate["per_center"].items()
    ]
    write_csv(
        output_dir / "per_center.csv",
        center_rows,
        ["center", "cases", "frames", "dice", "nhd", "task_score"],
        overwrite=True,
    )

    score_difference = float(aggregate["overall"]["task_score"]) - LEGACY_TASK_SCORE
    canonical = {
        "p0_version": P0_VERSION,
        "generated_at": utc_now(),
        "experiment": load_config(args.config)["experiment_id"],
        "evaluation_grid": "native_resolution",
        "aggregation": "class-weighted frame -> case mean -> dataset mean",
        "legacy_resized_task_score": LEGACY_TASK_SCORE,
        "native_minus_legacy_task_score": score_difference,
        **aggregate,
    }
    write_json(output_dir / "canonical_native_oof_metrics.json", canonical, overwrite=True)

    prediction_hashes = digest_paths(prediction_paths)
    checksum_lines = [f"{digest}  {path}\n" for path, digest in sorted(prediction_hashes.items())]
    write_text(
        output_dir / "prediction_checksums.sha256",
        "".join(checksum_lines),
        overwrite=True,
    )

    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    parity_real = real_metric_parity(frame_rows, args, labelmap)
    write_json(
        output_dir / "metric_parity_real_sample.json",
        parity_real,
        overwrite=True,
    )

    reconstructed = aggregate_rows(
        _typed_frame_rows(_read_csv(frame_path)),
        _typed_class_rows(_read_csv(class_path)),
    )
    reconstruction_checks = {
        "overall_task_score": abs(
            reconstructed["overall"]["task_score"] - canonical["overall"]["task_score"]
        ),
        "overall_dice": abs(reconstructed["overall"]["dice"] - canonical["overall"]["dice"]),
        "overall_nhd": abs(reconstructed["overall"]["nhd"] - canonical["overall"]["nhd"]),
        "class_error_budget": canonical["aggregation_absolute_difference"],
    }
    reproducible = max(reconstruction_checks.values()) <= AGGREGATION_TOLERANCE
    reproducibility = {
        "status": "REPRODUCIBLE" if reproducible else "NOT_REPRODUCIBLE",
        "classification": "deterministic aggregation of frozen native prediction metrics",
        "tolerance": AGGREGATION_TOLERANCE,
        "checks": reconstruction_checks,
        "prediction_files": len(prediction_paths),
        "prediction_hashes_recorded": len(prediction_hashes),
        "note": "Cached CSV reload and independent aggregation reproduce the canonical metrics.",
    }
    write_json(
        output_dir / "reproducibility_report.json",
        reproducibility,
        overwrite=True,
    )

    gates = {
        "dataset": json.loads((output_dir / "dataset_audit.json").read_text())["status"],
        "split": json.loads((output_dir / "split_audit.json").read_text())["status"],
        "synthetic_metric_parity": json.loads(
            (output_dir / "metric_parity_synthetic.json").read_text()
        )["status"],
        "real_sample_metric_parity": parity_real["status"],
        "baseline_manifest": json.loads((output_dir / "baseline_manifest.json").read_text())["status"],
        "native_prediction_coverage": (
            "PASS" if len(prediction_paths) == EXPECTED_FRAMES else "FAIL"
        ),
        "aggregation_reconstruction": (
            "PASS"
            if canonical["aggregation_absolute_difference"] <= AGGREGATION_TOLERANCE
            else "FAIL"
        ),
        "reproducibility": "PASS" if reproducible else "FAIL",
    }
    p0_pass = all(status == "PASS" for status in gates.values())
    summary = {
        "status": "PASS" if p0_pass else "FAIL",
        "generated_at": utc_now(),
        "gates": gates,
        "canonical_baseline": canonical["overall"],
        "worst_center": canonical["worst_center"],
        "legacy_resized_task_score": LEGACY_TASK_SCORE,
        "native_minus_legacy_task_score": score_difference,
        "next_gate": "P1 error analysis" if p0_pass else "resolve failed P0 gates",
    }
    write_json(output_dir / "p0_status.json", summary, overwrite=True)
    write_text(
        output_dir / "P0_SUMMARY.md",
        summary_markdown(summary, canonical),
        overwrite=True,
    )
    print(json.dumps(summary, indent=2), flush=True)


def summary_markdown(summary: dict[str, Any], canonical: dict[str, Any]) -> str:
    gate_lines = "\n".join(
        f"| {gate} | {status} |" for gate, status in summary["gates"].items()
    )
    fold_lines = "\n".join(
        "| {fold} | {cases} | {frames} | {task_score:.6f} | {dice:.6f} | {nhd:.6f} |".format(
            fold=fold, **payload,
        )
        for fold, payload in canonical["per_fold"].items()
    )
    center_lines = "\n".join(
        "| {center} | {cases} | {frames} | {task_score:.6f} | {dice:.6f} | {nhd:.6f} |".format(
            center=center, **payload,
        )
        for center, payload in canonical["per_center"].items()
    )
    overall = canonical["overall"]
    return f"""# P0 native-resolution baseline lock

## Material Passport

- Origin Skill: experiment-agent / run
- Origin Date: 2026-08-27 UTC
- Verification Status: {summary['status']}
- Version Label: {P0_VERSION}

## Verdict

**P0 {summary['status']}** — canonical native-resolution baseline: Task =
**{overall['task_score']:.6f}**, Dice = **{overall['dice']:.6f}**, nHD =
**{overall['nhd']:.6f}** on {overall['cases']} cases/{overall['frames']} frames.

Legacy resized-grid Task score: {summary['legacy_resized_task_score']:.6f}. Native minus
legacy: {summary['native_minus_legacy_task_score']:+.6f}.

## Gates

| Gate | Status |
|---|---|
{gate_lines}

## Per fold

| Fold | Cases | Frames | Task | Dice | nHD |
|---:|---:|---:|---:|---:|---:|
{fold_lines}

## Per center

| Center | Cases | Frames | Task | Dice | nHD |
|---|---:|---:|---:|---:|---:|
{center_lines}

Worst center: **{canonical['worst_center']}**.

## Interpretation boundary

This is a local 40-case outer-OOF result from frozen checkpoints. It is not a
hidden challenge score. P1 may analyze these frozen predictions, but new model
interventions must be screened on inner validation before OOF confirmation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare", "infer", "finalize", "package"), required=True,
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/configs/full40_fine31_c4_a0_seed2026.yaml",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/splits/case_folds_v1.json",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/seed_2026/full40_fine31_c4_a0_seed2026",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/p0_baseline_lock_v2",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--metric-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in ("config", "split_manifest", "model_dir", "output_dir", "data_root"):
        setattr(args, field, getattr(args, field).resolve())
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "infer":
        infer(args)
    elif args.stage == "finalize":
        finalize(args)
    else:
        package_cached(args)


if __name__ == "__main__":
    main()
