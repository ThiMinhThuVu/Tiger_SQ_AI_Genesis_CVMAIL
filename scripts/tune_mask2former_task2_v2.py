#!/usr/bin/env python3
"""Error-driven, class-specific Task-2 tuning for Mask2Former Swin-Tiny."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as connected_components
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_mask2former import build_model, collate, semantic_logits
from scripts.tune_mask2former_task1_v2 import (
    apply_rules,
    complete_misses,
    metric_arrays,
    summarize_arrays,
    update_metric_arrays,
    write_csv,
)
from tiger_models.data import LabelMap, TigerMultitaskDataset, records_for_fold
from tiger_models.metrics import binary_dice, normalized_hausdorff, segmentation_metrics


VERSION = "mask2former_swin_tiny_task2_tuned_v2_error_driven"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mask2former_swin_tiny.yaml")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "artifacts/mask2former/mask2former_swin_tiny")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/mask2former" / VERSION)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--max-candidate-classes", type=int, default=8)
    parser.add_argument("--max-rules", type=int, default=4)
    parser.add_argument("--minimum-step-gain", type=float, default=0.0005)
    return parser.parse_args()


def prepare_prediction(semantic: torch.Tensor, labelmap: LabelMap) -> dict:
    values = semantic.float().clamp_min(0)
    fine_probability = (values / values.sum(0, keepdim=True).clamp_min(1e-8)).cpu().numpy()
    fine_base = fine_probability.argmax(0).astype(np.uint8)
    base = labelmap.fine_to_coarse[fine_base].astype(np.uint8)
    probability = np.zeros((len(labelmap.coarse_names), *base.shape), dtype=np.float32)
    for fine_id, coarse_id in enumerate(labelmap.fine_to_coarse):
        probability[int(coarse_id)] += fine_probability[fine_id]
    alternatives = probability.copy()
    np.put_along_axis(alternatives, base[None], -1.0, axis=0)
    second = alternatives.argmax(0).astype(np.uint8)
    flat_probability = probability.reshape(probability.shape[0], -1)
    flat_second = second.ravel()
    components: list[dict] = []
    for class_id in range(1, probability.shape[0]):
        labels, count = connected_components(base == class_id)
        for component_id in range(1, count + 1):
            flat_indices = np.flatnonzero(labels.ravel() == component_id)
            class_probability = flat_probability[class_id, flat_indices]
            second_probability = flat_probability[flat_second[flat_indices], flat_indices]
            components.append({
                "class_id": class_id,
                "component_id": component_id,
                "area": int(flat_indices.size),
                "mean_confidence": float(class_probability.mean()),
                "max_confidence": float(class_probability.max()),
                "mean_margin": float((class_probability - second_probability).mean()),
                "flat_indices": flat_indices,
            })
    return {"base": base, "second": second, "components": components}


@torch.no_grad()
def infer_split(model, config: dict, data_root: Path, labelmap: LabelMap, fold: int,
                split: str, device: torch.device) -> tuple[list[dict], list[np.ndarray], list[str], list[str]]:
    dataset = TigerMultitaskDataset(
        records_for_fold(data_root, fold, split), labelmap,
        config["height"], config["width"], training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    prepared, targets, cases, names = [], [], [], []
    for batch in loader:
        output = model(pixel_values=batch["image"].to(device))
        semantic = torch.nn.functional.interpolate(
            semantic_logits(output), size=batch["fine"].shape[-2:],
            mode="bilinear", align_corners=False,
        )[0]
        prepared.append(prepare_prediction(semantic, labelmap))
        fine_target = batch["fine"].numpy()[0]
        targets.append(labelmap.fine_to_coarse[fine_target].astype(np.uint8))
        cases.append(batch["case_id"][0])
        names.append(batch["name"][0])
    return prepared, targets, cases, names


def class_and_image_rows(fold: int, split: str, stage: str, predictions: list[np.ndarray],
                         targets: list[np.ndarray], cases: list[str], names: list[str],
                         labelmap: LabelMap) -> tuple[list[dict], list[dict]]:
    image_rows: list[dict] = []
    for prediction, target, case, name in zip(predictions, targets, cases, names):
        for class_id, class_name in enumerate(labelmap.coarse_names):
            pred_binary, target_binary = prediction == class_id, target == class_id
            tp = int(np.logical_and(pred_binary, target_binary).sum())
            fp = int(np.logical_and(pred_binary, ~target_binary).sum())
            fn = int(np.logical_and(~pred_binary, target_binary).sum())
            gt_present, pred_present = bool(target_binary.any()), bool(pred_binary.any())
            _, pred_components = connected_components(pred_binary)
            image_rows.append({
                "fold": fold, "split": split, "stage": stage, "case_id": case,
                "image": name, "class_id": class_id, "class_name": class_name,
                "weight": float(labelmap.coarse_weights[class_id]),
                "gt_present": int(gt_present), "pred_present": int(pred_present),
                "gt_area": int(target_binary.sum()), "pred_area": int(pred_binary.sum()),
                "pred_components": int(pred_components), "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "dice": binary_dice(pred_binary, target_binary),
                "nhd": normalized_hausdorff(pred_binary, target_binary),
                "absent_false_positive": int(not gt_present and pred_present),
                "complete_miss": int(gt_present and not pred_present),
            })
    class_rows: list[dict] = []
    for class_id, class_name in enumerate(labelmap.coarse_names):
        selected = [row for row in image_rows if row["class_id"] == class_id]
        present = sum(row["gt_present"] for row in selected)
        absent = len(selected) - present
        dice = float(np.mean([row["dice"] for row in selected]))
        nhd = float(np.mean([row["nhd"] for row in selected]))
        occurrence = present / len(selected)
        absent_fp_count = sum(row["absent_false_positive"] for row in selected)
        complete_miss_count = sum(row["complete_miss"] for row in selected)
        class_rows.append({
            "fold": fold, "split": split, "stage": stage, "class_id": class_id,
            "class_name": class_name, "weight": float(labelmap.coarse_weights[class_id]),
            "images": len(selected), "present_images": present,
            "occurrence_rate": occurrence, "mean_dice": dice, "mean_nhd": nhd,
            "absent_fp_count": absent_fp_count,
            "absent_fp_rate": absent_fp_count / absent if absent else 0.0,
            "complete_miss_count": complete_miss_count,
            "complete_miss_rate": complete_miss_count / present if present else 0.0,
            "error_priority": float(labelmap.coarse_weights[class_id] * occurrence
                                    * ((1.0 - dice) + nhd)),
        })
    return class_rows, image_rows


def component_rows(fold: int, split: str, prepared: list[dict], targets: list[np.ndarray],
                   cases: list[str], names: list[str], labelmap: LabelMap) -> list[dict]:
    rows: list[dict] = []
    for item, target, case, name in zip(prepared, targets, cases, names):
        distance_cache: dict[int, np.ndarray | None] = {}
        for component in item["components"]:
            class_id = component["class_id"]
            target_binary = target == class_id
            component_mask = np.zeros(target.shape, dtype=bool)
            component_mask.ravel()[component["flat_indices"]] = True
            overlap = int(np.logical_and(component_mask, target_binary).sum())
            if target_binary.any():
                if class_id not in distance_cache:
                    distance_cache[class_id] = distance_transform_edt(~target_binary)
                distance = distance_cache[class_id]
                assert distance is not None
                minimum_distance = float(distance[component_mask].min(initial=np.inf))
            else:
                distance_cache[class_id] = None
                minimum_distance = None
            rows.append({
                "fold": fold, "split": split, "case_id": case, "image": name,
                "class_id": class_id, "class_name": labelmap.coarse_names[class_id],
                "weight": float(labelmap.coarse_weights[class_id]),
                "component_id": component["component_id"], "area": component["area"],
                "mean_confidence": component["mean_confidence"],
                "max_confidence": component["max_confidence"],
                "mean_margin": component["mean_margin"], "overlap_gt_pixels": overlap,
                "is_false_positive_component": int(overlap == 0),
                "gt_class_present": int(target_binary.any()),
                "minimum_distance_to_gt": minimum_distance,
            })
    return rows


def confusion_rows(fold: int, split: str, stage: str, predictions: list[np.ndarray],
                   targets: list[np.ndarray], labelmap: LabelMap) -> list[dict]:
    classes = len(labelmap.coarse_names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        matrix += np.bincount(
            target.ravel() * classes + prediction.ravel(), minlength=classes * classes,
        ).reshape(classes, classes)
    rows = []
    for true_id in range(classes):
        for pred_id in range(classes):
            if matrix[true_id, pred_id]:
                rows.append({
                    "fold": fold, "split": split, "stage": stage,
                    "true_class_id": true_id,
                    "true_class_name": labelmap.coarse_names[true_id],
                    "pred_class_id": pred_id,
                    "pred_class_name": labelmap.coarse_names[pred_id],
                    "pixels": int(matrix[true_id, pred_id]),
                })
    return rows


def save_analysis(directory: Path, fold: int, split: str, stage: str,
                  prepared: list[dict], predictions: list[np.ndarray], targets: list[np.ndarray],
                  cases: list[str], names: list[str], labelmap: LabelMap,
                  include_components: bool = False) -> list[dict]:
    class_rows, image_rows = class_and_image_rows(
        fold, split, stage, predictions, targets, cases, names, labelmap,
    )
    write_csv(directory / f"{split}_{stage}_error_by_class.csv", class_rows)
    write_csv(directory / f"{split}_{stage}_error_by_image_class.csv", image_rows)
    write_csv(directory / f"{split}_{stage}_confusion.csv",
              confusion_rows(fold, split, stage, predictions, targets, labelmap))
    if include_components:
        write_csv(directory / f"{split}_baseline_components.csv",
                  component_rows(fold, split, prepared, targets, cases, names, labelmap))
    return class_rows


def tune_rules(prepared: list[dict], targets: list[np.ndarray], class_rows: list[dict],
               labelmap: LabelMap, args: argparse.Namespace) -> tuple[dict[int, dict], list[dict]]:
    rules: dict[int, dict] = {}
    current_predictions = [item["base"] for item in prepared]
    classes = len(labelmap.coarse_names)
    current_arrays = metric_arrays(current_predictions, targets, classes)
    current_summary = summarize_arrays(*current_arrays, labelmap.coarse_weights)
    current_misses = complete_misses(current_predictions, targets, labelmap.coarse_weights)
    eligible = sorted(
        (row for row in class_rows if row["class_id"] != 0 and row["absent_fp_count"] > 0),
        key=lambda row: (row["absent_fp_rate"] * row["weight"], row["error_priority"]),
        reverse=True,
    )[:args.max_candidate_classes]
    search: list[dict] = []
    candidate_parameters = [
        {"minimum_area": area, "minimum_confidence": confidence, "replacement": replacement}
        for area in (16, 64, 256)
        for confidence in (0.45, 0.60, 1.01)
        for replacement in ("second", "background")
    ]
    for step in range(args.max_rules):
        best = None
        for class_row in eligible:
            class_id = int(class_row["class_id"])
            if class_id in rules:
                continue
            for parameters in candidate_parameters:
                trial_rules = {**rules, class_id: parameters}
                predictions = [apply_rules(item, trial_rules) for item in prepared]
                arrays = update_metric_arrays(
                    current_predictions, predictions, targets, current_arrays,
                )
                summary = summarize_arrays(*arrays, labelmap.coarse_weights)
                frame_delta = summary["per_image_score"] - current_summary["per_image_score"]
                positive_frames = int((frame_delta > 1e-12).sum())
                negative_frames = int((frame_delta < -1e-12).sum())
                misses = complete_misses(predictions, targets, labelmap.coarse_weights)
                row = {
                    "step": step, "class_id": class_id,
                    "class_name": class_row["class_name"], **parameters,
                    "coarse_dice": summary["fine_dice"],
                    "coarse_nhd": summary["fine_nhd"],
                    "task2_score": summary["task1_score"],
                    "step_gain": summary["task1_score"] - current_summary["task1_score"],
                    "positive_frames": positive_frames, "negative_frames": negative_frames,
                    "protected_complete_misses": misses,
                    "eligible": int(
                        summary["task1_score"] >= current_summary["task1_score"] + args.minimum_step_gain
                        and misses <= current_misses and positive_frames >= negative_frames
                    ),
                }
                search.append(row)
                if row["eligible"] and (best is None or row["task2_score"] > best[0]["task2_score"]):
                    best = (row, parameters, predictions, arrays, summary, misses)
        if best is None:
            break
        row, parameters, current_predictions, current_arrays, current_summary, current_misses = best
        rules[int(row["class_id"])] = dict(parameters)
    return rules, search


def exact_metrics(predictions: list[np.ndarray], targets: list[np.ndarray], cases: list[str],
                  labelmap: LabelMap) -> dict:
    details = segmentation_metrics(
        predictions, targets, cases, labelmap.coarse_weights, labelmap.coarse_names,
    )
    accuracy = float(np.mean([(prediction == target).mean()
                              for prediction, target in zip(predictions, targets)]))
    return {
        "coarse_dice": details["dice"], "coarse_nhd": details["nhd"],
        "coarse_pixel_accuracy": accuracy,
        "task2_score": float((details["dice"] + 1.0 - details["nhd"]) / 2.0),
    }


def run_fold(args: argparse.Namespace, config: dict, labelmap: LabelMap,
             fold: int, device: torch.device) -> dict:
    checkpoint_path = args.model_dir / f"fold_{fold}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    fold_dir = args.output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True)

    val_prepared, val_targets, val_cases, val_names = infer_split(
        model, config, args.data_root, labelmap, fold, "validation", device,
    )
    val_baseline = [item["base"] for item in val_prepared]
    class_rows = save_analysis(
        fold_dir, fold, "validation", "baseline", val_prepared, val_baseline,
        val_targets, val_cases, val_names, labelmap, include_components=True,
    )
    rules, search = tune_rules(val_prepared, val_targets, class_rows, labelmap, args)
    val_tuned = [apply_rules(item, rules) for item in val_prepared]
    save_analysis(fold_dir, fold, "validation", "tuned", val_prepared, val_tuned,
                  val_targets, val_cases, val_names, labelmap)
    calibration = {
        "fold": fold, "calibration_split": "validation",
        "calibration_cases": sorted(set(val_cases)),
        "selected_rules": {str(key): value for key, value in rules.items()},
        "baseline": exact_metrics(val_baseline, val_targets, val_cases, labelmap),
        "tuned": exact_metrics(val_tuned, val_targets, val_cases, labelmap),
    }
    calibration["validation_gain"] = (
        calibration["tuned"]["task2_score"] - calibration["baseline"]["task2_score"]
    )
    (fold_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    write_csv(fold_dir / "validation_rule_search.csv", search)

    test_prepared, test_targets, test_cases, test_names = infer_split(
        model, config, args.data_root, labelmap, fold, "test", device,
    )
    baseline_predictions = [item["base"] for item in test_prepared]
    tuned_predictions = [apply_rules(item, rules) for item in test_prepared]
    save_analysis(fold_dir, fold, "test", "baseline", test_prepared, baseline_predictions,
                  test_targets, test_cases, test_names, labelmap, include_components=True)
    save_analysis(fold_dir, fold, "test", "tuned", test_prepared, tuned_predictions,
                  test_targets, test_cases, test_names, labelmap)
    prediction_dir = fold_dir / "test_predictions_coarse_ids"
    prediction_dir.mkdir()
    for name, prediction in zip(test_names, tuned_predictions):
        Image.fromarray(prediction, mode="L").save(prediction_dir / name)
    baseline_metrics = exact_metrics(baseline_predictions, test_targets, test_cases, labelmap)
    tuned_metrics = exact_metrics(tuned_predictions, test_targets, test_cases, labelmap)
    result = {
        "fold": fold, "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": checkpoint.get("epoch"),
        "calibration_cases": sorted(set(val_cases)), "test_cases": sorted(set(test_cases)),
        "selected_rules": {str(key): value for key, value in rules.items()},
        "validation_gain": calibration["validation_gain"],
        "baseline": baseline_metrics, "tuned": tuned_metrics,
        "test_gain": tuned_metrics["task2_score"] - baseline_metrics["task2_score"],
    }
    (fold_dir / "test_quantitative.json").write_text(json.dumps(result, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return result


def metric_summary(rows: list[dict], stage: str, key: str) -> dict:
    values = np.asarray([row[stage][key] for row in rows], dtype=float)
    return {"mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1))}


def main() -> None:
    args = parse_args()
    if args.output_dir.resolve() == args.model_dir.resolve():
        raise ValueError("Output directory must differ from source model directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-2 v2 experiment requires a CUDA node")
    config = yaml.safe_load(args.config.read_text())
    if config.get("experiment_id") not in {"mask2former_swin_tiny", "mask2former_swin_small"}:
        raise ValueError("This experiment supports Mask2Former Swin-Tiny/Small only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "version": args.output_dir.name, "source_experiment": config["experiment_id"],
        "tuning_scope": "Task 2 error-driven class-specific post-processing",
        "baseline_definition": "fine argmax mapped through labelmap fine_to_coarse",
        "test_ground_truth_used_for_tuning": False,
        "max_candidate_classes": args.max_candidate_classes,
        "max_rules": args.max_rules, "minimum_step_gain": args.minimum_step_gain,
        "candidate_areas": [16, 64, 256],
        "candidate_confidences": [0.45, 0.60, 1.01],
        "candidate_replacements": ["second", "background"],
    }
    (args.output_dir / "version.json").write_text(json.dumps(metadata, indent=2) + "\n")
    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    device = torch.device("cuda")
    folds = [run_fold(args, config, labelmap, fold, device) for fold in range(5)]
    keys = ("coarse_dice", "coarse_nhd", "coarse_pixel_accuracy", "task2_score")
    summary = {
        stage: {key: metric_summary(folds, stage, key) for key in keys}
        for stage in ("baseline", "tuned")
    }
    gains = np.asarray([row["test_gain"] for row in folds], dtype=float)
    selected_classes = Counter(class_id for row in folds for class_id in row["selected_rules"])
    output = {
        **metadata, "completed_test_folds": len(folds), "folds": folds,
        "summary": summary,
        "test_gain": {"mean": float(gains.mean()),
                      "sample_standard_deviation": float(gains.std(ddof=1))},
        "selected_class_frequency": dict(selected_classes),
        "acceptance": {
            "mean_task2_score_improved": bool(gains.mean() > 0),
            "folds_improved": int((gains > 0).sum()),
            "passes_four_of_five_rule": bool((gains > 0).sum() >= 4),
        },
    }
    (args.output_dir / "test_quantitative.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
