#!/usr/bin/env python3
"""Assign a frozen, auditable component-error taxonomy to SAFE-Graph OOF data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_v1",
    )
    parser.add_argument("--labelmap", type=Path, default=ROOT / "data/labelmap.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--correct-dice-threshold", type=float, default=0.50)
    parser.add_argument("--swap-dominance-threshold", type=float, default=0.50)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def class_matching(prediction: np.ndarray, target: np.ndarray, class_id: int) -> dict:
    pred_labels, pred_count = components(prediction == class_id)
    gt_labels, gt_count = components(target == class_id)
    pred_area = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)[1:]
    gt_area = np.bincount(gt_labels.ravel(), minlength=gt_count + 1)[1:]
    overlap = np.zeros((pred_count, gt_count), dtype=np.int64)
    both = (pred_labels > 0) & (gt_labels > 0)
    if both.any():
        np.add.at(overlap, (pred_labels[both] - 1, gt_labels[both] - 1), 1)
    if pred_count and gt_count:
        dice = 2.0 * overlap / np.maximum(pred_area[:, None] + gt_area[None, :], 1)
        pred_indices, gt_indices = linear_sum_assignment(-dice)
        matched_pred_to_gt = {
            int(pred): int(gt) for pred, gt in zip(pred_indices, gt_indices) if overlap[pred, gt] > 0
        }
    else:
        dice = np.zeros((pred_count, gt_count), dtype=np.float64)
        matched_pred_to_gt = {}
    matched_gt_to_pred = {gt: pred for pred, gt in matched_pred_to_gt.items()}
    return {
        "pred_labels": pred_labels,
        "gt_labels": gt_labels,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "overlap": overlap,
        "dice": dice,
        "matched_pred_to_gt": matched_pred_to_gt,
        "matched_gt_to_pred": matched_gt_to_pred,
    }


def label_frame(
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: list[str],
    class_weights: np.ndarray,
    correct_dice_threshold: float,
    swap_dominance_threshold: float,
) -> tuple[list[dict], list[dict]]:
    predicted_rows: list[dict] = []
    gt_rows: list[dict] = []
    predicted_component_id = 0
    gt_component_id = 0
    for class_id in range(1, len(class_names)):
        match = class_matching(prediction, target, class_id)
        class_present = bool((target == class_id).any())
        for pred_index in range(match["pred_count"]):
            predicted_component_id += 1
            component = match["pred_labels"] == pred_index + 1
            area = int(match["pred_area"][pred_index])
            target_counts = np.bincount(target[component], minlength=len(class_names))
            dominant_gt = int(target_counts.argmax())
            dominant_fraction = float(target_counts[dominant_gt] / area)
            same_overlap = int(target_counts[class_id])
            best_same_gt = int(match["overlap"][pred_index].argmax()) if match["gt_count"] else None
            best_same_dice = (
                float(match["dice"][pred_index, best_same_gt]) if best_same_gt is not None else 0.0
            )
            matched_gt = match["matched_pred_to_gt"].get(pred_index)
            matched_dice = float(match["dice"][pred_index, matched_gt]) if matched_gt is not None else 0.0
            if matched_gt is not None and matched_dice >= correct_dice_threshold:
                error_label = "CORRECT_COMPONENT"
            elif matched_gt is not None:
                error_label = "BOUNDARY_OR_PARTIAL_ERROR"
            elif same_overlap > 0:
                error_label = "FRAGMENTED_PREDICTION"
            elif dominant_gt != 0 and dominant_gt != class_id and dominant_fraction >= swap_dominance_threshold:
                error_label = "CLASS_SWAP"
            elif not class_present:
                error_label = "ABSENT_CLASS_HALLUCINATION"
            elif dominant_gt == 0 and dominant_fraction >= swap_dominance_threshold:
                error_label = "BACKGROUND_HALLUCINATION"
            else:
                error_label = "AMBIGUOUS_FALSE_POSITIVE"
            risk_weight = max(float(class_weights[class_id]), float(class_weights[dominant_gt]))
            predicted_rows.append({
                "component_id": predicted_component_id,
                "predicted_class_id": class_id,
                "predicted_class_name": class_names[class_id],
                "area_pixels": area,
                "matched_gt_component_local_id": "" if matched_gt is None else matched_gt + 1,
                "matched_same_class_dice": matched_dice,
                "best_same_class_dice": best_same_dice,
                "same_class_overlap_pixels": same_overlap,
                "same_class_fraction_of_prediction": same_overlap / area,
                "dominant_gt_class_id": dominant_gt,
                "dominant_gt_class_name": class_names[dominant_gt],
                "dominant_gt_fraction": dominant_fraction,
                "error_label": error_label,
                "clinical_risk_weight": risk_weight,
                "dangerous_error_candidate": risk_weight >= 3 and error_label != "CORRECT_COMPONENT",
            })
        for gt_index in range(match["gt_count"]):
            gt_component_id += 1
            area = int(match["gt_area"][gt_index])
            matched_pred = match["matched_gt_to_pred"].get(gt_index)
            matched_dice = float(match["dice"][matched_pred, gt_index]) if matched_pred is not None else 0.0
            same_coverage = int(match["overlap"][:, gt_index].sum()) / area if match["pred_count"] else 0.0
            if matched_pred is not None and matched_dice >= correct_dice_threshold:
                error_label = "DETECTED_COMPONENT"
            elif matched_pred is not None:
                error_label = "PARTIAL_OR_BOUNDARY_MISS"
            elif same_coverage > 0:
                error_label = "FRAGMENTED_TARGET"
            else:
                error_label = "COMPLETE_MISS"
            risk_weight = float(class_weights[class_id])
            gt_rows.append({
                "gt_component_id": gt_component_id,
                "gt_class_id": class_id,
                "gt_class_name": class_names[class_id],
                "area_pixels": area,
                "matched_pred_component_local_id": "" if matched_pred is None else matched_pred + 1,
                "matched_same_class_dice": matched_dice,
                "same_class_prediction_coverage": same_coverage,
                "error_label": error_label,
                "clinical_risk_weight": risk_weight,
                "dangerous_error_candidate": risk_weight >= 3 and error_label != "DETECTED_COMPONENT",
            })
    return predicted_rows, gt_rows


def merge_raw_rows(raw_rows: list[dict[str, str]], labels: list[dict]) -> list[dict]:
    if len(raw_rows) != len(labels):
        raise ValueError(f"Raw/label component count mismatch: {len(raw_rows)} != {len(labels)}")
    output = []
    for raw, label in zip(raw_rows, labels):
        id_field = "component_id" if "component_id" in label else "gt_component_id"
        if int(raw[id_field]) != int(label[id_field]):
            raise ValueError(f"Component ordering mismatch for {id_field}")
        raw_without_old_label = {key: value for key, value in raw.items() if key != "error_label"}
        label_without_duplicates = {
            key: value for key, value in label.items()
            if key not in raw_without_old_label or key in {"error_label", "clinical_risk_weight", "dangerous_error_candidate"}
        }
        output.append({**raw_without_old_label, **label_without_duplicates})
    return output


def build_corpus(
    oof_root: Path,
    output_root: Path,
    class_names: list[str],
    class_weights: np.ndarray,
    correct_dice_threshold: float,
    swap_dominance_threshold: float,
) -> dict:
    all_predicted: list[dict] = []
    all_gt: list[dict] = []
    frame_summaries: list[dict] = []
    for fold in range(5):
        fold_dir = oof_root / f"fold_{fold}"
        frame_index = read_csv(fold_dir / "frame_index.csv")
        raw_predicted_by_frame: dict[str, list[dict]] = defaultdict(list)
        raw_gt_by_frame: dict[str, list[dict]] = defaultdict(list)
        for row in read_csv(fold_dir / "predicted_components_raw.csv"):
            raw_predicted_by_frame[row["frame"]].append(row)
        for row in read_csv(fold_dir / "gt_components_raw.csv"):
            raw_gt_by_frame[row["frame"]].append(row)
        for frame_row in frame_index:
            frame = frame_row["frame"]
            arrays = np.load(fold_dir / frame_row["array_file"])
            pred_labels, gt_labels = label_frame(
                arrays["prediction"], arrays["target"], class_names, class_weights,
                correct_dice_threshold, swap_dominance_threshold,
            )
            predicted = merge_raw_rows(raw_predicted_by_frame[frame], pred_labels)
            targets = merge_raw_rows(raw_gt_by_frame[frame], gt_labels)
            all_predicted.extend(predicted)
            all_gt.extend(targets)
            pred_counts = Counter(row["error_label"] for row in predicted)
            gt_counts = Counter(row["error_label"] for row in targets)
            frame_summaries.append({
                **frame_row,
                "predicted_error_counts_json": json.dumps(dict(sorted(pred_counts.items())), sort_keys=True),
                "gt_error_counts_json": json.dumps(dict(sorted(gt_counts.items())), sort_keys=True),
                "dangerous_predicted_error_components": sum(
                    str(row["dangerous_error_candidate"]).lower() == "true" for row in predicted
                ),
                "dangerous_gt_error_components": sum(
                    str(row["dangerous_error_candidate"]).lower() == "true" for row in targets
                ),
            })
    write_csv(output_root / "predicted_component_errors.csv", all_predicted)
    write_csv(output_root / "gt_component_errors.csv", all_gt)
    write_csv(output_root / "frame_error_summary.csv", frame_summaries)
    return {
        "frames": len(frame_summaries),
        "cases": sorted({row["case_id"] for row in frame_summaries}),
        "predicted_components": len(all_predicted),
        "gt_components": len(all_gt),
        "predicted_error_counts": dict(sorted(Counter(row["error_label"] for row in all_predicted).items())),
        "gt_error_counts": dict(sorted(Counter(row["error_label"] for row in all_gt).items())),
        "dangerous_predicted_error_components": sum(row["dangerous_error_candidate"] for row in all_predicted),
        "dangerous_gt_error_components": sum(row["dangerous_error_candidate"] for row in all_gt),
    }


def sensitivity_rows(
    oof_root: Path, class_names: list[str], class_weights: np.ndarray,
    swap_dominance_threshold: float,
) -> list[dict]:
    rows = []
    for threshold in (0.25, 0.50, 0.75):
        predicted_counts: Counter = Counter()
        gt_counts: Counter = Counter()
        for fold in range(5):
            fold_dir = oof_root / f"fold_{fold}"
            for frame_row in read_csv(fold_dir / "frame_index.csv"):
                arrays = np.load(fold_dir / frame_row["array_file"])
                predicted, targets = label_frame(
                    arrays["prediction"], arrays["target"], class_names, class_weights,
                    threshold, swap_dominance_threshold,
                )
                predicted_counts.update(row["error_label"] for row in predicted)
                gt_counts.update(row["error_label"] for row in targets)
        rows.append({
            "correct_dice_threshold": threshold,
            "predicted_error_counts_json": json.dumps(dict(sorted(predicted_counts.items())), sort_keys=True),
            "gt_error_counts_json": json.dumps(dict(sorted(gt_counts.items())), sort_keys=True),
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    if not 0 < args.correct_dice_threshold <= 1 or not 0 < args.swap_dominance_threshold <= 1:
        raise ValueError("Thresholds must be in (0, 1]")
    source_manifest = json.loads((args.oof_root / "manifest.json").read_text())
    if source_manifest.get("split") != "out_of_fold_validation_only" or source_manifest.get("test_data_used") is not False:
        raise ValueError("Error corpus requires validation-only, test-free OOF evidence")
    label_rows = read_csv(args.labelmap)
    class_names = [row["fine_name"] for row in label_rows]
    class_weights = np.asarray([float(row["weight"]) for row in label_rows])
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = build_corpus(
        args.oof_root, args.output_root, class_names, class_weights,
        args.correct_dice_threshold, args.swap_dominance_threshold,
    )
    write_csv(
        args.output_root / "threshold_sensitivity.csv",
        sensitivity_rows(args.oof_root, class_names, class_weights, args.swap_dominance_threshold),
    )
    manifest = {
        "artifact": "SAFE-Graph OOF component error corpus",
        "version": "improved_v2_validation_error_v1",
        "source_oof_root": str(args.oof_root),
        "source_manifest_sha256": file_sha256(args.oof_root / "manifest.json"),
        "source_split": source_manifest["split"],
        "test_data_used": False,
        "primary_thresholds": {
            "correct_same_class_dice": args.correct_dice_threshold,
            "wrong_class_or_background_dominance": args.swap_dominance_threshold,
            "connectivity": 8,
            "matching": "class-wise one-to-one Hungarian maximizing component Dice; positive-overlap pairs only",
        },
        "sensitivity_thresholds": [0.25, 0.50, 0.75],
        "taxonomy_status": "FROZEN_FOR_EXPLORATORY_VERIFIER_V1",
        "clinical_status": "Dangerous-error candidates use labelmap weight>=3; expert taxonomy review remains required.",
        "station_prediction_source": "unavailable_in_improved_v2",
        "coverage_limitation": source_manifest["coverage_limitation"],
        **summary,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
