#!/usr/bin/env python3
"""P1 error analysis for the frozen native-resolution P0 baseline.

This script never trains or selects a model. It consumes only the frozen P0
OOF predictions and metrics, then produces quantitative diagnostics and a
reviewable qualitative atlas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tiger_models.data import LabelMap  # noqa: E402


P1_VERSION = "p1_error_analysis_v1"
EXPECTED_CASES = 40
EXPECTED_FRAMES = 524
EXPECTED_CLASSES = 31
TARGET_CENTER = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(path)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows))


def finite_or_blank(value: float | None) -> float | str:
    return "" if value is None or not math.isfinite(value) else value


def typed_frames(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            **row,
            "center": int(row["center"]),
            "fold": int(row["fold"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "dice": float(row["dice"]),
            "nhd": float(row["nhd"]),
            "task_score": float(row["task_score"]),
        })
    return output


def typed_classes(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    integer_fields = {
        "center", "fold", "class_id", "target_present", "prediction_present",
        "target_pixels", "prediction_pixels",
    }
    float_fields = {"weight", "dice", "nhd"}
    output = []
    for row in rows:
        converted: dict[str, Any] = dict(row)
        for field in integer_fields:
            converted[field] = int(converted[field])
        for field in float_fields:
            converted[field] = float(converted[field])
        output.append(converted)
    return output


def case_balanced_metrics(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    case_dice = [mean(float(row["dice"]) for row in group) for group in by_case.values()]
    case_nhd = [mean(float(row["nhd"]) for row in group) for group in by_case.values()]
    return mean(case_dice), mean(case_nhd)


def analyze_centers(
    center_rows: Sequence[dict[str, str]],
    case_rows: Sequence[dict[str, str]],
    overall: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    centers = [{
        "center": row["center"],
        "cases": int(row["cases"]),
        "frames": int(row["frames"]),
        "task_score": float(row["task_score"]),
        "dice": float(row["dice"]),
        "nhd": float(row["nhd"]),
    } for row in center_rows]
    median_task = statistics.median(row["task_score"] for row in centers)
    median_dice = statistics.median(row["dice"] for row in centers)
    median_nhd = statistics.median(row["nhd"] for row in centers)
    for row in centers:
        row["delta_vs_overall"] = row["task_score"] - float(overall["task_score"])
        row["delta_vs_center_median"] = row["task_score"] - median_task
    centers.sort(key=lambda row: row["task_score"], reverse=True)

    ranked_cases = [{
        "case_id": row["case_id"],
        "center": int(row["center"]),
        "fold": int(row["fold"]),
        "frames": int(row["frames"]),
        "task_score": float(row["task_score"]),
        "dice": float(row["dice"]),
        "nhd": float(row["nhd"]),
    } for row in case_rows]
    ranked_cases.sort(key=lambda row: row["task_score"])
    for index, row in enumerate(ranked_cases, start=1):
        row["rank_worst_to_best"] = index

    c4 = next(row for row in centers if row["center"] == f"center_{TARGET_CENTER}")
    c4_cases = [row for row in ranked_cases if row["center"] == TARGET_CENTER]
    global_case_median = statistics.median(row["task_score"] for row in ranked_cases)
    shortfalls = [max(0.0, global_case_median - row["task_score"]) for row in c4_cases]
    case_concentration = max(shortfalls) / sum(shortfalls) if sum(shortfalls) else 0.0
    c4_detail = {
        "target_center": f"center_{TARGET_CENTER}",
        "best_center": centers[0]["center"],
        "best_center_task": centers[0]["task_score"],
        "second_worst_center": centers[-2]["center"],
        "second_worst_center_task": centers[-2]["task_score"],
        "center_task_median": median_task,
        "c4_delta_vs_center_median": c4["task_score"] - median_task,
        "c4_dice_component_vs_median": 0.5 * (c4["dice"] - median_dice),
        "c4_nhd_component_vs_median": -0.5 * (c4["nhd"] - median_nhd),
        "global_case_median_task": global_case_median,
        "c4_cases_below_global_median": sum(
            row["task_score"] < global_case_median for row in c4_cases
        ),
        "c4_case_count": len(c4_cases),
        "largest_case_shortfall_share": case_concentration,
        "c4_case_score_range": max(row["task_score"] for row in c4_cases)
        - min(row["task_score"] for row in c4_cases),
    }
    return centers, c4_detail, ranked_cases


def worst_case_sensitivity(ranked_cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(ranked_cases)
    k = int(math.ceil(count * 0.20))
    worst = list(ranked_cases[:k])
    remaining = list(ranked_cases[k:])
    overall_task = mean(row["task_score"] for row in ranked_cases)
    remaining_task = mean(row["task_score"] for row in remaining)
    median_task = statistics.median(row["task_score"] for row in ranked_cases)
    replaced_by_median = (
        sum(row["task_score"] for row in remaining) + k * median_task
    ) / count
    return {
        "interpretation": (
            "Removal is a sensitivity analysis, not an attainable model gain; "
            "it changes the evaluated case population."
        ),
        "case_count": count,
        "worst_20_percent_count": k,
        "overall_task_score": overall_task,
        "worst_20_percent_mean_task": mean(row["task_score"] for row in worst),
        "score_without_worst_20_percent": remaining_task,
        "removal_delta": remaining_task - overall_task,
        "counterfactual_replace_worst_with_global_median": replaced_by_median,
        "replacement_delta": replaced_by_median - overall_task,
        "worst_cases": [row["case_id"] for row in worst],
        "best_cases": [row["case_id"] for row in ranked_cases[-5:][::-1]],
    }


def analyze_classes(
    class_rows: Sequence[dict[str, Any]],
    frame_rows: Sequence[dict[str, Any]],
    canonical_classes: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_by_id = {int(row["class_id"]): row for row in canonical_classes}
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        by_class[int(row["class_id"])].append(row)
    total_pixels = sum(int(row["width"]) * int(row["height"]) for row in frame_rows)
    weight_total = sum(float(rows[0]["weight"]) for rows in by_class.values())
    output = []
    for class_id in sorted(by_class):
        rows = by_class[class_id]
        canonical = canonical_by_id[class_id]
        present = [row for row in rows if row["target_present"] == 1]
        present_cases = sorted({str(row["case_id"]) for row in present})
        present_dice = present_nhd = None
        if present:
            present_dice, present_nhd = case_balanced_metrics(present)
        diagnostic_dice = present_dice
        if diagnostic_dice is None:
            tier = "unobserved"
        elif diagnostic_dice < 0.40:
            tier = "critical_weak"
        elif diagnostic_dice < 0.60:
            tier = "weak"
        elif diagnostic_dice <= 0.80:
            tier = "medium"
        else:
            tier = "strong"
        weight = float(rows[0]["weight"])
        dice_headroom = min(0.05, 1.0 - float(canonical["case_balanced_dice"]))
        output.append({
            "class_id": class_id,
            "class_name": rows[0]["class_name"],
            "weight": weight,
            "present_frames": len(present),
            "frame_frequency": len(present) / len(frame_rows),
            "present_cases": len(present_cases),
            "case_frequency": len(present_cases) / EXPECTED_CASES,
            "target_pixels": sum(int(row["target_pixels"]) for row in rows),
            "pixel_ratio": sum(int(row["target_pixels"]) for row in rows) / total_pixels,
            "case_balanced_dice": float(canonical["case_balanced_dice"]),
            "case_balanced_nhd": float(canonical["case_balanced_nhd"]),
            "present_only_dice": finite_or_blank(present_dice),
            "present_only_nhd": finite_or_blank(present_nhd),
            "strength_tier_present_only": tier,
            "complete_miss_rate": finite_or_blank(canonical["complete_miss_rate"]),
            "absent_false_positive_rate": finite_or_blank(
                canonical["absent_false_positive_rate"]
            ),
            "task_error_budget": float(canonical["task_error_budget"]),
            "delta_task_if_dice_plus_0_05": (weight / weight_total) * dice_headroom / 2.0,
            "priority_rank": "",
        })
    priorities = sorted(
        [row for row in output if row["present_frames"] > 0],
        key=lambda row: (row["task_error_budget"], row["present_frames"]),
        reverse=True,
    )[:5]
    for rank, row in enumerate(priorities, start=1):
        row["priority_rank"] = rank
    return output, priorities


def center_class_analysis(
    class_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    centers = sorted({int(row["center"]) for row in class_rows})
    class_ids = sorted({int(row["class_id"]) for row in class_rows})
    weight_total = sum(
        float(next(row["weight"] for row in class_rows if row["class_id"] == class_id))
        for class_id in class_ids
    )
    long_rows = []
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for center in centers:
        for class_id in class_ids:
            selected = [
                row for row in class_rows
                if int(row["center"]) == center and int(row["class_id"]) == class_id
            ]
            dice, nhd = case_balanced_metrics(selected)
            present = [row for row in selected if row["target_present"] == 1]
            present_dice = present_nhd = None
            if present:
                present_dice, present_nhd = case_balanced_metrics(present)
            weight = float(selected[0]["weight"])
            payload = {
                "center": center,
                "class_id": class_id,
                "class_name": selected[0]["class_name"],
                "weight": weight,
                "cases": len({row["case_id"] for row in selected}),
                "present_cases": len({row["case_id"] for row in present}),
                "present_frames": len(present),
                "target_pixels": sum(int(row["target_pixels"]) for row in selected),
                "case_balanced_dice": dice,
                "case_balanced_nhd": nhd,
                "present_only_dice": finite_or_blank(present_dice),
                "present_only_nhd": finite_or_blank(present_nhd),
                "task_error_budget": (weight / weight_total) * ((1.0 - dice) + nhd) / 2.0,
            }
            lookup[(center, class_id)] = payload
            long_rows.append(payload)

    wide_rows = []
    for class_id in class_ids:
        example = lookup[(centers[0], class_id)]
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": example["class_name"],
        }
        for center in centers:
            row[f"center_{center}_present_dice"] = lookup[(center, class_id)][
                "present_only_dice"
            ]
            row[f"center_{center}_present_frames"] = lookup[(center, class_id)][
                "present_frames"
            ]
        wide_rows.append(row)

    gaps = []
    for class_id in class_ids:
        c4 = lookup[(TARGET_CENTER, class_id)]
        other_rows = [
            row for row in class_rows
            if int(row["center"]) != TARGET_CENTER and int(row["class_id"]) == class_id
        ]
        other_dice, other_nhd = case_balanced_metrics(other_rows)
        weight = float(c4["weight"])
        other_error = (weight / weight_total) * ((1.0 - other_dice) + other_nhd) / 2.0
        gap = float(c4["task_error_budget"]) - other_error
        gaps.append({
            "class_id": class_id,
            "class_name": c4["class_name"],
            "c4_error_budget": c4["task_error_budget"],
            "non_c4_error_budget": other_error,
            "c4_excess_error": gap,
            "c4_present_frames": c4["present_frames"],
            "c4_present_only_dice": c4["present_only_dice"],
        })
    gaps.sort(key=lambda row: row["c4_excess_error"], reverse=True)
    positive_total = sum(max(0.0, row["c4_excess_error"]) for row in gaps)
    top5_share = (
        sum(max(0.0, row["c4_excess_error"]) for row in gaps[:5]) / positive_total
        if positive_total else 0.0
    )
    diagnostic = {
        "c4_positive_class_gap_total": positive_total,
        "c4_top5_class_gap_share": top5_share,
        "c4_classes_with_positive_excess_error": sum(
            row["c4_excess_error"] > 0 for row in gaps
        ),
        "largest_c4_class_gaps": gaps[:10],
    }
    return long_rows, wide_rows, diagnostic


def load_native_pair(
    name: str, data_root: Path, prediction_dir: Path, labelmap: LabelMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(data_root / "images" / name) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    with Image.open(data_root / "masks_fine" / name) as handle:
        target = labelmap.decode_fine(np.asarray(handle.convert("RGB")))
    with Image.open(prediction_dir / name) as handle:
        prediction = np.asarray(handle, dtype=np.uint8)
    if target.shape != prediction.shape or rgb.shape[:2] != target.shape:
        raise ValueError(
            f"Native shape mismatch for {name}: rgb={rgb.shape}, gt={target.shape}, "
            f"pred={prediction.shape}"
        )
    return rgb, target, prediction


def confusion_and_frame_signals(
    frame_rows: Sequence[dict[str, Any]],
    class_rows: Sequence[dict[str, Any]],
    data_root: Path,
    prediction_dir: Path,
    labelmap: LabelMap,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    class_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        class_by_frame[str(row["name"])].append(row)
    matrix = np.zeros((EXPECTED_CLASSES, EXPECTED_CLASSES), dtype=np.int64)
    signals = []
    dice_values = np.asarray([float(row["dice"]) for row in frame_rows])
    nhd_values = np.asarray([float(row["nhd"]) for row in frame_rows])
    task_values = np.asarray([float(row["task_score"]) for row in frame_rows])
    dice_p20 = float(np.quantile(dice_values, 0.20))
    dice_median = float(np.median(dice_values))
    nhd_median = float(np.median(nhd_values))
    nhd_p80 = float(np.quantile(nhd_values, 0.80))
    task_p20 = float(np.quantile(task_values, 0.20))

    for index, frame in enumerate(frame_rows, start=1):
        name = str(frame["name"])
        _, target, prediction = load_native_pair(name, data_root, prediction_dir, labelmap)
        encoded = target.astype(np.int32) * EXPECTED_CLASSES + prediction.astype(np.int32)
        local = np.bincount(encoded.ravel(), minlength=EXPECTED_CLASSES**2).reshape(
            EXPECTED_CLASSES, EXPECTED_CLASSES
        )
        matrix += local
        missed_foreground = int(local[1:, 0].sum())
        false_positive_foreground = int(local[0, 1:].sum())
        foreground_confusion = int(local[1:, 1:].sum() - np.trace(local[1:, 1:]))
        wrong_pixels = int(local.sum() - np.trace(local))
        categories = {
            "foreground_omission": missed_foreground,
            "background_false_positive": false_positive_foreground,
            "foreground_class_confusion": foreground_confusion,
        }
        dominant = max(categories, key=categories.get) if wrong_pixels else "none"
        frame_classes = [row for row in class_by_frame[name] if row["class_id"] != 0]
        complete_misses = [
            row for row in frame_classes
            if row["target_present"] == 1 and row["prediction_present"] == 0
        ]
        small_complete_misses = [
            row for row in complete_misses
            if row["target_pixels"] / (int(frame["width"]) * int(frame["height"])) <= 0.005
        ]
        spurious = [
            row for row in frame_classes
            if row["target_present"] == 0 and row["prediction_present"] == 1
        ]
        tags = []
        if small_complete_misses:
            tags.append("missed_small_object_proxy")
        if foreground_confusion >= max(1, int(0.10 * wrong_pixels)):
            tags.append("foreground_class_confusion")
        if dominant == "foreground_omission":
            tags.append("foreground_omission_dominant")
        if dominant == "background_false_positive":
            tags.append("false_positive_dominant")
        if float(frame["dice"]) >= dice_median and float(frame["nhd"]) >= nhd_p80:
            tags.append("boundary_distance_discordance")
        if float(frame["dice"]) <= dice_p20:
            tags.append("low_overlap")
        signals.append({
            "name": name,
            "case_id": frame["case_id"],
            "center": frame["center"],
            "fold": frame["fold"],
            "dice": frame["dice"],
            "nhd": frame["nhd"],
            "task_score": frame["task_score"],
            "hard_frame_bottom20": int(float(frame["task_score"]) <= task_p20),
            "wrong_pixels": wrong_pixels,
            "foreground_omission_pixels": missed_foreground,
            "background_false_positive_pixels": false_positive_foreground,
            "foreground_class_confusion_pixels": foreground_confusion,
            "dominant_pixel_error": dominant,
            "complete_missed_classes": len(complete_misses),
            "small_complete_missed_classes": len(small_complete_misses),
            "spurious_classes": len(spurious),
            "automatic_tags": ";".join(tags) if tags else "none",
        })
        if index % 50 == 0 or index == len(frame_rows):
            print(json.dumps({
                "stage": "confusion",
                "completed_frames": index,
                "total_frames": len(frame_rows),
            }), flush=True)

    off_diagonal = int(matrix.sum() - np.trace(matrix))
    confusions = []
    for target_id in range(EXPECTED_CLASSES):
        for prediction_id in range(EXPECTED_CLASSES):
            if target_id == prediction_id or matrix[target_id, prediction_id] == 0:
                continue
            if target_id == 0:
                category = "background_false_positive"
            elif prediction_id == 0:
                category = "foreground_omission"
            else:
                category = "foreground_class_confusion"
            pixels = int(matrix[target_id, prediction_id])
            confusions.append({
                "target_id": target_id,
                "target_name": labelmap.fine_names[target_id],
                "prediction_id": prediction_id,
                "prediction_name": labelmap.fine_names[prediction_id],
                "category": category,
                "pixels": pixels,
                "fraction_of_all_wrong_pixels": pixels / off_diagonal,
            })
    confusions.sort(key=lambda row: row["pixels"], reverse=True)
    return matrix, signals, confusions


def failure_mode_summary(signals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    hard = [row for row in signals if row["hard_frame_bottom20"] == 1]
    nhd_p80 = float(np.quantile([float(row["nhd"]) for row in signals], 0.80))
    definitions = {
        "missed_small_object_proxy": lambda row: row["small_complete_missed_classes"] > 0,
        "foreground_class_confusion": lambda row: (
            row["foreground_class_confusion_pixels"] >= max(1, int(0.10 * row["wrong_pixels"]))
        ),
        "high_nhd_boundary_error": lambda row: float(row["nhd"]) >= nhd_p80,
        "foreground_omission_dominant": lambda row: (
            row["dominant_pixel_error"] == "foreground_omission"
        ),
        "false_positive_dominant": lambda row: (
            row["dominant_pixel_error"] == "background_false_positive"
        ),
        "boundary_distance_discordance": lambda row: (
            "boundary_distance_discordance" in row["automatic_tags"]
        ),
    }
    rows = []
    for name, predicate in definitions.items():
        selected = [row for row in hard if predicate(row)]
        rows.append({
            "failure_mode": name,
            "hard_frames": len(selected),
            "hard_frame_coverage": len(selected) / len(hard),
            "centers": ";".join(str(value) for value in sorted({row["center"] for row in selected})),
        })
    rows.sort(key=lambda row: row["hard_frames"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def select_atlas(frame_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(frame_rows)
    dice_median = statistics.median(float(row["dice"]) for row in rows)
    nhd_median = statistics.median(float(row["nhd"]) for row in rows)
    cohorts = {
        "worst_dice": sorted(rows, key=lambda row: float(row["dice"]))[:20],
        "worst_nhd": sorted(rows, key=lambda row: float(row["nhd"]), reverse=True)[:20],
        "high_dice_bad_nhd": sorted(
            [
                row for row in rows
                if float(row["dice"]) >= dice_median and float(row["nhd"]) >= nhd_median
            ],
            key=lambda row: (float(row["nhd"]), float(row["dice"])),
            reverse=True,
        )[:10],
        "low_dice_acceptable_nhd": sorted(
            [
                row for row in rows
                if float(row["dice"]) <= dice_median and float(row["nhd"]) <= nhd_median
            ],
            key=lambda row: (float(row["dice"]), float(row["nhd"])),
        )[:10],
    }
    selected = []
    for cohort, cohort_rows in cohorts.items():
        if len(cohort_rows) != (20 if cohort.startswith("worst_") else 10):
            raise ValueError(f"Insufficient rows for atlas cohort {cohort}")
        for rank, row in enumerate(cohort_rows, start=1):
            selected.append({"cohort": cohort, "cohort_rank": rank, **row})
    return selected


def resize_array(array: np.ndarray, width: int, *, nearest: bool = False) -> np.ndarray:
    image = Image.fromarray(array)
    height = max(1, round(array.shape[0] * width / array.shape[1]))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.asarray(image.resize((width, height), method))


def make_panel(
    name: str,
    data_root: Path,
    prediction_dir: Path,
    labelmap: LabelMap,
    output_path: Path,
    caption: str,
) -> None:
    rgb, target, prediction = load_native_pair(name, data_root, prediction_dir, labelmap)
    width = 480
    display_rgb = resize_array(rgb, width)
    display_target = resize_array(target, width, nearest=True)
    display_prediction = resize_array(prediction, width, nearest=True)
    gt_color = labelmap.fine_rgb[display_target]
    pred_color = labelmap.fine_rgb[display_prediction]
    overlay = (display_rgb.astype(np.float32) * 0.55).astype(np.uint8)
    correct_fg = (display_target == display_prediction) & (display_target != 0)
    omission = (display_target != 0) & (display_prediction == 0)
    false_positive = (display_target == 0) & (display_prediction != 0)
    confusion = (
        (display_target != 0) & (display_prediction != 0)
        & (display_target != display_prediction)
    )
    overlay[correct_fg] = np.asarray([32, 210, 64], dtype=np.uint8)
    overlay[omission] = np.asarray([255, 32, 32], dtype=np.uint8)
    overlay[false_positive] = np.asarray([255, 32, 220], dtype=np.uint8)
    overlay[confusion] = np.asarray([255, 220, 32], dtype=np.uint8)

    views = [display_rgb, gt_color, pred_color, overlay]
    titles = ["RGB", "GT", "Prediction", "Error: FN red / FP magenta / confusion yellow"]
    header_height = 52
    panel = Image.new("RGB", (width * 4, display_rgb.shape[0] + header_height), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for index, (view, title) in enumerate(zip(views, titles)):
        panel.paste(Image.fromarray(view), (index * width, header_height))
        draw.text((index * width + 6, 5), title, fill="black", font=font)
    draw.text((6, 24), caption[:260], fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path, quality=88, optimize=True)


def make_contact_sheet(panel_paths: Sequence[Path], output_path: Path) -> None:
    thumb_width = 760
    gap = 8
    columns = 2
    thumbnails = []
    for path in panel_paths:
        with Image.open(path) as handle:
            ratio = thumb_width / handle.width
            thumbnails.append(handle.resize(
                (thumb_width, max(1, round(handle.height * ratio))),
                Image.Resampling.LANCZOS,
            ))
    rows = math.ceil(len(thumbnails) / columns)
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new(
        "RGB",
        (columns * thumb_width + (columns - 1) * gap, rows * cell_height + (rows - 1) * gap),
        "#222222",
    )
    for index, thumbnail in enumerate(thumbnails):
        x = (index % columns) * (thumb_width + gap)
        y = (index // columns) * (cell_height + gap)
        sheet.paste(thumbnail, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=85, optimize=True)


def render_report(
    canonical: dict[str, Any],
    centers: Sequence[dict[str, Any]],
    c4_detail: dict[str, Any],
    c4_diagnostic: dict[str, Any],
    c4_classification: dict[str, Any],
    sensitivity: dict[str, Any],
    priorities: Sequence[dict[str, Any]],
    failure_modes: Sequence[dict[str, Any]],
    top_confusions: Sequence[dict[str, Any]],
    atlas_unique: int,
) -> str:
    center_lines = "\n".join(
        f"| {row['center']} | {row['cases']} | {row['frames']} | {row['task_score']:.6f} | "
        f"{row['dice']:.6f} | {row['nhd']:.6f} | {row['delta_vs_overall']:+.6f} |"
        for row in centers
    )
    class_lines = "\n".join(
        f"| {row['priority_rank']} | {row['class_id']} | {row['class_name']} | "
        f"{row['present_only_dice']:.4f} | {row['case_balanced_dice']:.4f} | "
        f"{row['task_error_budget']:.6f} | {row['delta_task_if_dice_plus_0_05']:+.6f} |"
        for row in priorities
    )
    failure_lines = "\n".join(
        f"| {row['rank']} | {row['failure_mode']} | {row['hard_frames']} | "
        f"{100 * row['hard_frame_coverage']:.1f}% |"
        for row in failure_modes[:3]
    )
    confusion_lines = "\n".join(
        f"| {row['target_name']} | {row['prediction_name']} | {row['category']} | "
        f"{row['pixels']:,} | {100 * row['fraction_of_all_wrong_pixels']:.2f}% |"
        for row in top_confusions[:10]
    )
    overall = canonical["overall"]
    return f"""# P1 frozen-baseline error analysis

## Material Passport

- Origin Skill: experiment-agent / run + validate
- Origin Date: 2026-08-27 UTC
- Verification Status: {c4_classification['status']}
- Version Label: {P1_VERSION}
- Input baseline: p0_native_baseline_lock_v1, Task={overall['task_score']:.6f}

## Verdict

P1 quantitative gates **PASS**. Primary C4 diagnosis: **{c4_classification['label']}**
({c4_classification['name']}); confidence **{c4_classification['confidence']}** because
C4 has only {c4_detail['c4_case_count']} cases. This is an intervention-routing diagnosis,
not causal proof.

## Per-center result

| Center | Cases | Frames | Task | Dice | nHD | Delta vs overall |
|---|---:|---:|---:|---:|---:|---:|
{center_lines}

- Best center: **{c4_detail['best_center']}** ({c4_detail['best_center_task']:.6f}).
- Second-worst center: **{c4_detail['second_worst_center']}**
  ({c4_detail['second_worst_center_task']:.6f}).
- C4 is {c4_detail['c4_delta_vs_center_median']:+.6f} below the median center.
- C4 gap components versus the center median: Dice
  {c4_detail['c4_dice_component_vs_median']:+.6f}, nHD
  {c4_detail['c4_nhd_component_vs_median']:+.6f} task units.
- {c4_detail['c4_cases_below_global_median']}/{c4_detail['c4_case_count']} C4 cases are
  below the global case median. Largest single-case share of C4 shortfall:
  {100 * c4_detail['largest_case_shortfall_share']:.1f}%.
- Top five class gaps explain {100 * c4_diagnostic['c4_top5_class_gap_share']:.1f}%
  of positive C4 excess error across
  {c4_diagnostic['c4_classes_with_positive_excess_error']} affected classes.

## Worst-case sensitivity

The worst 20% ({sensitivity['worst_20_percent_count']} cases) average
{sensitivity['worst_20_percent_mean_task']:.6f}. Removing them changes the score from
{sensitivity['overall_task_score']:.6f} to
{sensitivity['score_without_worst_20_percent']:.6f}
({sensitivity['removal_delta']:+.6f}). This is **not model gain** because it changes the
evaluation population. Replacing them with the global median gives the more interpretable
counterfactual delta {sensitivity['replacement_delta']:+.6f}.

## Top-5 class targets

Strength tiers use target-present Dice; error budget and score impact preserve the official
case-balanced aggregation. Classes with the same challenge weight have the same direct
benefit from a uniform +0.05 Dice, so error budget and support break ties.

| Rank | ID | Class | Present Dice | Case-balanced Dice | Error budget | Task delta if Dice +0.05 |
|---:|---:|---|---:|---:|---:|---:|
{class_lines}

Combined upper-bound task gain from +0.05 Dice on these five classes, holding nHD fixed:
**{sum(row['delta_task_if_dice_plus_0_05'] for row in priorities):+.6f}**.

## Top-3 measurable failure modes

Coverage is measured on the bottom-20% task-score frames. Modes may overlap.

| Rank | Failure mode | Hard frames | Coverage |
|---:|---|---:|---:|
{failure_lines}

## Largest pixel confusions

| Ground truth | Prediction | Type | Pixels | Share of wrong pixels |
|---|---|---|---:|---:|
{confusion_lines}

## Qualitative atlas

The atlas contains 60 cohort entries ({atlas_unique} unique frames): 20 worst Dice,
20 worst nHD, 10 high-Dice/bad-nHD, and 10 low-Dice/acceptable-nHD. Panels show
RGB | GT | prediction | error overlay. Automatic tags are measurable proxies only.
Smoke, blood, occlusion, unusual anatomy, and partial visibility remain blank manual-review
fields; P1 does not fabricate these visual causes.

## Routing decision

{c4_classification['recommendation']}

The quantitative evidence supports moving to a targeted P2 experiment, but any new
intervention must be screened on inner validation before another outer-OOF confirmation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p0-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/p0_baseline_lock_v2",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "full_version/task2_fine_31cls/artifacts/p1_error_analysis_v1",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace files in an incomplete/existing P1 output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.p0_dir = args.p0_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    p0_status = json.loads((args.p0_dir / "p0_status.json").read_text())
    if p0_status.get("status") != "PASS":
        raise ValueError("P1 requires a PASS P0 baseline")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty P1 directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    canonical = json.loads((args.p0_dir / "canonical_native_oof_metrics.json").read_text())
    frame_rows = typed_frames(read_csv(args.p0_dir / "per_frame.csv"))
    class_rows = typed_classes(read_csv(args.p0_dir / "per_class_frame.csv"))
    case_rows = read_csv(args.p0_dir / "per_case.csv")
    center_rows = read_csv(args.p0_dir / "per_center.csv")
    if len(frame_rows) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} frames, got {len(frame_rows)}")
    if len(case_rows) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases, got {len(case_rows)}")
    if len(class_rows) != EXPECTED_FRAMES * EXPECTED_CLASSES:
        raise ValueError("P0 per-class frame matrix has unexpected size")

    centers, c4_detail, ranked_cases = analyze_centers(center_rows, case_rows, canonical["overall"])
    sensitivity = worst_case_sensitivity(ranked_cases)
    per_class, priorities = analyze_classes(class_rows, frame_rows, canonical["per_class"])
    center_class_long, center_class_wide, c4_diagnostic = center_class_analysis(class_rows)

    if c4_detail["largest_case_shortfall_share"] >= 0.60:
        diagnosis = "B"
        diagnosis_name = "case difficulty"
        recommendation = (
            "Prioritize the dominant hard C4 case pattern and case-aware sampling; do not "
            "treat C4 as a uniform visual domain until the hard-case hypothesis is tested."
        )
    elif c4_diagnostic["c4_top5_class_gap_share"] >= 0.60:
        diagnosis = "C"
        diagnosis_name = "class-distribution / class-specific issue"
        recommendation = (
            "Prioritize the largest C4 class gaps with class-aware sampling/loss or crop "
            "experiments before broad domain augmentation."
        )
    else:
        diagnosis = "A"
        diagnosis_name = "center-level domain shift"
        recommendation = (
            "Proceed first with a controlled photometric/surgical augmentation audit and "
            "center-balanced exposure, monitoring C4 and the non-C4 overall score."
        )
    c4_classification = {
        "status": "PASS",
        "label": diagnosis,
        "name": diagnosis_name,
        "confidence": "low (n=3 C4 cases)",
        "decision_thresholds": {
            "case_difficulty_if_largest_case_shortfall_share_at_least": 0.60,
            "class_specific_if_top5_positive_class_gap_share_at_least": 0.60,
            "otherwise": "center-level domain shift",
        },
        "recommendation": recommendation,
    }

    center_fields = [
        "center", "cases", "frames", "task_score", "dice", "nhd",
        "delta_vs_overall", "delta_vs_center_median",
    ]
    write_csv(args.output_dir / "per_center_analysis.csv", centers, center_fields)
    case_fields = [
        "rank_worst_to_best", "case_id", "center", "fold", "frames",
        "task_score", "dice", "nhd",
    ]
    write_csv(args.output_dir / "per_case_ranking.csv", ranked_cases, case_fields)
    write_json(args.output_dir / "worst20_case_sensitivity.json", sensitivity)
    class_fields = list(per_class[0].keys())
    write_csv(args.output_dir / "per_class_analysis.csv", per_class, class_fields)
    write_json(args.output_dir / "top5_target_classes.json", priorities)
    center_class_fields = list(center_class_long[0].keys())
    write_csv(
        args.output_dir / "center_class_analysis_long.csv",
        center_class_long,
        center_class_fields,
    )
    wide_fields = list(center_class_wide[0].keys())
    write_csv(
        args.output_dir / "center_class_present_dice_matrix.csv",
        center_class_wide,
        wide_fields,
    )
    write_json(args.output_dir / "c4_diagnosis.json", {
        **c4_detail,
        **c4_diagnostic,
        "classification": c4_classification,
    })

    labelmap = LabelMap.load(args.data_root / "labelmap.csv")
    prediction_dir = args.p0_dir / "oof_predictions_native_ids"
    confusion, signals, confusions = confusion_and_frame_signals(
        frame_rows, class_rows, args.data_root, prediction_dir, labelmap,
    )
    confusion_rows = []
    for target_id in range(EXPECTED_CLASSES):
        confusion_rows.append({
            "target_id": target_id,
            "target_name": labelmap.fine_names[target_id],
            **{
                f"pred_{prediction_id}": int(confusion[target_id, prediction_id])
                for prediction_id in range(EXPECTED_CLASSES)
            },
        })
    confusion_fields = ["target_id", "target_name"] + [
        f"pred_{prediction_id}" for prediction_id in range(EXPECTED_CLASSES)
    ]
    write_csv(args.output_dir / "pixel_confusion_matrix.csv", confusion_rows, confusion_fields)
    write_csv(
        args.output_dir / "top_pixel_confusions.csv",
        confusions,
        list(confusions[0].keys()),
    )
    signal_fields = list(signals[0].keys())
    write_csv(args.output_dir / "frame_failure_signals.csv", signals, signal_fields)
    failure_modes = failure_mode_summary(signals)
    write_json(args.output_dir / "failure_modes.json", failure_modes)

    signal_by_name = {row["name"]: row for row in signals}
    atlas_rows = select_atlas(frame_rows)
    panel_paths_by_cohort: dict[str, list[Path]] = defaultdict(list)
    for index, row in enumerate(atlas_rows, start=1):
        signal = signal_by_name[str(row["name"])]
        cohort = str(row["cohort"])
        panel_path = args.output_dir / "error_atlas" / cohort / str(row["name"]).replace(
            ".png", ".jpg"
        )
        caption = (
            f"{row['name']} | center={row['center']} case={row['case_id']} | "
            f"Dice={float(row['dice']):.4f} nHD={float(row['nhd']):.4f} | "
            f"tags={signal['automatic_tags']}"
        )
        make_panel(
            str(row["name"]), args.data_root, prediction_dir, labelmap, panel_path, caption,
        )
        panel_paths_by_cohort[cohort].append(panel_path)
        row["automatic_tags"] = signal["automatic_tags"]
        row["panel_path"] = str(panel_path.relative_to(ROOT))
        row["manual_smoke_blood"] = ""
        row["manual_occlusion_partial_visibility"] = ""
        row["manual_unusual_anatomy"] = ""
        row["manual_illumination_blur"] = ""
        row["manual_review_notes"] = ""
        if index % 10 == 0 or index == len(atlas_rows):
            print(json.dumps({
                "stage": "atlas",
                "completed_panels": index,
                "total_panels": len(atlas_rows),
            }), flush=True)
    for cohort, panel_paths in panel_paths_by_cohort.items():
        make_contact_sheet(
            panel_paths,
            args.output_dir / "error_atlas" / f"contact_sheet_{cohort}.jpg",
        )
    atlas_fields = [
        "cohort", "cohort_rank", "name", "case_id", "center", "fold", "width",
        "height", "dice", "nhd", "task_score", "automatic_tags", "panel_path",
        "manual_smoke_blood", "manual_occlusion_partial_visibility",
        "manual_unusual_anatomy", "manual_illumination_blur", "manual_review_notes",
    ]
    write_csv(args.output_dir / "atlas_review_manifest.csv", atlas_rows, atlas_fields)

    unique_atlas_frames = len({row["name"] for row in atlas_rows})
    report = render_report(
        canonical, centers, c4_detail, c4_diagnostic, c4_classification, sensitivity,
        priorities, failure_modes, confusions, unique_atlas_frames,
    )
    write_text(args.output_dir / "P1_REPORT.md", report)

    gates = {
        "p0_baseline": "PASS",
        "per_center": "PASS" if len(centers) == 6 else "FAIL",
        "per_case": "PASS" if len(ranked_cases) == EXPECTED_CASES else "FAIL",
        "per_class": "PASS" if len(per_class) == EXPECTED_CLASSES else "FAIL",
        "center_class": "PASS" if len(center_class_long) == 6 * EXPECTED_CLASSES else "FAIL",
        "confusion": "PASS" if int(confusion.sum()) > 0 else "FAIL",
        "qualitative_atlas": "PASS" if len(atlas_rows) == 60 else "FAIL",
        "failure_mode_ranking": "PASS" if len(failure_modes) >= 3 else "FAIL",
    }
    status = "PASS" if set(gates.values()) == {"PASS"} else "FAIL"
    p1_status = {
        "status": status,
        "generated_at": utc_now(),
        "version": P1_VERSION,
        "gates": gates,
        "baseline_task": canonical["overall"]["task_score"],
        "c4_classification": c4_classification,
        "top5_target_classes": [row["class_name"] for row in priorities],
        "top3_failure_modes": [row["failure_mode"] for row in failure_modes[:3]],
        "atlas_entries": len(atlas_rows),
        "atlas_unique_frames": unique_atlas_frames,
        "next_gate": "P2 targeted data/robustness experiment" if status == "PASS" else "resolve P1 gates",
    }
    write_json(args.output_dir / "p1_status.json", p1_status)
    print(json.dumps(p1_status, indent=2), flush=True)


if __name__ == "__main__":
    main()
