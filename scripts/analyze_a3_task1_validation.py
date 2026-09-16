#!/usr/bin/env python3
"""Validation-only diagnostics for SurgiGraph-Q A3 Task-1 ablations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def pearson(first, second):
    if len(first) < 2 or len(first) != len(second):
        return None
    first_mean, second_mean = mean(first), mean(second)
    numerator = sum((x-first_mean)*(y-second_mean) for x, y in zip(first, second))
    first_scale = math.sqrt(sum((x-first_mean)**2 for x in first))
    second_scale = math.sqrt(sum((y-second_mean)**2 for y in second))
    denominator = first_scale * second_scale
    return numerator / denominator if denominator else None


def load_folds(root: Path):
    metrics = []
    histories = []
    for fold in range(5):
        fold_root = root / f"fold_{fold}"
        metrics.append(json.loads((fold_root / "best_validation_metrics.json").read_text()))
        histories.extend(json.loads((fold_root / "epoch_metrics.json").read_text()))
    return metrics, histories


def class_rows(folds):
    by_class = {}
    for fold in folds:
        for row in fold["failure_metrics"]["per_class"]:
            by_class.setdefault(int(row["class_id"]), []).append(row)
    output = {}
    for class_id, rows in by_class.items():
        output[class_id] = {
            "class_id": class_id,
            "class_name": rows[0]["class_name"],
            "present_only_dice": mean(row["present_only_dice"] for row in rows),
            "complete_miss_rate": mean(row["complete_miss_rate"] for row in rows),
            "absent_fp_rate": mean(row["absent_fp_rate"] for row in rows),
        }
    return output


def summarize(name, metrics, histories):
    errors = [row["train_ot_marginal_error"] for row in histories
              if "train_ot_marginal_error" in row]
    scores = [row["val_task1_score"] for row in histories
              if "train_ot_marginal_error" in row]
    return {
        "name": name,
        "task1_score": mean(row["task1_score"] for row in metrics),
        "fine_dice": mean(row["fine_dice"] for row in metrics),
        "fine_nhd": mean(row["fine_nhd"] for row in metrics),
        "fold_scores": [row["task1_score"] for row in metrics],
        "marginal_error": {
            "mean": mean(errors),
            "max": max(errors) if errors else None,
            "over_1e_5": sum(error > 1e-5 for error in errors),
            "epochs": len(errors),
            "score_correlation": pearson(errors, scores),
        },
        "classes": class_rows(metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path,
                        default=ROOT / "artifacts/mask2former_research")
    parser.add_argument("--baseline", type=Path, default=ROOT /
                        "artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2")
    args = parser.parse_args()
    baseline_metrics, _ = load_folds(args.baseline)
    baseline = {
        "name": "improved_v2",
        "task1_score": mean((row["fine_dice"]+1-row["fine_nhd"])/2
                            for row in baseline_metrics),
        "fine_dice": mean(row["fine_dice"] for row in baseline_metrics),
        "fine_nhd": mean(row["fine_nhd"] for row in baseline_metrics),
        "fold_scores": [(row["fine_dice"]+1-row["fine_nhd"])/2
                        for row in baseline_metrics],
        "classes": class_rows(baseline_metrics),
    }
    variants = {}
    for short in ("c0", "m", "mf"):
        path = args.artifacts / f"mask2former_swin_small_surgigraph_q_a3_{short}_task1"
        metrics, histories = load_folds(path)
        variants[short] = summarize(short, metrics, histories)
        variants[short]["fold_delta_vs_baseline"] = [
            score-base for score, base in zip(variants[short]["fold_scores"],
                                              baseline["fold_scores"])]
    per_class = []
    for class_id, base in baseline["classes"].items():
        candidate = variants["mf"]["classes"][class_id]
        per_class.append({
            "class_id": class_id,
            "class_name": base["class_name"],
            "present_dice_delta": (candidate["present_only_dice"]-base["present_only_dice"]
                                   if candidate["present_only_dice"] is not None
                                   and base["present_only_dice"] is not None else None),
            "complete_miss_delta": (candidate["complete_miss_rate"]-base["complete_miss_rate"]
                                    if candidate["complete_miss_rate"] is not None
                                    and base["complete_miss_rate"] is not None else None),
            "absent_fp_delta": (candidate["absent_fp_rate"]-base["absent_fp_rate"]
                                if candidate["absent_fp_rate"] is not None
                                and base["absent_fp_rate"] is not None else None),
        })
    print(json.dumps({"split":"validation_only","test_data_used":False,
                      "baseline":baseline,"variants":variants,"a3_mf_per_class":per_class},
                     indent=2))


if __name__ == "__main__":
    main()
