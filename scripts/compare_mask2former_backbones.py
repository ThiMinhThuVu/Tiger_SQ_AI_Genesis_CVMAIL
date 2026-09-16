#!/usr/bin/env python3
"""Create a paired 5-fold quantitative comparison for two Mask2Former runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd", "fine_pixel_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_complete(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("completed_test_folds") != 5 or len(payload.get("folds", [])) != 5:
        raise ValueError(f"Expected complete 5-fold result: {path}")
    folds = sorted(int(row["fold"]) for row in payload["folds"])
    if folds != list(range(5)):
        raise ValueError(f"Fold IDs must be 0..4 in {path}, got {folds}")
    return payload


def by_fold(payload: dict) -> dict[int, dict]:
    return {int(row["fold"]): row for row in payload["folds"]}


def summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    candidate = load_complete(args.candidate)
    baseline = load_complete(args.baseline)
    candidate_folds, baseline_folds = by_fold(candidate), by_fold(baseline)
    comparisons = {}
    for metric in METRICS:
        candidate_values = np.asarray([candidate_folds[fold][metric] for fold in range(5)], dtype=float)
        baseline_values = np.asarray([baseline_folds[fold][metric] for fold in range(5)], dtype=float)
        raw_delta = candidate_values - baseline_values
        higher_is_better = metric not in {"fine_nhd", "coarse_nhd"}
        improvement = raw_delta if higher_is_better else -raw_delta
        comparisons[metric] = {
            "higher_is_better": higher_is_better,
            "candidate": summary(candidate_values),
            "baseline": summary(baseline_values),
            "raw_candidate_minus_baseline": summary(raw_delta),
            "improvement": summary(improvement),
            "folds_improved": int((improvement > 0).sum()),
            "folds_tied": int(np.isclose(improvement, 0).sum()),
        }
    candidate_task1 = np.asarray([
        (candidate_folds[fold]["fine_dice"] + 1.0 - candidate_folds[fold]["fine_nhd"]) / 2.0
        for fold in range(5)
    ])
    baseline_task1 = np.asarray([
        (baseline_folds[fold]["fine_dice"] + 1.0 - baseline_folds[fold]["fine_nhd"]) / 2.0
        for fold in range(5)
    ])
    task1_delta = candidate_task1 - baseline_task1
    comparisons["task1_surrogate_internal"] = {
        "formula": "(fine_dice + 1 - fine_nhd) / 2",
        "candidate": summary(candidate_task1),
        "baseline": summary(baseline_task1),
        "improvement": summary(task1_delta),
        "folds_improved": int((task1_delta > 0).sum()),
        "folds_tied": int(np.isclose(task1_delta, 0).sum()),
    }
    output = {
        "candidate": candidate.get("experiment", str(args.candidate)),
        "baseline": baseline.get("experiment", str(args.baseline)),
        "paired_folds": 5,
        "comparisons": comparisons,
        "decision": {
            "candidate_task1_surrogate_better_mean": bool(task1_delta.mean() > 0),
            "candidate_task1_surrogate_wins_at_least_4_folds": bool((task1_delta > 0).sum() >= 4),
            "promote_over_baseline": bool(task1_delta.mean() > 0 and (task1_delta > 0).sum() >= 4),
        },
        "note": "Task-1 surrogate is internal and is not an official selection score.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
