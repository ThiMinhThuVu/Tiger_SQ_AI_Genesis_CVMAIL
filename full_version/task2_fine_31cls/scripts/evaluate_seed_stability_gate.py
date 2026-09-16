#!/usr/bin/env python3
"""Evaluate a fixed validation-only seed-stability gate.

This script deliberately accepts only per-fold validation artifacts. It does not
read OOF predictions or test metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_fold(root: Path, fold: int, require_completion: bool) -> dict[str, float]:
    fold_root = root / f"fold_{fold}"
    if require_completion:
        completion_path = fold_root / "completion.json"
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "completed":
            raise RuntimeError(f"Fold {fold} is not completed: {completion_path}")
    metrics_path = fold_root / "best_validation_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return {
        "task1_score": float(metrics["task1_score"]),
        "fine_dice": float(metrics["fine_dice"]),
        "fine_nhd": float(metrics["fine_nhd"]),
    }


def aggregate(rows: dict[int, dict[str, float]]) -> dict[str, float]:
    return {
        f"{metric}_mean": mean(row[metric] for row in rows.values())
        for metric in ("task1_score", "fine_dice", "fine_nhd")
    }


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    folds = list(dict.fromkeys(args.folds))
    if not folds:
        raise ValueError("At least one fold is required")

    candidate = {fold: load_fold(args.candidate_root, fold, True) for fold in folds}
    reference = {fold: load_fold(args.reference_root, fold, False) for fold in folds}
    baseline = {fold: load_fold(args.baseline_root, fold, False) for fold in folds}
    candidate_agg = aggregate(candidate)
    reference_agg = aggregate(reference)
    baseline_agg = aggregate(baseline)

    delta_reference = {
        metric: candidate_agg[metric] - reference_agg[metric]
        for metric in candidate_agg
    }
    fold_score_delta = {
        str(fold): candidate[fold]["task1_score"] - reference[fold]["task1_score"]
        for fold in folds
    }
    delta_baseline_score = (
        candidate_agg["task1_score_mean"] - baseline_agg["task1_score_mean"]
    )

    gates = {
        "mean_score_delta_vs_reference_ge_minus_0.005": (
            delta_reference["task1_score_mean"] >= -0.005
        ),
        "each_fold_score_delta_vs_reference_ge_minus_0.010": all(
            delta >= -0.010 for delta in fold_score_delta.values()
        ),
        "mean_score_delta_vs_p2_ge_0.002": delta_baseline_score >= 0.002,
        "mean_dice_delta_vs_reference_ge_minus_0.005": (
            delta_reference["fine_dice_mean"] >= -0.005
        ),
        "mean_nhd_delta_vs_reference_le_0.005": (
            delta_reference["fine_nhd_mean"] <= 0.005
        ),
    }
    gates["all_pass"] = all(gates.values())

    payload = {
        "candidate_root": str(args.candidate_root.resolve()),
        "reference_root": str(args.reference_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "folds": folds,
        "candidate_by_fold": candidate,
        "candidate_mean": candidate_agg,
        "reference_mean": reference_agg,
        "baseline_mean": baseline_agg,
        "delta_vs_reference_mean": delta_reference,
        "fold_score_delta_vs_reference": fold_score_delta,
        "delta_vs_p2_task1_score_mean": delta_baseline_score,
        "gates": gates,
        "oof_consulted": False,
        "verification_status": "ANALYZED",
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"output": str(args.output), "all_pass": gates["all_pass"]}))


if __name__ == "__main__":
    main()
