#!/usr/bin/env python3
"""Leakage-safe local warning baselines for the SAFE-Graph error corpus."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = [
    "risk_predicted_class_probability",
    "risk_top1_probability",
    "risk_margin",
    "entropy",
    "log_area_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-corpus", type=Path,
        default=ROOT / "artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1",
    )
    parser.add_argument("--output-root", type=Path, required=True)
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


def parse_bool(value: str | bool) -> bool:
    return value is True or str(value).strip().lower() == "true"


def features(row: dict[str, str]) -> list[float]:
    area_fraction = max(float(row["area_fraction"]), 1e-12)
    return [
        1.0 - float(row["mean_predicted_class_probability"]),
        1.0 - float(row["mean_top1_probability"]),
        1.0 - float(row["mean_top1_top2_margin"]),
        float(row["mean_entropy"]),
        math.log(area_fraction),
    ]


def binary_metrics(target: np.ndarray, risk: np.ndarray) -> dict[str, float | int | None]:
    if len(target) != len(risk) or not len(target):
        raise ValueError("Target/risk arrays must be non-empty and aligned")
    unique = np.unique(target)
    return {
        "components": len(target),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "auprc": float(average_precision_score(target, risk)) if len(unique) == 2 else None,
        "auroc": float(roc_auc_score(target, risk)) if len(unique) == 2 else None,
        "brier": float(brier_score_loss(target, np.clip(risk, 0, 1))),
    }


def targets(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "dangerous_error": np.asarray([parse_bool(row["dangerous_error_candidate"]) for row in rows], dtype=np.uint8),
        "any_component_error": np.asarray([row["error_label"] != "CORRECT_COMPONENT" for row in rows], dtype=np.uint8),
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    manifest = json.loads((args.error_corpus / "manifest.json").read_text())
    if manifest.get("test_data_used") is not False or manifest.get("taxonomy_status") != "FROZEN_FOR_EXPLORATORY_VERIFIER_V1":
        raise ValueError("Expected frozen exploratory, test-free error corpus")
    rows = read_csv(args.error_corpus / "predicted_component_errors.csv")
    case_ids = np.asarray([row["case_id"] for row in rows])
    unique_cases = sorted(set(case_ids))
    if len(unique_cases) != 5:
        raise ValueError(f"Expected five OOF validation cases, got {unique_cases}")
    x = np.asarray([features(row) for row in rows], dtype=np.float64)
    y_by_target = targets(rows)
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []

    for target_name, y in y_by_target.items():
        q0_risk = x[:, 0]
        q1_risk = np.zeros(len(rows), dtype=np.float64)
        for held_case in unique_cases:
            train = case_ids != held_case
            test = ~train
            if len(np.unique(y[train])) != 2:
                raise ValueError(f"Training target {target_name} is single-class without {held_case}")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=2026,
                ),
            )
            model.fit(x[train], y[train])
            q1_risk[test] = model.predict_proba(x[test])[:, 1]
        for method, risk in (("Q0_confidence", q0_risk), ("Q1_local_logistic", q1_risk)):
            for held_case in unique_cases:
                selected = case_ids == held_case
                metric_rows.append({
                    "target": target_name,
                    "method": method,
                    "scope": "per_case",
                    "case_id": held_case,
                    **binary_metrics(y[selected], risk[selected]),
                })
            per_case = [row for row in metric_rows if row["target"] == target_name and row["method"] == method and row["scope"] == "per_case"]
            pooled = binary_metrics(y, risk)
            metric_rows.append({
                "target": target_name,
                "method": method,
                "scope": "pooled_components",
                "case_id": "ALL",
                **pooled,
            })
            metric_rows.append({
                "target": target_name,
                "method": method,
                "scope": "macro_case_mean",
                "case_id": "ALL",
                "components": len(rows),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auprc": float(np.mean([row["auprc"] for row in per_case if row["auprc"] is not None])),
                "auroc": float(np.mean([row["auroc"] for row in per_case if row["auroc"] is not None])),
                "brier": float(np.mean([row["brier"] for row in per_case])),
            })
        for index, row in enumerate(rows):
            prediction_rows.append({
                "target": target_name,
                "fold": row["fold"],
                "case_id": row["case_id"],
                "frame": row["frame"],
                "component_id": row["component_id"],
                "true_label": int(y[index]),
                "error_label": row["error_label"],
                "q0_confidence_risk": q0_risk[index],
                "q1_local_logistic_risk": q1_risk[index],
            })

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metrics.csv", metric_rows)
    write_csv(args.output_root / "oof_risk_predictions.csv", prediction_rows)
    primary = [
        row for row in metric_rows
        if row["target"] == "dangerous_error" and row["scope"] == "macro_case_mean"
    ]
    output_manifest = {
        "artifact": "SAFE-Graph warning baselines Q0/Q1",
        "version": "warning_baselines_v1",
        "source_error_corpus": str(args.error_corpus),
        "source_split": manifest["source_split"],
        "test_data_used": False,
        "evaluation": "leave-one-case-out across five OOF validation cases",
        "methods": {
            "Q0_confidence": "1 - mean predicted-class probability; no fitting",
            "Q1_local_logistic": {
                "features": FEATURE_NAMES,
                "preprocessing": "fit StandardScaler on four training cases",
                "classifier": "balanced logistic regression fitted on four training cases",
            },
        },
        "primary_target": "dangerous_error_candidate from frozen exploratory taxonomy",
        "primary_macro_case_results": primary,
        "limitations": [
            "Only five validation-rotation cases are available",
            "Component labels are algorithmic and pending clinical taxonomy review",
            "Q1 probabilities use class-balanced training and are not post-hoc calibrated",
            "No station, relation, image-feature, ensemble, or temporal evidence is used",
        ],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
