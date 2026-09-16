#!/usr/bin/env python3
"""Pair-feasibility audit and pair-specific SAFE-Graph exploratory verifier."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


PAIR_METHODS = {
    "P1_local_plus_pair_class": [
        "local_confidence_risk", "local_top1_risk", "local_margin_risk",
        "local_entropy", "local_log_area", "pair_class_indicator",
    ],
    "P2_counterpart_probability": [
        "counterpart_probability", "fold_excluded_confusion_score",
    ],
    "P3_relation_only": [
        "relation_distance_violation", "relation_touch_mismatch",
    ],
    "PCTRL_local_class_plus_confusion": [
        "local_confidence_risk", "local_top1_risk", "local_margin_risk",
        "local_entropy", "local_log_area", "pair_class_indicator",
        "counterpart_probability", "fold_excluded_confusion_score",
    ],
    "P4_local_confusion_relation": [
        "local_confidence_risk", "local_top1_risk", "local_margin_risk",
        "local_entropy", "local_log_area", "pair_class_indicator",
        "counterpart_probability", "fold_excluded_confusion_score",
        "relation_distance_violation", "relation_touch_mismatch",
    ],
    "P5_gated_full": [
        "local_confidence_risk", "local_top1_risk", "local_margin_risk",
        "local_entropy", "local_log_area", "pair_class_indicator",
        "gated_counterpart_probability", "gated_confusion_score",
        "gated_relation_distance", "gated_relation_touch", "reliability_gate",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-corpus", type=Path,
        default=ROOT / "artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1",
    )
    parser.add_argument(
        "--q3-features", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/q3_ai_provisional_v1_r1/component_features.csv",
    )
    parser.add_argument(
        "--ai-review-root", type=Path,
        default=ROOT / (
            "artifacts/safe_graph_knowledge/kg_v1_20260808/"
            "clinical_pilot_v1_r2_ai_provisional"
        ),
    )
    parser.add_argument("--target-rule", default="CF-005")
    parser.add_argument("--min-positive-total", type=int, default=10)
    parser.add_argument("--min-positive-cases", type=int, default=3)
    parser.add_argument("--min-negative-cases", type=int, default=3)
    parser.add_argument("--min-nontiny-positive-fraction", type=float, default=0.25)
    parser.add_argument("--warning-coverage", type=float, default=0.10)
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


def pair_target(row: dict[str, str], class_i: int, class_j: int) -> bool:
    predicted = int(row["predicted_class_id"])
    dominant = int(row["dominant_gt_class_id"])
    return (
        row["error_label"] == "CLASS_SWAP"
        and predicted in {class_i, class_j}
        and dominant in {class_i, class_j}
        and predicted != dominant
    )


def size_bin(area_fraction: float) -> str:
    if area_fraction < 0.0005:
        return "tiny"
    if area_fraction < 0.002:
        return "small"
    if area_fraction < 0.01:
        return "medium"
    return "large"


def feasibility_for_pair(
    error_rows: list[dict[str, str]], pair: dict[str, str], relation_pairs: set[tuple[int, int]],
    min_positive_total: int, min_positive_cases: int, min_negative_cases: int,
    min_nontiny_positive_fraction: float,
) -> tuple[dict, list[dict]]:
    first, second = int(pair["class_i"]), int(pair["class_j"])
    cohort = [row for row in error_rows if int(row["predicted_class_id"]) in {first, second}]
    case_rows = []
    all_positive = []
    all_negative = []
    for case_id in sorted({row["case_id"] for row in error_rows}):
        selected = [row for row in cohort if row["case_id"] == case_id]
        positive = [row for row in selected if pair_target(row, first, second)]
        negative = [row for row in selected if not pair_target(row, first, second)]
        all_positive.extend(positive)
        all_negative.extend(negative)
        case_rows.append({
            "confusion_rule_id": pair["confusion_rule_id"],
            "case_id": case_id,
            "cohort_components": len(selected),
            "pair_swap_positives": len(positive),
            "negatives": len(negative),
            "positive_tiny": sum(
                float(row["area_fraction"]) < 0.0005 for row in positive
            ),
            "positive_non_tiny": sum(
                float(row["area_fraction"]) >= 0.0005 for row in positive
            ),
        })
    positive_cases = sum(row["pair_swap_positives"] > 0 for row in case_rows)
    negative_cases = sum(row["negatives"] > 0 for row in case_rows)
    non_tiny = sum(float(row["area_fraction"]) >= 0.0005 for row in all_positive)
    non_tiny_fraction = non_tiny / len(all_positive) if all_positive else 0.0
    pair_relation = tuple(sorted((first, second))) in relation_pairs
    eligibility_reasons = []
    if len(all_positive) < min_positive_total:
        eligibility_reasons.append("positive_total_below_threshold")
    if positive_cases < min_positive_cases:
        eligibility_reasons.append("positive_case_support_below_threshold")
    if negative_cases < min_negative_cases:
        eligibility_reasons.append("negative_case_support_below_threshold")
    if non_tiny_fraction < min_nontiny_positive_fraction:
        eligibility_reasons.append("positive_errors_tiny_dominated")
    summary = {
        "confusion_rule_id": pair["confusion_rule_id"],
        "class_i": first, "name_i": pair["name_i"],
        "class_j": second, "name_j": pair["name_j"],
        "ai_decision": pair["expert_decision_valid_invalid_depends"],
        "ai_warning_eligible": pair["expert_warning_eligible"],
        "cohort_components": len(cohort),
        "pair_swap_positives": len(all_positive),
        "negatives": len(all_negative),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "positive_non_tiny": non_tiny,
        "positive_non_tiny_fraction": non_tiny_fraction,
        "reviewed_relation_available": pair_relation,
        "eligible": not eligibility_reasons,
        "eligibility_reasons": "|".join(eligibility_reasons) if eligibility_reasons else "PASS",
    }
    return summary, case_rows


def counterpart_probability(row: dict[str, str], counterpart_id: int) -> float:
    ids = [int(item) for item in row["alternative_class_ids"].split("|") if item]
    probabilities = [
        float(item) for item in row["alternative_mean_probabilities"].split("|") if item
    ]
    if len(ids) != len(probabilities):
        raise ValueError("Alternative class/probability mismatch")
    return max((probability for class_id, probability in zip(ids, probabilities)
                if class_id == counterpart_id), default=0.0)


def build_pair_cohort(
    error_rows: list[dict[str, str]], q3_rows: list[dict[str, str]],
    pair: dict[str, str],
) -> list[dict]:
    q3_lookup = {
        (row["fold"], row["case_id"], row["frame"], row["component_id"]): row
        for row in q3_rows
    }
    first, second = int(pair["class_i"]), int(pair["class_j"])
    output = []
    for row in error_rows:
        predicted = int(row["predicted_class_id"])
        if predicted not in {first, second}:
            continue
        key = (row["fold"], row["case_id"], row["frame"], row["component_id"])
        q3 = q3_lookup[key]
        counterpart = second if predicted == first else first
        probability = counterpart_probability(row, counterpart)
        gate = float(q3["reliability_gate"])
        output.append({
            "fold": row["fold"], "case_id": row["case_id"], "frame": row["frame"],
            "component_id": row["component_id"],
            "predicted_class_id": predicted,
            "predicted_class_name": row["predicted_class_name"],
            "dominant_gt_class_id": row["dominant_gt_class_id"],
            "dominant_gt_class_name": row["dominant_gt_class_name"],
            "error_label": row["error_label"],
            "pair_swap_target": pair_target(row, first, second),
            "area_fraction": row["area_fraction"],
            "area_bin": size_bin(float(row["area_fraction"])),
            "mean_tool_probability": row["mean_tool_probability"],
            "local_confidence_risk": q3["local_confidence_risk"],
            "local_top1_risk": q3["local_top1_risk"],
            "local_margin_risk": q3["local_margin_risk"],
            "local_entropy": q3["local_entropy"],
            "local_log_area": q3["local_log_area"],
            "pair_class_indicator": float(predicted == second),
            "counterpart_probability": probability,
            "fold_excluded_confusion_score": q3["confusion_valid_score"],
            "relation_distance_violation": q3["relation_distance_violation"],
            "relation_touch_mismatch": q3["relation_touch_mismatch"],
            "relation_evidence_available": q3["relation_evidence_available"],
            "reliability_gate": gate,
            "gated_counterpart_probability": gate * probability,
            "gated_confusion_score": q3["gated_confusion_all_score"],
            "gated_relation_distance": q3["gated_relation_distance_violation"],
            "gated_relation_touch": q3["gated_relation_touch_mismatch"],
        })
    return output


def threshold_metrics(target: np.ndarray, risk: np.ndarray, threshold: float) -> dict:
    warning = risk >= threshold
    true_positive = int((warning & (target == 1)).sum())
    warnings = int(warning.sum())
    positives = int(target.sum())
    return {
        "threshold": float(threshold),
        "warning_count": warnings,
        "warning_coverage": float(warning.mean()),
        "warning_ppv": true_positive / warnings if warnings else None,
        "warning_recall": true_positive / positives if positives else None,
    }


def evaluate_pair(cohort: list[dict], warning_coverage: float):
    case_ids = np.asarray([row["case_id"] for row in cohort])
    unique_cases = sorted(set(case_ids))
    y = np.asarray([str(row["pair_swap_target"]).lower() == "true" for row in cohort], dtype=np.uint8)
    matrices = {
        method: np.asarray([[float(row[name]) for name in names] for row in cohort])
        for method, names in PAIR_METHODS.items()
    }
    risks = {"P0_confidence": np.asarray([float(row["local_confidence_risk"]) for row in cohort])}
    thresholds = {method: {} for method in ["P0_confidence", *PAIR_METHODS]}
    for method, x in matrices.items():
        risk = np.zeros(len(cohort), dtype=float)
        for held_case in unique_cases:
            train, test = case_ids != held_case, case_ids == held_case
            if len(np.unique(y[train])) != 2:
                raise ValueError(f"Training split without {held_case} is single-class")
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026),
            )
            model.fit(x[train], y[train])
            risk[test] = model.predict_proba(x[test])[:, 1]
            thresholds[method][held_case] = float(np.quantile(
                model.predict_proba(x[train])[:, 1], 1 - warning_coverage
            ))
        risks[method] = risk
    for held_case in unique_cases:
        train = case_ids != held_case
        thresholds["P0_confidence"][held_case] = float(
            np.quantile(risks["P0_confidence"][train], 1 - warning_coverage)
        )

    metric_rows, operational_rows, prediction_rows = [], [], []
    for method, risk in risks.items():
        case_metrics = []
        for held_case in unique_cases:
            selected = case_ids == held_case
            result = binary_metrics(y[selected], risk[selected])
            case_metrics.append(result)
            metric_rows.append({
                "method": method, "scope": "per_case", "case_id": held_case, **result,
            })
            operational_rows.append({
                "method": method, "case_id": held_case,
                **threshold_metrics(y[selected], risk[selected], thresholds[method][held_case]),
            })
        valid_auprc = [item["auprc"] for item in case_metrics if item["auprc"] is not None]
        valid_auroc = [item["auroc"] for item in case_metrics if item["auroc"] is not None]
        metric_rows.append({
            "method": method, "scope": "pooled_components", "case_id": "ALL",
            **binary_metrics(y, risk),
        })
        metric_rows.append({
            "method": method, "scope": "macro_case_mean", "case_id": "ALL",
            "components": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
            "auprc": float(np.mean(valid_auprc)) if valid_auprc else None,
            "auroc": float(np.mean(valid_auroc)) if valid_auroc else None,
            "brier": float(np.mean([item["brier"] for item in case_metrics])),
        })
        for index, row in enumerate(cohort):
            prediction_rows.append({
                "method": method, "fold": row["fold"], "case_id": row["case_id"],
                "frame": row["frame"], "component_id": row["component_id"],
                "true_label": int(y[index]), "risk": risk[index],
            })
    return metric_rows, operational_rows, prediction_rows, risks, y


def stratified_metrics(cohort: list[dict], risks: dict[str, np.ndarray], y: np.ndarray) -> list[dict]:
    strata = {
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in cohort]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in cohort]),
        "tool_low": np.asarray([float(row["mean_tool_probability"]) < 0.10 for row in cohort]),
        "tool_high": np.asarray([float(row["mean_tool_probability"]) >= 0.10 for row in cohort]),
        "reliability_low": np.asarray([float(row["reliability_gate"]) < 0.50 for row in cohort]),
        "reliability_high": np.asarray([float(row["reliability_gate"]) >= 0.50 for row in cohort]),
    }
    output = []
    for stratum, selected in strata.items():
        if not selected.any():
            continue
        for method, risk in risks.items():
            output.append({
                "stratum": stratum, "method": method,
                **binary_metrics(y[selected], risk[selected]),
            })
    return output


def operational_summary(operational_rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in operational_rows}):
        rows = [row for row in operational_rows if row["method"] == method]
        ppv = [float(row["warning_ppv"]) for row in rows if row["warning_ppv"] is not None]
        recall = [float(row["warning_recall"]) for row in rows if row["warning_recall"] is not None]
        output.append({
            "method": method,
            "cases": len(rows),
            "mean_warning_coverage": float(np.mean([row["warning_coverage"] for row in rows])),
            "mean_warning_ppv": float(np.mean(ppv)) if ppv else None,
            "mean_warning_recall": float(np.mean(recall)) if recall else None,
            "cases_ppv_at_least_0_5": sum(
                row["warning_ppv"] is not None and float(row["warning_ppv"]) >= 0.5
                for row in rows
            ),
            "cases_zero_true_positive_warning": sum(
                row["warning_count"] and row["warning_ppv"] is not None
                and float(row["warning_ppv"]) == 0.0 for row in rows
            ),
        })
    return output


def decision(metric_rows: list[dict], operational_rows: list[dict]) -> dict:
    macro = {
        row["method"]: row for row in metric_rows if row["scope"] == "macro_case_mean"
    }
    p1 = float(macro["P1_local_plus_pair_class"]["auprc"])
    p2 = float(macro["P2_counterpart_probability"]["auprc"])
    pctrl = float(macro["PCTRL_local_class_plus_confusion"]["auprc"])
    p5 = float(macro["P5_gated_full"]["auprc"])
    per_case = defaultdict(dict)
    for row in metric_rows:
        if row["scope"] == "per_case" and row["auprc"] is not None:
            per_case[row["case_id"]][row["method"]] = float(row["auprc"])
    comparable = [values for values in per_case.values()
                  if "P1_local_plus_pair_class" in values and "P5_gated_full" in values]
    improved = sum(values["P5_gated_full"] > values["P1_local_plus_pair_class"] for values in comparable)
    confusion_improved = sum(
        values.get("P2_counterpart_probability", -1) > values["P1_local_plus_pair_class"]
        for values in comparable
    )
    relation_improved = sum(
        values["P5_gated_full"] > values.get("PCTRL_local_class_plus_confusion", 1)
        for values in comparable
    )
    delta = p5 - p1
    pair_signal_passed = delta >= 0.02 and improved >= 3
    confusion_signal_passed = p2 - p1 >= 0.02 and confusion_improved >= 3
    relation_delta = p5 - pctrl
    relation_signal_passed = relation_delta >= 0.02 and relation_improved >= 3
    op_summary = {row["method"]: row for row in operational_summary(operational_rows)}
    p5_operational = op_summary["P5_gated_full"]
    status = (
        "EXPLORATORY_CONFUSION_SIGNAL_RELATION_NO_GO"
        if confusion_signal_passed and not relation_signal_passed
        else "NO_GO_PAIR_SPECIFIC_CURRENT_FEATURES"
    )
    return {
        "primary_comparison": "P5_gated_full versus P1_local_plus_pair_class",
        "p1_macro_case_auprc": p1,
        "p5_macro_case_auprc": p5,
        "absolute_delta_auprc": delta,
        "evaluable_cases": len(comparable),
        "cases_improved": improved,
        "pair_signal_gate_pass": pair_signal_passed,
        "p2_counterpart_probability_auprc": p2,
        "confusion_delta_over_p1": p2 - p1,
        "confusion_cases_improved": confusion_improved,
        "confusion_signal_gate_pass": confusion_signal_passed,
        "local_class_confusion_control_auprc": pctrl,
        "relation_increment_over_local_class_confusion": relation_delta,
        "relation_cases_improved": relation_improved,
        "relation_signal_gate_pass": relation_signal_passed,
        "p5_operational_at_training_10pct_coverage": p5_operational,
        "clinical_warning_ready": False,
        "status": status,
        "clinical_authority": False,
        "confirmatory_authority": False,
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    error_rows = read_csv(args.error_corpus / "predicted_component_errors.csv")
    q3_rows = read_csv(args.q3_features)
    confusion_rows = read_csv(args.ai_review_root / "priority_confusion_review.csv")
    relation_pairs = {
        tuple(sorted((int(row["subject_id"]), int(row["object_id"]))))
        for row in read_csv(args.ai_review_root / "priority_anatomy_relation_review.csv")
    }
    feasibility_rows, case_rows = [], []
    for pair in confusion_rows:
        summary, per_case = feasibility_for_pair(
            error_rows, pair, relation_pairs,
            args.min_positive_total, args.min_positive_cases, args.min_negative_cases,
            args.min_nontiny_positive_fraction,
        )
        feasibility_rows.append(summary)
        case_rows.extend(per_case)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "pair_feasibility.csv", feasibility_rows)
    write_csv(args.output_root / "pair_feasibility_by_case.csv", case_rows)
    pair = next((row for row in confusion_rows if row["confusion_rule_id"] == args.target_rule), None)
    if pair is None:
        raise ValueError(f"Unknown target rule {args.target_rule}")
    feasibility = next(row for row in feasibility_rows if row["confusion_rule_id"] == args.target_rule)
    if not feasibility["eligible"]:
        manifest = {
            "artifact": "SAFE-Graph pair-specific feasibility and verifier",
            "version": args.output_root.name,
            "status": "STOPPED_AT_FEASIBILITY_GATE",
            "target_rule": args.target_rule,
            "target_feasibility": feasibility,
            "test_data_used": False,
            "clinical_authority": False,
        }
        (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    cohort = build_pair_cohort(error_rows, q3_rows, pair)
    metric_rows, operational_rows, predictions, risks, y = evaluate_pair(
        cohort, args.warning_coverage
    )
    strata = stratified_metrics(cohort, risks, y)
    op_summary = operational_summary(operational_rows)
    result_decision = decision(metric_rows, operational_rows)
    write_csv(args.output_root / "pair_cohort.csv", cohort)
    write_csv(args.output_root / "metrics.csv", metric_rows)
    write_csv(args.output_root / "operational_metrics.csv", operational_rows)
    write_csv(args.output_root / "operational_summary.csv", op_summary)
    write_csv(args.output_root / "stratified_metrics.csv", strata)
    write_csv(args.output_root / "oof_risk_predictions.csv", predictions)
    (args.output_root / "decision.json").write_text(json.dumps(result_decision, indent=2) + "\n")
    primary = [row for row in metric_rows if row["scope"] == "macro_case_mean"]
    manifest = {
        "artifact": "SAFE-Graph pair-specific feasibility and exploratory verifier",
        "version": args.output_root.name,
        "status": result_decision["status"],
        "target_rule": args.target_rule,
        "pair": {
            "class_i": int(pair["class_i"]), "name_i": pair["name_i"],
            "class_j": int(pair["class_j"]), "name_j": pair["name_j"],
        },
        "endpoint": "Within-pair predicted-component swap; no clinical weight in target",
        "target_feasibility": feasibility,
        "eligibility_thresholds": {
            "min_positive_total": args.min_positive_total,
            "min_positive_cases": args.min_positive_cases,
            "min_negative_cases": args.min_negative_cases,
            "min_nontiny_positive_fraction": args.min_nontiny_positive_fraction,
        },
        "evaluation": "leave-one-case-out across five OOF validation cases",
        "warning_threshold": f"training-fold quantile targeting {args.warning_coverage:.0%} coverage",
        "test_data_used": False,
        "ai_provisional_selection": True,
        "selection_leakage": "Target pair came from all-fold OOF aggregate; exploratory only",
        "clinical_expert_complete": False,
        "freeze_authorized": False,
        "correction_authorized": False,
        "methods": {"P0_confidence": ["local_confidence_risk"], **PAIR_METHODS},
        "primary_macro_case_results": primary,
        "decision": result_decision,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "feasibility": feasibility, "primary": primary, "decision": result_decision,
    }, indent=2))


if __name__ == "__main__":
    main()
