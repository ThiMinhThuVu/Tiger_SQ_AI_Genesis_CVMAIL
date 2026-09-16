#!/usr/bin/env python3
"""Evaluate AI-provisional SAFE-Graph Q3 relation/confusion warning features.

This is explicitly exploratory: reviewed pairs were selected from the complete
OOF validation aggregate and adjudicated by an AI agent, not a clinician.
Fold-specific feature statistics still exclude the held validation fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.train_safe_graph_warning_baselines import (  # noqa: E402
    binary_metrics,
    features as local_features,
    parse_bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-corpus", type=Path,
        default=ROOT / "artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1",
    )
    parser.add_argument(
        "--oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_v1",
    )
    parser.add_argument(
        "--kg-root", type=Path,
        default=ROOT / "artifacts/safe_graph_knowledge/kg_v1_20260808",
    )
    parser.add_argument(
        "--ai-review-root", type=Path,
        default=ROOT / (
            "artifacts/safe_graph_knowledge/kg_v1_20260808/"
            "clinical_pilot_v1_r2_ai_provisional"
        ),
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / (
            "artifacts/mask2former_improved/"
            "mask2former_swin_small_clinical_hier_presence_v2"
        ),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(first: int, second: int) -> tuple[int, int]:
    return tuple(sorted((int(first), int(second))))


def parse_pipe_numbers(value: str, cast) -> list:
    return [cast(item) for item in value.split("|") if item != ""]


def load_reviewed_pairs(review_root: Path) -> tuple[dict[tuple[int, int], dict], set[tuple[int, int]]]:
    validation = json.loads((review_root / "review_validation.json").read_text())
    if validation.get("status") != "AI_PROVISIONAL_COMPLETE":
        raise ValueError("Q3 provisional run requires a complete AI-provisional review")
    if validation.get("clinical_expert_complete") is not False or validation.get("freeze_authorized") is not False:
        raise ValueError("AI review safety flags are invalid")
    confusion = {}
    warning_yes = set()
    for row in read_csv(review_root / "priority_confusion_review.csv"):
        key = pair_key(row["class_i"], row["class_j"])
        confusion[key] = row
        if row["expert_warning_eligible"].strip().upper() == "YES":
            warning_yes.add(key)
    relations = {
        pair_key(row["subject_id"], row["object_id"])
        for row in read_csv(review_root / "priority_anatomy_relation_review.csv")
        if row["expert_decision_valid_invalid_depends"].strip().upper() in {"VALID", "DEPENDS"}
        and row["expert_constraint_hard_soft"].strip().upper() == "SOFT"
        and row["expert_correction_eligible"].strip().upper() == "NO"
    }
    return confusion, warning_yes, relations


def fold_excluded_confusion_weights(model_dir: Path) -> dict[int, dict[tuple[int, int], float]]:
    matrices = []
    for fold in range(5):
        metrics = json.loads((model_dir / f"fold_{fold}/best_validation_metrics.json").read_text())
        matrix = np.asarray(metrics["failure_metrics"]["confusion_pixels"], dtype=np.int64)
        if matrix.shape != (31, 31):
            raise ValueError(f"Fold {fold} confusion matrix is {matrix.shape}, expected (31, 31)")
        matrices.append(matrix)
    output = {}
    for held_fold in range(5):
        aggregate = sum(matrix for index, matrix in enumerate(matrices) if index != held_fold)
        support = aggregate.sum(axis=1)
        directed = aggregate / np.maximum(support[:, None], 1)
        np.fill_diagonal(directed, 0.0)
        output[held_fold] = {
            (first, second): float(max(directed[first, second], directed[second, first]))
            for first in range(1, 31) for second in range(first + 1, 31)
        }
    return output


def load_relation_envelopes(
    kg_root: Path, relation_pairs: set[tuple[int, int]],
) -> dict[int, dict[tuple[int, int], dict[str, float]]]:
    output: dict[int, dict[tuple[int, int], dict[str, float]]] = {}
    for fold in range(5):
        grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        rows = read_csv(kg_root / f"outer_fold_{fold}_train/anatomy_pair_relations.csv")
        for row in rows:
            key = pair_key(row["anatomy_i_id"], row["anatomy_j_id"])
            if key in relation_pairs and int(row["relation_observations"]) > 0:
                grouped[key].append(row)
        fold_envelopes = {}
        for key, values in grouped.items():
            observations = np.asarray([int(row["relation_observations"]) for row in values], dtype=float)
            fold_envelopes[key] = {
                "distance_q05": min(float(row["boundary_distance_norm_q05"]) for row in values),
                "distance_q95": max(float(row["boundary_distance_norm_q95"]) for row in values),
                "touch_probability": float(np.average(
                    [float(row["touch_probability_when_covisible"]) for row in values],
                    weights=observations,
                )),
                "observations": float(observations.sum()),
                "station_rows": float(len(values)),
            }
        output[fold] = fold_envelopes
    return output


def recover_global_components(prediction: np.ndarray) -> dict[int, np.ndarray]:
    output = {}
    component_id = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in range(1, 31):
        labels, count = ndimage.label(prediction == class_id, structure=structure)
        for local_id in range(1, count + 1):
            component_id += 1
            output[component_id] = labels == local_id
    return output


def relation_features_for_frame(
    prediction: np.ndarray,
    frame_rows: list[dict[str, str]],
    envelopes: dict[tuple[int, int], dict[str, float]],
    relation_pairs: set[tuple[int, int]],
) -> dict[int, dict[str, float]]:
    components = recover_global_components(prediction)
    diagonal = math.hypot(*prediction.shape)
    partner_classes: dict[int, set[int]] = defaultdict(set)
    for first, second in relation_pairs:
        partner_classes[first].add(second)
        partner_classes[second].add(first)
    distance_maps = {
        class_id: ndimage.distance_transform_edt(prediction != class_id)
        for class_id in {item for pair in relation_pairs for item in pair}
        if np.any(prediction == class_id)
    }
    output = {}
    for row in frame_rows:
        component_id = int(row["component_id"])
        class_id = int(row["predicted_class_id"])
        focus = components[component_id]
        distance_violations = []
        touch_mismatches = []
        supports = []
        for partner in partner_classes.get(class_id, set()):
            key = pair_key(class_id, partner)
            if key not in envelopes or partner not in distance_maps:
                continue  # Missing partner/evidence is UNKNOWN, never a violation.
            observed_distance = float(distance_maps[partner][focus].min() / diagonal)
            expected = envelopes[key]
            low, high = expected["distance_q05"], expected["distance_q95"]
            width = max(high - low, 1.0 / diagonal)
            violation = max(low - observed_distance, observed_distance - high, 0.0) / width
            violation = min(violation, 10.0)
            observed_touch = observed_distance <= math.sqrt(2.0) / diagonal
            touch_mismatch = expected["touch_probability"] * (0.0 if observed_touch else 1.0)
            distance_violations.append(violation)
            touch_mismatches.append(touch_mismatch)
            supports.append(math.log1p(expected["observations"]))
        output[component_id] = {
            "relation_distance_violation": max(distance_violations, default=0.0),
            "relation_touch_mismatch": max(touch_mismatches, default=0.0),
            "relation_support_log": max(supports, default=0.0),
            "relation_evidence_available": float(bool(distance_violations)),
        }
    return output


def confusion_features(
    row: dict[str, str], reviewed: dict[tuple[int, int], dict], warning_yes: set[tuple[int, int]],
    weights: dict[tuple[int, int], float],
) -> dict[str, float]:
    predicted = int(row["predicted_class_id"])
    alternatives = parse_pipe_numbers(row["alternative_class_ids"], int)
    probabilities = parse_pipe_numbers(row["alternative_mean_probabilities"], float)
    if len(alternatives) != len(probabilities):
        raise ValueError(f"Alternative class/probability mismatch for {row['frame']}")
    all_scores, valid_scores = [], []
    all_probabilities, valid_probabilities = [], []
    for alternative, probability in zip(alternatives, probabilities):
        if alternative == predicted:
            continue
        key = pair_key(predicted, alternative)
        if key not in reviewed:
            continue
        score = probability * weights.get(key, 0.0)
        all_scores.append(score)
        all_probabilities.append(probability)
        if key in warning_yes:
            valid_scores.append(score)
            valid_probabilities.append(probability)
    return {
        "confusion_all_score": max(all_scores, default=0.0),
        "confusion_all_alt_probability": max(all_probabilities, default=0.0),
        "confusion_valid_score": max(valid_scores, default=0.0),
        "confusion_valid_alt_probability": max(valid_probabilities, default=0.0),
    }


def reliability(row: dict[str, str]) -> float:
    entropy_reliability = 1.0 - min(float(row["mean_entropy"]) / math.log(31), 1.0)
    tool_reliability = 1.0 - min(float(row["mean_tool_probability"]), 1.0)
    return max(entropy_reliability * tool_reliability, 0.0)


def build_feature_rows(
    rows: list[dict[str, str]], oof_root: Path,
    reviewed_confusions: dict[tuple[int, int], dict], warning_yes: set[tuple[int, int]],
    relation_pairs: set[tuple[int, int]], confusion_weights, relation_envelopes,
) -> list[dict]:
    by_fold_frame: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_fold_frame[(int(row["fold"]), row["frame"])].append(row)
    array_paths = {}
    for fold in range(5):
        fold_dir = oof_root / f"fold_{fold}"
        for row in read_csv(fold_dir / "frame_index.csv"):
            array_paths[(fold, row["frame"])] = fold_dir / row["array_file"]
    feature_rows = []
    for (fold, frame), frame_rows in sorted(by_fold_frame.items()):
        arrays = np.load(array_paths[(fold, frame)])
        relation_by_component = relation_features_for_frame(
            arrays["prediction"], frame_rows, relation_envelopes[fold], relation_pairs
        )
        for row in frame_rows:
            relation = relation_by_component[int(row["component_id"])]
            confusion = confusion_features(
                row, reviewed_confusions, warning_yes, confusion_weights[fold]
            )
            gate = reliability(row)
            reason_codes = []
            if confusion["confusion_valid_score"] > 0:
                reason_codes.append("AI_PROVISIONAL_CONFUSION_PAIR")
            if relation["relation_distance_violation"] > 0:
                reason_codes.append("AI_PROVISIONAL_RELATION_DISTANCE")
            if relation["relation_touch_mismatch"] > 0:
                reason_codes.append("AI_PROVISIONAL_EXPECTED_TOUCH")
            if gate < 0.5:
                reason_codes.append("ABSTAIN_LOW_RELIABILITY")
            local = local_features(row)
            feature_rows.append({
                "fold": fold, "case_id": row["case_id"], "frame": frame,
                "component_id": row["component_id"],
                "predicted_class_id": row["predicted_class_id"],
                "error_label": row["error_label"],
                "dangerous_error_candidate": row["dangerous_error_candidate"],
                "any_component_error": row["error_label"] != "CORRECT_COMPONENT",
                "local_confidence_risk": local[0],
                "local_top1_risk": local[1],
                "local_margin_risk": local[2],
                "local_entropy": local[3],
                "local_log_area": local[4],
                **relation, **confusion,
                "reliability_gate": gate,
                "gated_relation_distance_violation": gate * relation["relation_distance_violation"],
                "gated_relation_touch_mismatch": gate * relation["relation_touch_mismatch"],
                "gated_confusion_all_score": gate * confusion["confusion_all_score"],
                "gated_confusion_all_alt_probability": gate * confusion["confusion_all_alt_probability"],
                "reason_codes": "|".join(reason_codes) if reason_codes else "NONE",
            })
    return feature_rows


METHOD_FEATURES = {
    "Q3a_relation_only": ["relation_distance_violation", "relation_touch_mismatch"],
    "Q3b_confusion_valid_only": ["confusion_valid_score", "confusion_valid_alt_probability"],
    "Q3b_all_provisional_confusions": ["confusion_all_score", "confusion_all_alt_probability"],
    "Q3c_relation_plus_confusion": [
        "relation_distance_violation", "relation_touch_mismatch",
        "confusion_all_score", "confusion_all_alt_probability",
    ],
    "Q3d_gated_knowledge": [
        "gated_relation_distance_violation", "gated_relation_touch_mismatch",
        "gated_confusion_all_score", "gated_confusion_all_alt_probability",
    ],
    "Q3e_local_plus_gated_knowledge": [
        "local_confidence_risk", "local_top1_risk", "local_margin_risk",
        "local_entropy", "local_log_area",
        "gated_relation_distance_violation", "gated_relation_touch_mismatch",
        "gated_confusion_all_score", "gated_confusion_all_alt_probability",
    ],
}


def evaluate(feature_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    case_ids = np.asarray([row["case_id"] for row in feature_rows])
    unique_cases = sorted(set(case_ids))
    targets = {
        "dangerous_error": np.asarray([
            parse_bool(row["dangerous_error_candidate"]) for row in feature_rows
        ], dtype=np.uint8),
        "any_component_error": np.asarray([
            parse_bool(row["any_component_error"]) for row in feature_rows
        ], dtype=np.uint8),
    }
    metric_rows, prediction_rows = [], []
    for target_name, y in targets.items():
        risks = {"Q0_confidence": np.asarray([
            float(row["local_confidence_risk"]) for row in feature_rows
        ])}
        matrices = {
            method: np.asarray([[float(row[name]) for name in names] for row in feature_rows])
            for method, names in METHOD_FEATURES.items()
        }
        class_ids = np.asarray([int(row["predicted_class_id"]) for row in feature_rows])
        class_identity = np.eye(30, dtype=float)[class_ids - 1]
        local_matrix = matrices["Q3e_local_plus_gated_knowledge"][:, :5]
        gated_knowledge = matrices["Q3d_gated_knowledge"]
        matrices["QCTRL_class_identity"] = class_identity
        matrices["QCTRL_local_plus_class_identity"] = np.column_stack([
            local_matrix, class_identity,
        ])
        matrices["Q3f_local_class_plus_gated_knowledge"] = np.column_stack([
            local_matrix, class_identity, gated_knowledge,
        ])
        for method, x in matrices.items():
            risk = np.zeros(len(feature_rows), dtype=float)
            for held_case in unique_cases:
                train, test = case_ids != held_case, case_ids == held_case
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026),
                )
                model.fit(x[train], y[train])
                risk[test] = model.predict_proba(x[test])[:, 1]
            risks[method] = risk
        for method, risk in risks.items():
            case_results = []
            for held_case in unique_cases:
                selected = case_ids == held_case
                result = binary_metrics(y[selected], risk[selected])
                case_results.append(result)
                metric_rows.append({
                    "target": target_name, "method": method, "scope": "per_case",
                    "case_id": held_case, **result,
                })
            metric_rows.append({
                "target": target_name, "method": method, "scope": "pooled_components",
                "case_id": "ALL", **binary_metrics(y, risk),
            })
            metric_rows.append({
                "target": target_name, "method": method, "scope": "macro_case_mean",
                "case_id": "ALL", "components": len(y), "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auprc": float(np.mean([item["auprc"] for item in case_results])),
                "auroc": float(np.mean([item["auroc"] for item in case_results])),
                "brier": float(np.mean([item["brier"] for item in case_results])),
            })
        for index, row in enumerate(feature_rows):
            prediction_rows.append({
                "target": target_name, "fold": row["fold"], "case_id": row["case_id"],
                "frame": row["frame"], "component_id": row["component_id"],
                "true_label": int(y[index]), "error_label": row["error_label"],
                "reason_codes": row["reason_codes"],
                **{f"{method}_risk": risk[index] for method, risk in risks.items()},
            })
    return metric_rows, prediction_rows


def decision(metric_rows: list[dict]) -> dict:
    primary = {
        row["method"]: row for row in metric_rows
        if row["target"] == "dangerous_error" and row["scope"] == "macro_case_mean"
    }
    q0 = float(primary["Q0_confidence"]["auprc"])
    baseline = float(primary["QCTRL_local_plus_class_identity"]["auprc"])
    candidate = primary["Q3f_local_class_plus_gated_knowledge"]
    improved_cases = 0
    for case_id in sorted({
        row["case_id"] for row in metric_rows
        if row["target"] == "dangerous_error" and row["scope"] == "per_case"
    }):
        values = {
            row["method"]: float(row["auprc"])
            for row in metric_rows
            if row["target"] == "dangerous_error" and row["scope"] == "per_case"
            and row["case_id"] == case_id
        }
        improved_cases += (
            values["Q3f_local_class_plus_gated_knowledge"]
            > values["QCTRL_local_plus_class_identity"]
        )
    delta = float(candidate["auprc"]) - baseline
    internal_gate = delta >= 0.02 and improved_cases >= 3
    return {
        "comparison": "Q3f local+class+knowledge versus QCTRL local+class identity",
        "q0_macro_case_auprc": q0,
        "q3e_local_plus_knowledge_macro_case_auprc": float(
            primary["Q3e_local_plus_gated_knowledge"]["auprc"]
        ),
        "class_control_macro_case_auprc": baseline,
        "q3f_class_control_plus_knowledge_macro_case_auprc": float(candidate["auprc"]),
        "knowledge_absolute_delta_over_class_control": delta,
        "cases_improved_over_class_control": improved_cases,
        "internal_exploratory_gate_pass": internal_gate,
        "status": "EXPLORATORY_SIGNAL_ONLY" if internal_gate else "NO_GO_CURRENT_Q3_FEATURES",
        "confirmatory_authority": False,
        "clinical_authority": False,
        "reason": (
            "Pair selection used all-fold OOF aggregate and AI-provisional review; "
            "the target is clinical-weight-derived, so class identity is a mandatory control; "
            "clinical review and a newly frozen confirmatory design are required."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    error_manifest = json.loads((args.error_corpus / "manifest.json").read_text())
    if error_manifest.get("test_data_used") is not False:
        raise ValueError("Q3 requires a test-free error corpus")
    reviewed_confusions, warning_yes, relation_pairs = load_reviewed_pairs(args.ai_review_root)
    confusion_weights = fold_excluded_confusion_weights(args.model_dir)
    relation_envelopes = load_relation_envelopes(args.kg_root, relation_pairs)
    rows = read_csv(args.error_corpus / "predicted_component_errors.csv")
    feature_rows = build_feature_rows(
        rows, args.oof_root, reviewed_confusions, warning_yes, relation_pairs,
        confusion_weights, relation_envelopes,
    )
    metric_rows, prediction_rows = evaluate(feature_rows)
    result_decision = decision(metric_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "component_features.csv", feature_rows)
    write_csv(args.output_root / "metrics.csv", metric_rows)
    write_csv(args.output_root / "oof_risk_predictions.csv", prediction_rows)
    (args.output_root / "decision.json").write_text(json.dumps(result_decision, indent=2) + "\n")
    primary = [
        row for row in metric_rows
        if row["target"] == "dangerous_error" and row["scope"] == "macro_case_mean"
    ]
    sources = [
        args.error_corpus / "predicted_component_errors.csv",
        args.ai_review_root / "priority_confusion_review.csv",
        args.ai_review_root / "priority_anatomy_relation_review.csv",
        args.ai_review_root / "review_validation.json",
    ]
    manifest = {
        "artifact": "SAFE-Graph Q3 AI-provisional exploratory warning ablations",
        "version": args.output_root.name,
        "status": result_decision["status"],
        "source_split": "out_of_fold_validation_only",
        "test_data_used": False,
        "clinical_expert_complete": False,
        "freeze_authorized": False,
        "correction_authorized": False,
        "selection_leakage": (
            "Pair shortlist was selected from the aggregate of all five OOF validation folds. "
            "Fold-specific confusion weights and relation envelopes exclude the held fold, but pair selection does not."
        ),
        "evaluation": "leave-one-case-out across five OOF validation cases",
        "target": "algorithmic dangerous_error_candidate; not AI review labels",
        "methods": {
            "Q0_confidence": ["local_confidence_risk"],
            **METHOD_FEATURES,
            "QCTRL_class_identity": ["30-way predicted-class one-hot"],
            "QCTRL_local_plus_class_identity": ["five local features", "30-way predicted-class one-hot"],
            "Q3f_local_class_plus_gated_knowledge": [
                "five local features", "30-way predicted-class one-hot", "four gated knowledge features"
            ],
        },
        "reviewed_pair_counts": {
            "confusion_all": len(reviewed_confusions),
            "confusion_warning_yes": len(warning_yes),
            "relation_pairs": len(relation_pairs),
        },
        "primary_macro_case_results": primary,
        "decision": result_decision,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "limitations": [
            "AI provisional review is not clinical ground truth",
            "Only five OOF validation cases are available",
            "Pair selection is exploratory and not held-fold independent",
            "Dangerous-error target is clinical-weight-derived; class-identity controls are mandatory",
            "Station prediction is unavailable and no oracle station feature is used",
            "Missing anatomy/relation evidence is treated as UNKNOWN, never as a violation",
            "No mask correction or held-out test evaluation is authorized",
        ],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"primary": primary, "decision": result_decision}, indent=2))


if __name__ == "__main__":
    main()
