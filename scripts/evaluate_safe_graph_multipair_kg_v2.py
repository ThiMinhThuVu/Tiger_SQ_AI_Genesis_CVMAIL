#!/usr/bin/env python3
"""Multi-pair contrastive SAFE-Graph verifier over high-risk OOF components."""
from __future__ import annotations

import argparse
import csv
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
from scripts.analyze_safe_graph_p2_cf005 import (  # noqa: E402
    bootstrap_deltas, macro_metrics, parse_bool, summarize_operational,
)
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.evaluate_safe_graph_q3_ai_provisional import recover_global_components  # noqa: E402
from scripts.evaluate_safe_graph_rllr_cf005 import station_mixture_log_compatibility  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


METHODS = {
    "M0_ALT_PROBABILITY_RAW": "raw",
    "M1_PROBABILITY_PAIR_ID": "learned",
    "M2_KG_PAIR_ID": "learned",
    "M3_PROBABILITY_PAIR_ID_KG": "learned",
    "M4_PROBABILITY_PAIR_ID_SHUFFLED_KG": "learned",
    "M5_PROBABILITY_PAIR_ID_PRESENCE": "learned",
    "M6_CONFIDENCE_PROBABILITY_PAIR_ID": "learned",
}


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--high-risk-weight", type=float, default=3.0)
    parser.add_argument("--minimum-observations", type=int, default=20)
    parser.add_argument("--minimum-station-rows", type=int, default=3)
    parser.add_argument("--shrinkage-observations", type=float, default=20.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
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


def parse_pipe(value: str, cast) -> list:
    return [cast(item) for item in value.split("|") if item]


def component_key(row: dict) -> tuple[int, str, str, int]:
    return int(row["fold"]), row["case_id"], row["frame"], int(row["component_id"])


def load_all_profiles(kg_root: Path):
    profiles = {}
    audit = []
    for fold in range(5):
        path = kg_root / f"outer_fold_{fold}_train/anatomy_pair_relations.csv"
        grouped = {hypothesis: defaultdict(list) for hypothesis in range(1, 31)}
        for row in read_csv(path):
            first, second = int(row["anatomy_i_id"]), int(row["anatomy_j_id"])
            if int(row["relation_observations"]) <= 0:
                continue
            grouped[first][second].append(row)
            grouped[second][first].append(row)
        profiles[fold] = {hypothesis: dict(partners) for hypothesis, partners in grouped.items()}
        for hypothesis, partners in profiles[fold].items():
            for partner, rows in partners.items():
                audit.append({
                    "fold": fold, "hypothesis_class_id": hypothesis,
                    "partner_class_id": partner,
                    "observations": sum(int(row["relation_observations"]) for row in rows),
                    "station_rows": len(rows), "source": str(path),
                })
    return profiles, audit


def relation_support(rows: list[dict[str, str]]) -> tuple[int, int]:
    return sum(int(row["relation_observations"]) for row in rows), len(rows)


def eligible_partners(
    profile: dict[int, dict[int, list[dict[str, str]]]], predicted: int, alternative: int,
    minimum_observations: int, minimum_station_rows: int,
) -> list[int]:
    common = set(profile[predicted]) & set(profile[alternative])
    output = []
    for partner in sorted(common - {predicted, alternative}):
        supports = [relation_support(profile[hypothesis][partner]) for hypothesis in [predicted, alternative]]
        if all(obs >= minimum_observations and stations >= minimum_station_rows for obs, stations in supports):
            output.append(partner)
    return output


def deterministic_partner_map(partners: list[int], fold: int, predicted: int, alternative: int, seed: int):
    rng = np.random.default_rng(seed + fold * 100_000 + predicted * 100 + alternative)
    shuffled = np.asarray(partners, dtype=int).copy()
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and np.all(shuffled == np.asarray(partners)):
        shuffled = np.roll(shuffled, 1)
    return dict(zip(partners, shuffled.tolist()))


def hypothesis_relation_score(
    prediction: np.ndarray, focus: np.ndarray, predicted: int, alternative: int,
    profile: dict[int, dict[int, list[dict[str, str]]]], distance_maps: dict[int, np.ndarray],
    partners: list[int], shrinkage: float, partner_map: dict[int, int] | None = None,
) -> dict[str, float]:
    diagonal = math.hypot(*prediction.shape)
    predicted_logs, alternative_logs, weights = [], [], []
    for observed_partner in partners:
        if observed_partner not in distance_maps:
            continue  # Missing predicted anatomy is UNKNOWN, not a violation.
        profile_partner = partner_map.get(observed_partner, observed_partner) if partner_map else observed_partner
        distance = float(distance_maps[observed_partner][focus].min() / diagonal)
        touch = float(distance <= math.sqrt(2.0) / diagonal)
        predicted_rows = profile[predicted][profile_partner]
        alternative_rows = profile[alternative][profile_partner]
        predicted_logs.append(station_mixture_log_compatibility(distance, touch, predicted_rows, diagonal))
        alternative_logs.append(station_mixture_log_compatibility(distance, touch, alternative_rows, diagonal))
        support = min(relation_support(predicted_rows)[0], relation_support(alternative_rows)[0])
        weights.append(support / (support + shrinkage))
    if not weights:
        return {
            "kg_predicted_log_compatibility": 0.0,
            "kg_alternative_log_compatibility": 0.0,
            "kg_llr": 0.0, "kg_visible_partners": 0.0, "kg_evidence_available": 0.0,
        }
    predicted_log = float(np.average(predicted_logs, weights=weights))
    alternative_log = float(np.average(alternative_logs, weights=weights))
    return {
        "kg_predicted_log_compatibility": predicted_log,
        "kg_alternative_log_compatibility": alternative_log,
        "kg_llr": alternative_log - predicted_log,
        "kg_visible_partners": float(len(weights)), "kg_evidence_available": 1.0,
    }


def candidate_rows(component_rows: list[dict[str, str]], high_risk_weight: float) -> list[dict]:
    output = []
    for row in component_rows:
        if float(row["clinical_risk_weight"]) < high_risk_weight:
            continue
        predicted = int(row["predicted_class_id"])
        alternatives = parse_pipe(row["alternative_class_ids"], int)
        probabilities = parse_pipe(row["alternative_mean_probabilities"], float)
        if len(alternatives) != len(probabilities):
            raise ValueError(f"Alternative export mismatch for {component_key(row)}")
        exported = [
            (class_id, probability)
            for class_id, probability in zip(alternatives, probabilities)
            if class_id != predicted
        ]
        invalid = [class_id for class_id, _ in exported if class_id < 0 or class_id > 30]
        if invalid:
            raise ValueError(f"Out-of-range alternative classes for {component_key(row)}: {invalid}")
        # Background has no anatomy-relation profile. It remains available to
        # segmentation/error analyses but is not a hypothesis in the anatomy KG.
        kept = [item for item in exported if item[0] != 0]
        if not kept or len(kept) > 2:
            raise ValueError(
                f"Expected one or two anatomical alternative hypotheses for "
                f"{component_key(row)}, got {kept}"
            )
        for rank, (alternative, probability) in enumerate(kept, start=1):
            target = (
                row["error_label"] == "CLASS_SWAP"
                and int(row["dominant_gt_class_id"]) == alternative
                and float(row["dominant_gt_fraction"]) >= 0.5
            )
            output.append({
                **row, "alternative_class_id": alternative,
                "alternative_probability": probability, "alternative_rank": rank,
                "exact_pair_swap_target": bool(target),
            })
    return output


def reliability(row: dict) -> float:
    entropy = 1.0 - min(float(row["mean_entropy"]) / math.log(31), 1.0)
    tool = 1.0 - min(float(row["mean_tool_probability"]), 1.0)
    return max(entropy * tool, 0.0)


def build_features(
    candidates: list[dict], oof_root: Path, profiles, minimum_observations: int,
    minimum_station_rows: int, shrinkage: float, seed: int,
):
    grouped = defaultdict(list)
    for row in candidates:
        grouped[(int(row["fold"]), row["frame"])].append(row)
    array_paths = {}
    for fold in range(5):
        for row in read_csv(oof_root / f"fold_{fold}/frame_index.csv"):
            array_paths[(fold, row["frame"])] = oof_root / f"fold_{fold}" / row["array_file"]
    output, pair_audit = [], {}
    for (fold, frame), frame_rows in sorted(grouped.items()):
        with np.load(array_paths[(fold, frame)]) as arrays:
            prediction = arrays["prediction"]
        components = recover_global_components(prediction)
        visible_classes = [class_id for class_id in range(1, 31) if np.any(prediction == class_id)]
        distance_maps = {
            class_id: ndimage.distance_transform_edt(prediction != class_id)
            for class_id in visible_classes
        }
        presence = {f"presence_{class_id}": float(class_id in visible_classes) for class_id in range(1, 31)}
        pair_cache = {}
        for row in frame_rows:
            predicted, alternative = int(row["predicted_class_id"]), int(row["alternative_class_id"])
            pair = (predicted, alternative)
            if pair not in pair_cache:
                partners = eligible_partners(
                    profiles[fold], predicted, alternative,
                    minimum_observations, minimum_station_rows,
                )
                pair_cache[pair] = (
                    partners,
                    deterministic_partner_map(partners, fold, predicted, alternative, seed),
                )
                pair_audit[(fold, predicted, alternative)] = partners
            partners, shuffled_map = pair_cache[pair]
            focus = components[int(row["component_id"])]
            if int(prediction[focus][0]) != predicted:
                raise ValueError(f"Component/prediction mismatch for {component_key(row)}")
            kg = hypothesis_relation_score(
                prediction, focus, predicted, alternative, profiles[fold], distance_maps,
                partners, shrinkage,
            )
            shuffled = hypothesis_relation_score(
                prediction, focus, predicted, alternative, profiles[fold], distance_maps,
                partners, shrinkage, shuffled_map,
            )
            output.append({
                **row, **kg,
                "shuffled_kg_predicted_log_compatibility": shuffled["kg_predicted_log_compatibility"],
                "shuffled_kg_alternative_log_compatibility": shuffled["kg_alternative_log_compatibility"],
                "shuffled_kg_llr": shuffled["kg_llr"],
                "shuffled_kg_visible_partners": shuffled["kg_visible_partners"],
                "shuffled_kg_evidence_available": shuffled["kg_evidence_available"],
                "relation_reliability": reliability(row), **presence,
            })
    pair_rows = [{
        "fold": fold, "predicted_class_id": predicted, "alternative_class_id": alternative,
        "eligible_partner_count": len(partners),
        "eligible_partner_ids": "|".join(map(str, partners)),
    } for (fold, predicted, alternative), partners in sorted(pair_audit.items())]
    return output, pair_rows


def pair_identity(rows: list[dict]) -> np.ndarray:
    result = np.zeros((len(rows), 60), dtype=float)
    for index, row in enumerate(rows):
        result[index, int(row["predicted_class_id"]) - 1] = 1.0
        result[index, 30 + int(row["alternative_class_id"]) - 1] = 1.0
    return result


def method_matrices(rows: list[dict]) -> dict[str, np.ndarray]:
    p2 = np.asarray([[float(row["alternative_probability"])] for row in rows])
    identity = pair_identity(rows)
    kg_names = [
        "kg_llr", "kg_predicted_log_compatibility", "kg_alternative_log_compatibility",
        "kg_visible_partners", "kg_evidence_available",
    ]
    shuffled_names = [
        "shuffled_kg_llr", "shuffled_kg_predicted_log_compatibility",
        "shuffled_kg_alternative_log_compatibility", "shuffled_kg_visible_partners",
        "shuffled_kg_evidence_available",
    ]
    kg = np.asarray([[float(row[name]) for name in kg_names] for row in rows])
    shuffled = np.asarray([[float(row[name]) for name in shuffled_names] for row in rows])
    presence = np.asarray([
        [float(row[f"presence_{class_id}"]) for class_id in range(1, 31)] for row in rows
    ])
    local = np.asarray([[
        1.0 - float(row["mean_predicted_class_probability"]),
        1.0 - float(row["mean_top1_probability"]),
        1.0 - float(row["mean_top1_top2_margin"]),
        float(row["mean_entropy"]), math.log(max(float(row["area_fraction"]), 1e-12)),
    ] for row in rows])
    return {
        "M0_ALT_PROBABILITY_RAW": p2,
        "M1_PROBABILITY_PAIR_ID": np.column_stack([p2, identity]),
        "M2_KG_PAIR_ID": np.column_stack([identity, kg]),
        "M3_PROBABILITY_PAIR_ID_KG": np.column_stack([p2, identity, kg]),
        "M4_PROBABILITY_PAIR_ID_SHUFFLED_KG": np.column_stack([p2, identity, shuffled]),
        "M5_PROBABILITY_PAIR_ID_PRESENCE": np.column_stack([p2, identity, presence]),
        "M6_CONFIDENCE_PROBABILITY_PAIR_ID": np.column_stack([local, p2, identity]),
    }


def aggregate_components(rows: list[dict], y: np.ndarray, risks: dict[str, np.ndarray]):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[component_key(row)].append(index)
    component_rows, component_y = [], []
    component_risks = {method: [] for method in risks}
    for key, indices in sorted(groups.items()):
        representative = rows[indices[0]]
        positives = [index for index in indices if y[index]]
        component_rows.append({
            "fold": key[0], "case_id": key[1], "frame": key[2], "component_id": key[3],
            "predicted_class_id": representative["predicted_class_id"],
            "area_bin": (
                "tiny" if float(representative["area_fraction"]) < 0.001
                else "small" if float(representative["area_fraction"]) < 0.01 else "large"
            ),
            "mean_tool_probability": representative["mean_tool_probability"],
            "relation_reliability": representative["relation_reliability"],
            "kg_evidence_available": max(float(rows[index]["kg_evidence_available"]) for index in indices),
            "positive_alternative_class_id": (
                rows[positives[0]]["alternative_class_id"] if positives else "NONE"
            ),
        })
        component_y.append(bool(positives))
        for method, risk in risks.items():
            component_risks[method].append(float(np.max(risk[indices])))
    return (
        component_rows, np.asarray(component_y, dtype=np.uint8),
        {method: np.asarray(values) for method, values in component_risks.items()},
    )


def aggregate_subset_risk(rows: list[dict], risk: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if len(selected) != len(rows) or len(risk) != int(selected.sum()):
        raise ValueError("Subset risk must align with selected candidate rows")
    groups = defaultdict(list)
    for risk_index, row_index in enumerate(np.flatnonzero(selected)):
        groups[component_key(rows[row_index])].append(risk[risk_index])
    return np.asarray([max(values) for _, values in sorted(groups.items())])


def evaluate(rows: list[dict], coverages: list[float], seed: int):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["exact_pair_swap_target"]) for row in rows], dtype=np.uint8)
    matrices = method_matrices(rows)
    risks, thresholds, coefficients = {}, {}, []
    for method, matrix in matrices.items():
        risk = np.zeros(len(rows), dtype=float)
        for held_case in cases:
            train, test = case_ids != held_case, case_ids == held_case
            if METHODS[method] == "raw":
                train_risk = matrix[train, 0]
                risk[test] = matrix[test, 0]
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=4000, class_weight="balanced", random_state=seed,
                        solver="liblinear", C=1.0,
                    ),
                ).fit(matrix[train], y[train])
                train_risk = model.predict_proba(matrix[train])[:, 1]
                risk[test] = model.predict_proba(matrix[test])[:, 1]
                classifier = model.named_steps["logisticregression"]
                coefficients.append({
                    "method": method, "held_case": held_case,
                    "feature_count": matrix.shape[1],
                    "coefficient_l2_norm": float(np.linalg.norm(classifier.coef_[0])),
                    "intercept": float(classifier.intercept_[0]),
                })
            train_component_risk = aggregate_subset_risk(rows, train_risk, train)
            for coverage in coverages:
                thresholds[(method, held_case, coverage)] = float(
                    np.quantile(train_component_risk, 1.0 - coverage)
                )
        risks[method] = risk

    candidate_metrics = [dict(row, level="candidate") for row in macro_metrics(y, risks, case_ids)]
    component_rows, component_y, component_risks = aggregate_components(rows, y, risks)
    component_cases = np.asarray([row["case_id"] for row in component_rows])
    component_metrics = [
        dict(row, level="component") for row in macro_metrics(component_y, component_risks, component_cases)
    ]
    operational, component_predictions, candidate_predictions = [], [], []
    for method, risk in component_risks.items():
        for case in cases:
            selected = component_cases == case
            for coverage in coverages:
                operational.append({
                    "method": method, "target_coverage": coverage, "case_id": case,
                    **threshold_metrics(
                        component_y[selected], risk[selected], thresholds[(method, case, coverage)]
                    ),
                })
        for index, row in enumerate(component_rows):
            component_predictions.append({
                "method": method, **row, "true_label": int(component_y[index]),
                "risk": float(risk[index]),
            })
    for method, risk in risks.items():
        for index, row in enumerate(rows):
            candidate_predictions.append({
                "method": method, "fold": row["fold"], "case_id": row["case_id"],
                "frame": row["frame"], "component_id": row["component_id"],
                "predicted_class_id": row["predicted_class_id"],
                "alternative_class_id": row["alternative_class_id"],
                "true_label": int(y[index]), "risk": float(risk[index]),
            })
    return {
        "candidate_y": y, "candidate_risks": risks, "candidate_cases": case_ids,
        "component_rows": component_rows, "component_y": component_y,
        "component_risks": component_risks, "component_cases": component_cases,
        "metrics": candidate_metrics + component_metrics, "operational": operational,
        "candidate_predictions": candidate_predictions,
        "component_predictions": component_predictions, "coefficients": coefficients,
    }


def component_subgroups(result):
    rows, y, risks, cases = (
        result["component_rows"], result["component_y"],
        result["component_risks"], result["component_cases"],
    )
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in rows]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in rows]),
        "kg_available": np.asarray([float(row["kg_evidence_available"]) > 0 for row in rows]),
        "kg_unknown": np.asarray([float(row["kg_evidence_available"]) == 0 for row in rows]),
        "reliability_low": np.asarray([float(row["relation_reliability"]) < 0.5 for row in rows]),
        "reliability_high": np.asarray([float(row["relation_reliability"]) >= 0.5 for row in rows]),
        "tool_low": np.asarray([float(row["mean_tool_probability"]) < 0.1 for row in rows]),
        "tool_high": np.asarray([float(row["mean_tool_probability"]) >= 0.1 for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        if not selected.any():
            continue
        for method, risk in risks.items():
            pooled = binary_metrics(y[selected], risk[selected])
            per_case = []
            for case in sorted(set(cases[selected])):
                metric = binary_metrics(y[selected & (cases == case)], risk[selected & (cases == case)])
                if metric["auprc"] is not None:
                    per_case.append(metric["auprc"])
            output.append({
                "group": group, "method": method, **pooled,
                "evaluable_cases": len(per_case),
                "macro_case_auprc": float(np.mean(per_case)) if per_case else None,
            })
    return output


def pair_metrics(rows, y, risks):
    pairs = defaultdict(list)
    for index, row in enumerate(rows):
        pairs[(int(row["predicted_class_id"]), int(row["alternative_class_id"]))].append(index)
    output, macro = [], []
    for pair, indices in sorted(pairs.items()):
        selected_y = y[indices]
        positive_cases = len({rows[index]["case_id"] for index in indices if y[index]})
        if int(selected_y.sum()) < 5 or positive_cases < 2 or len(np.unique(selected_y)) < 2:
            continue
        for method, risk in risks.items():
            metric = binary_metrics(selected_y, risk[indices])
            output.append({
                "predicted_class_id": pair[0], "alternative_class_id": pair[1],
                "positive_cases": positive_cases, "method": method, **metric,
            })
    for method in risks:
        selected = [row for row in output if row["method"] == method]
        if selected:
            macro.append({
                "method": method, "supported_ordered_pairs": len(selected),
                "pair_macro_auprc": float(np.mean([row["auprc"] for row in selected])),
                "pair_macro_auroc": float(np.mean([row["auroc"] for row in selected])),
            })
    return output, macro


def gate_decision(metrics, subgroups, bootstrap, operational_summary):
    macro = {
        row["method"]: row for row in metrics
        if row["level"] == "component" and row["scope"] == "macro_case_mean"
    }
    boot = {(row["method"], row["reference"]): row for row in bootstrap}
    subgroup = {(row["group"], row["method"]): row for row in subgroups}
    operational = {(row["method"], float(row["target_coverage"])): row for row in operational_summary}
    primary = "M3_PROBABILITY_PAIR_ID_KG"
    comparisons, core = {}, []
    for reference in ["M0_ALT_PROBABILITY_RAW", "M1_PROBABILITY_PAIR_ID"]:
        item = boot[(primary, reference)]
        delta = float(macro[primary]["auprc"]) - float(macro[reference]["auprc"])
        comparisons[reference] = {
            "delta_auprc": delta, "cases_improved": int(item["cases_improved"]),
            "ci_low": float(item["case_bootstrap_ci_low"]),
            "ci_high": float(item["case_bootstrap_ci_high"]),
        }
        core.append(delta >= 0.02 and int(item["cases_improved"]) >= 4 and float(item["case_bootstrap_ci_low"]) > 0)
    non_tiny_delta = (
        float(subgroup[("area_non_tiny", primary)]["macro_case_auprc"])
        - float(subgroup[("area_non_tiny", "M1_PROBABILITY_PAIR_ID")]["macro_case_auprc"])
    )
    ppv_deltas = {}
    for coverage in [0.05, 0.10]:
        current = operational[(primary, coverage)]["min_case_ppv"]
        baseline = operational[("M1_PROBABILITY_PAIR_ID", coverage)]["min_case_ppv"]
        ppv_deltas[str(coverage)] = (
            float(current) - float(baseline) if current is not None and baseline is not None else None
        )
    controls = []
    for control in ["M4_PROBABILITY_PAIR_ID_SHUFFLED_KG", "M5_PROBABILITY_PAIR_ID_PRESENCE"]:
        item = boot[(control, "M1_PROBABILITY_PAIR_ID")]
        delta = float(macro[control]["auprc"]) - float(macro["M1_PROBABILITY_PAIR_ID"]["auprc"])
        if delta >= 0.02 and int(item["cases_improved"]) >= 4 and float(item["case_bootstrap_ci_low"]) > 0:
            controls.append(control)
    passed = (
        all(core) and non_tiny_delta >= -0.01 and not controls
        and all(value is not None and value >= -0.05 for value in ppv_deltas.values())
    )
    return {
        "status": "MULTIPAIR_KG_INFORMATION_GAIN_GO" if passed else "MULTIPAIR_KG_INFORMATION_GAIN_NO_GO",
        "primary": primary, "comparisons": comparisons,
        "non_tiny_delta_vs_pair_control": non_tiny_delta,
        "min_case_ppv_deltas_vs_pair_control": ppv_deltas,
        "controls_reproducing_gate": controls, "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    error_manifest = json.loads((args.error_corpus / "manifest.json").read_text())
    oof_manifest = json.loads((args.oof_root / "manifest.json").read_text())
    if error_manifest.get("test_data_used") is not False or oof_manifest.get("test_data_used") is not False:
        raise ValueError("Multi-pair KG requires test-free OOF artifacts")
    components = read_csv(args.error_corpus / "predicted_component_errors.csv")
    candidates = candidate_rows(components, args.high_risk_weight)
    if len(candidates) != 1412 or sum(parse_bool(row["exact_pair_swap_target"]) for row in candidates) != 249:
        raise ValueError("Frozen cohort mismatch; expected 1,412 anatomy candidates and 249 exact swaps")
    profiles, profile_audit = load_all_profiles(args.kg_root)
    rows, pair_audit = build_features(
        candidates, args.oof_root, profiles, args.minimum_observations,
        args.minimum_station_rows, args.shrinkage_observations, args.seed,
    )
    coverages = [0.05, 0.10, 0.20, 0.30]
    result = evaluate(rows, coverages, args.seed)
    operational_summary = summarize_operational(result["operational"])
    subgroups = component_subgroups(result)
    pairs, pair_macro = pair_metrics(
        rows, result["candidate_y"], result["candidate_risks"],
    )
    component_metrics = [row for row in result["metrics"] if row["level"] == "component"]
    bootstrap = []
    for reference in ["M0_ALT_PROBABILITY_RAW", "M1_PROBABILITY_PAIR_ID"]:
        bootstrap.extend(bootstrap_deltas(
            component_metrics, [method for method in METHODS if method != reference],
            reference, args.bootstrap_replicates, args.seed,
        ))
    decision = gate_decision(result["metrics"], subgroups, bootstrap, operational_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "candidate_features.csv", rows)
    write_csv(args.output_root / "profile_support_audit.csv", profile_audit)
    write_csv(args.output_root / "candidate_pair_support_audit.csv", pair_audit)
    write_csv(args.output_root / "metrics.csv", result["metrics"])
    write_csv(args.output_root / "operational_metrics.csv", result["operational"])
    write_csv(args.output_root / "operational_summary.csv", operational_summary)
    write_csv(args.output_root / "component_subgroup_metrics.csv", subgroups)
    write_csv(args.output_root / "supported_pair_metrics.csv", pairs)
    write_csv(args.output_root / "pair_macro_metrics.csv", pair_macro)
    write_csv(args.output_root / "case_bootstrap_deltas.csv", bootstrap)
    write_csv(args.output_root / "coefficient_audit.csv", result["coefficients"])
    write_csv(args.output_root / "candidate_risk_predictions.csv", result["candidate_predictions"])
    write_csv(args.output_root / "component_risk_predictions.csv", result["component_predictions"])
    macro = {
        row["method"]: row for row in result["metrics"]
        if row["level"] == "component" and row["scope"] == "macro_case_mean"
    }
    manifest = {
        "artifact": "SAFE-Graph multi-pair contrastive KG verifier v2",
        "version": args.output_root.name, **decision,
        "frozen_protocol": "docs/13.08__MULTIPAIR_CONTRASTIVE_KG_V2_PROTOCOL.md",
        "source_error_corpus": str(args.error_corpus), "source_oof": str(args.oof_root),
        "source_kg": str(args.kg_root), "components": len(result["component_rows"]),
        "candidate_hypotheses": len(rows), "positive_components": int(result["component_y"].sum()),
        "positive_candidates": int(result["candidate_y"].sum()),
        "excluded_background_candidate_hypotheses": 34,
        "methods": METHODS, "evaluation": "five-case leave-one-case-out",
        "missing_partner_policy": "UNKNOWN; zero contribution",
        "station_policy": "fold-train latent mixture; no held-case station input",
        "test_data_used": False, "clinical_expert_complete": False,
        "clinical_warning_ready": False, "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": decision["status"],
        "component_macro_case_auprc": {method: float(row["auprc"]) for method, row in macro.items()},
        "comparisons": decision["comparisons"],
        "components": len(result["component_rows"]), "candidates": len(rows),
        "positive_components": int(result["component_y"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
