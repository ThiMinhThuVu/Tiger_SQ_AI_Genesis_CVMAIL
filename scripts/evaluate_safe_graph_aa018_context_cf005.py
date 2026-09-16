#!/usr/bin/env python3
"""Test fold-safe AA-018 single-frame context beyond P2-DENSE for CF005."""
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
    bootstrap_deltas,
    macro_metrics,
    parse_bool,
    summarize_operational,
)
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.evaluate_safe_graph_q3_ai_provisional import recover_global_components  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


RELATION_FEATURES = [
    "aa018_evidence_available", "aa018_distance_norm", "aa018_touch",
    "aa018_distance_violation", "aa018_touch_mismatch",
    "aa018_pred7_violation", "aa018_pred7_touch_mismatch",
    "aa018_pred19_compatibility", "aa018_pred19_touch",
]
GATED_RELATION_FEATURES = [f"gated_{name}" for name in RELATION_FEATURES]
PAIR_TOUCH_FEATURES = [
    "pair_partner_available", "pair_partner_distance_norm", "pair_partner_touch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-cohort", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/p2_dense_cf005_v1/dense_pair_cohort.csv",
    )
    parser.add_argument(
        "--dense-oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_p2_dense_cf005_v1",
    )
    parser.add_argument(
        "--kg-root", type=Path,
        default=ROOT / "artifacts/safe_graph_knowledge/kg_v1_20260808",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tool-threshold", type=float, default=0.10)
    parser.add_argument("--reliability-threshold", type=float, default=0.50)
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


def station_from_frame(case_id: str, frame: str) -> str:
    stem = Path(frame).stem
    prefix = f"{case_id}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Cannot parse station from {frame} for {case_id}")
    station = stem[len(prefix):]
    if not station:
        raise ValueError(f"Empty station in {frame}")
    return station


def load_aa018_envelopes(kg_root: Path) -> tuple[dict[int, dict], list[dict]]:
    """Pool station rows for protocol relation AA-018 using outer-fold train GT only."""
    envelopes = {}
    audit = []
    for fold in range(5):
        path = kg_root / f"outer_fold_{fold}_train/anatomy_pair_relations.csv"
        selected = [
            row for row in read_csv(path)
            if {int(row["anatomy_i_id"]), int(row["anatomy_j_id"])} == {6, 7}
            and int(row["relation_observations"]) > 0
        ]
        if not selected:
            raise ValueError(f"Fold {fold} has no class-6/7 AA-018 observations")
        weights = np.asarray([int(row["relation_observations"]) for row in selected], dtype=float)
        envelope = {
            "distance_q05": min(float(row["boundary_distance_norm_q05"]) for row in selected),
            "distance_q95": max(float(row["boundary_distance_norm_q95"]) for row in selected),
            "touch_probability": float(np.average(
                [float(row["touch_probability_when_covisible"]) for row in selected], weights=weights,
            )),
            "observations": int(weights.sum()),
            "station_rows": len(selected),
        }
        envelopes[fold] = envelope
        audit.append({"fold": fold, "source": str(path), **envelope})
    return envelopes, audit


def relation_measurements(
    prediction: np.ndarray, focus: np.ndarray, predicted_class: int, envelope: dict,
) -> dict[str, float]:
    diagonal = math.hypot(*prediction.shape)
    class6 = prediction == 6
    available = bool(class6.any())
    if available:
        distance = float(ndimage.distance_transform_edt(~class6)[focus].min() / diagonal)
        touch = float(distance <= math.sqrt(2.0) / diagonal)
        low, high = envelope["distance_q05"], envelope["distance_q95"]
        width = max(high - low, 1.0 / diagonal)
        violation = min(max(low - distance, distance - high, 0.0) / width, 10.0)
        touch_mismatch = envelope["touch_probability"] * (1.0 - touch)
        compatibility = 1.0 / (1.0 + violation)
    else:
        # Missing partner is UNKNOWN: all relation values are neutral, never a violation.
        distance = touch = violation = touch_mismatch = compatibility = 0.0
    pred7 = float(predicted_class == 7)
    pred19 = float(predicted_class == 19)
    return {
        "aa018_evidence_available": float(available),
        "aa018_distance_norm": distance,
        "aa018_touch": touch,
        "aa018_distance_violation": violation,
        "aa018_touch_mismatch": touch_mismatch,
        "aa018_pred7_violation": pred7 * violation,
        "aa018_pred7_touch_mismatch": pred7 * touch_mismatch,
        "aa018_pred19_compatibility": pred19 * compatibility * float(available),
        "aa018_pred19_touch": pred19 * touch,
    }


def pair_touch_measurements(
    prediction: np.ndarray, focus: np.ndarray, predicted_class: int,
) -> dict[str, float]:
    counterpart = 19 if predicted_class == 7 else 7
    partner = prediction == counterpart
    available = bool(partner.any())
    if available:
        diagonal = math.hypot(*prediction.shape)
        distance = float(ndimage.distance_transform_edt(~partner)[focus].min() / diagonal)
        touch = float(distance <= math.sqrt(2.0) / diagonal)
    else:
        distance = touch = 0.0
    return {
        "pair_partner_available": float(available),
        "pair_partner_distance_norm": distance,
        "pair_partner_touch": touch,
    }


def reliability(row: dict[str, str]) -> float:
    entropy_reliability = 1.0 - min(float(row["local_entropy"]) / math.log(31), 1.0)
    tool_reliability = 1.0 - min(float(row["mean_tool_probability"]), 1.0)
    return max(entropy_reliability * tool_reliability, 0.0)


def build_feature_rows(
    rows: list[dict[str, str]], dense_root: Path, envelopes: dict[int, dict],
    tool_threshold: float, reliability_threshold: float,
) -> list[dict]:
    by_frame: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[(int(row["fold"]), row["frame"])].append(row)
    output = []
    for (fold, frame), frame_rows in sorted(by_frame.items()):
        array_path = dense_root / f"fold_{fold}/frame_arrays/{Path(frame).stem}.npz"
        with np.load(array_path) as arrays:
            prediction = arrays["prediction"]
        components = recover_global_components(prediction)
        for row in frame_rows:
            component = components[int(row["component_id"])]
            predicted = int(row["predicted_class_id"])
            if int(prediction[component][0]) != predicted:
                raise ValueError(f"Component mismatch for {frame}/{row['component_id']}")
            relation = relation_measurements(prediction, component, predicted, envelopes[fold])
            pair_touch = pair_touch_measurements(prediction, component, predicted)
            score_reliability = reliability(row)
            tool_clear = float(row["mean_tool_probability"]) < tool_threshold
            uncertainty_reliable = score_reliability >= reliability_threshold
            gate = float(
                relation["aa018_evidence_available"] > 0 and tool_clear and uncertainty_reliable
            )
            station = station_from_frame(row["case_id"], frame)
            output.append({
                **row,
                "station_oracle": station,
                "direction_pred19": float(predicted == 19),
                "aa018_reliability": score_reliability,
                "aa018_tool_clear": float(tool_clear),
                "aa018_uncertainty_reliable": float(uncertainty_reliable),
                "aa018_evidence_gate": gate,
                **relation,
                **{f"gated_{name}": gate * value for name, value in relation.items()},
                **pair_touch,
            })
    if len(output) != len(rows):
        raise ValueError("Context feature row count changed")
    return output


def method_features(rows: list[dict]) -> dict[str, list[str]]:
    stations = sorted({row["station_oracle"] for row in rows})
    for row in rows:
        for station in stations:
            row[f"station_oracle_{station}"] = float(row["station_oracle"] == station)
    station_columns = [f"station_oracle_{station}" for station in stations]
    return {
        "P2_DENSE_RAW": ["dense_counterpart_probability"],
        "K0_DIRECTION": ["direction_pred19"],
        "K1_AA018_GATED_ONLY": GATED_RELATION_FEATURES,
        "K2_P2_DIRECTION": ["dense_counterpart_probability", "direction_pred19"],
        "K3_P2_DIRECTION_AA018_GATED": [
            "dense_counterpart_probability", "direction_pred19", *GATED_RELATION_FEATURES,
        ],
        "K4_P2_DIRECTION_AA018_UNGATED": [
            "dense_counterpart_probability", "direction_pred19", *RELATION_FEATURES,
        ],
        "K5_P2_DIRECTION_STATION_ORACLE": [
            "dense_counterpart_probability", "direction_pred19", *station_columns,
        ],
        "K6_P2_DIRECTION_EMPIRICAL_PAIR_TOUCH": [
            "dense_counterpart_probability", "direction_pred19", *PAIR_TOUCH_FEATURES,
        ],
    }


def evaluate(rows: list[dict], methods: dict[str, list[str]], coverages, seed):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["pair_swap_target"]) for row in rows], dtype=np.uint8)
    risks = {}
    thresholds = {}
    coefficients = []
    for method, names in methods.items():
        x = np.asarray([[float(row[name]) for name in names] for row in rows])
        risk = np.zeros(len(rows), dtype=float)
        for held_case in cases:
            train = case_ids != held_case
            test = ~train
            if method == "P2_DENSE_RAW":
                train_risk = x[train, 0]
                risk[test] = x[test, 0]
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=seed,
                        solver="liblinear",
                    ),
                ).fit(x[train], y[train])
                train_risk = model.predict_proba(x[train])[:, 1]
                risk[test] = model.predict_proba(x[test])[:, 1]
                classifier = model.named_steps["logisticregression"]
                for name, coefficient in zip(names, classifier.coef_[0]):
                    coefficients.append({
                        "method": method, "held_case": held_case, "feature": name,
                        "standardized_coefficient": float(coefficient),
                    })
            for coverage in coverages:
                thresholds[(method, held_case, coverage)] = float(
                    np.quantile(train_risk, 1.0 - coverage)
                )
        risks[method] = risk
    metrics = macro_metrics(y, risks, case_ids)
    operational = []
    predictions = []
    for method, risk in risks.items():
        for held_case in cases:
            selected = case_ids == held_case
            for coverage in coverages:
                operational.append({
                    "method": method, "target_coverage": coverage, "case_id": held_case,
                    **threshold_metrics(y[selected], risk[selected], thresholds[(method, held_case, coverage)]),
                })
        for index, row in enumerate(rows):
            predictions.append({
                "method": method, "fold": row["fold"], "case_id": row["case_id"],
                "frame": row["frame"], "component_id": row["component_id"],
                "predicted_class_id": row["predicted_class_id"], "area_bin": row["area_bin"],
                "aa018_evidence_available": row["aa018_evidence_available"],
                "aa018_evidence_gate": row["aa018_evidence_gate"],
                "true_label": int(y[index]), "risk": float(risk[index]),
            })
    return y, case_ids, risks, metrics, operational, predictions, coefficients


def subgroup_metrics(rows, y, risks, case_ids):
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "direction_pred_7": np.asarray([int(row["predicted_class_id"]) == 7 for row in rows]),
        "direction_pred_19": np.asarray([int(row["predicted_class_id"]) == 19 for row in rows]),
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in rows]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in rows]),
        "aa018_available": np.asarray([float(row["aa018_evidence_available"]) > 0 for row in rows]),
        "aa018_unknown": np.asarray([float(row["aa018_evidence_available"]) == 0 for row in rows]),
        "aa018_gate_on": np.asarray([float(row["aa018_evidence_gate"]) > 0 for row in rows]),
        "aa018_gate_off": np.asarray([float(row["aa018_evidence_gate"]) == 0 for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        for method, risk in risks.items():
            pooled = binary_metrics(y[selected], risk[selected])
            case_values = []
            for case_id in sorted(set(case_ids[selected])):
                subset = selected & (case_ids == case_id)
                result = binary_metrics(y[subset], risk[subset])
                if result["auprc"] is not None:
                    case_values.append(result["auprc"])
            output.append({
                "group": group, "method": method, **pooled,
                "evaluable_cases": len(case_values),
                "macro_case_auprc": float(np.mean(case_values)) if case_values else None,
            })
    return output


def evidence_summary(rows: list[dict]) -> list[dict]:
    groups = {"all": rows}
    for case in sorted({row["case_id"] for row in rows}):
        groups[f"case:{case}"] = [row for row in rows if row["case_id"] == case]
    output = []
    for group, selected in groups.items():
        positives = [row for row in selected if parse_bool(row["pair_swap_target"])]
        available = [row for row in selected if float(row["aa018_evidence_available"]) > 0]
        gate_on = [row for row in selected if float(row["aa018_evidence_gate"]) > 0]
        output.append({
            "group": group, "components": len(selected), "positives": len(positives),
            "evidence_available": len(available), "evidence_available_fraction": len(available) / len(selected),
            "positive_evidence_available": sum(parse_bool(row["pair_swap_target"]) for row in available),
            "gate_on": len(gate_on), "gate_on_fraction": len(gate_on) / len(selected),
            "positive_gate_on": sum(parse_bool(row["pair_swap_target"]) for row in gate_on),
        })
    return output


def gate_decision(metrics, subgroups, bootstrap, operational_summary):
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    boot = {(row["method"], row["reference"]): row for row in bootstrap}
    subgroup = {(row["group"], row["method"]): row for row in subgroups}
    operational = {(row["method"], float(row["target_coverage"])): row for row in operational_summary}
    primary = "K3_P2_DIRECTION_AA018_GATED"
    controls = ["K2_P2_DIRECTION", "P2_DENSE_RAW"]
    comparisons = {}
    core_passes = []
    for reference in controls:
        result = boot[(primary, reference)]
        delta = float(macro[primary]["auprc"]) - float(macro[reference]["auprc"])
        comparisons[reference] = {
            "macro_case_delta_auprc": delta,
            "cases_improved": int(result["cases_improved"]),
            "case_bootstrap_ci_low": float(result["case_bootstrap_ci_low"]),
            "case_bootstrap_ci_high": float(result["case_bootstrap_ci_high"]),
        }
        core_passes.append(
            delta >= 0.02 and int(result["cases_improved"]) >= 4
            and float(result["case_bootstrap_ci_low"]) > 0
        )
    non_tiny_delta = (
        float(subgroup[("area_non_tiny", primary)]["macro_case_auprc"])
        - float(subgroup[("area_non_tiny", "P2_DENSE_RAW")]["macro_case_auprc"])
    )
    ppv_deltas = {}
    for coverage in [0.05, 0.10]:
        current = operational[(primary, coverage)]["min_case_ppv"]
        baseline = operational[("P2_DENSE_RAW", coverage)]["min_case_ppv"]
        ppv_deltas[str(coverage)] = (
            float(current) - float(baseline) if current is not None and baseline is not None else None
        )
    ppv_safe = all(value is not None and value >= -0.05 for value in ppv_deltas.values())
    passed = all(core_passes) and non_tiny_delta >= -0.01 and ppv_safe
    return {
        "status": "AA018_INFORMATION_GAIN_GO" if passed else "AA018_INFORMATION_GAIN_NO_GO",
        "primary": primary, "comparisons": comparisons,
        "non_tiny_delta_vs_raw_p2": non_tiny_delta,
        "min_case_ppv_deltas_vs_raw_p2": ppv_deltas,
        "material_ppv_degradation_definition": "absolute min-case PPV delta below -0.05",
        "passed": passed,
    }


def render_readme(macro: dict, gate: dict, evidence: list[dict]) -> str:
    table = []
    baseline = float(macro["P2_DENSE_RAW"]["auprc"])
    for method, row in macro.items():
        table.append(
            f"| {method} | {float(row['auprc']):.6f} | {float(row['auprc']) - baseline:+.6f} | "
            f"{float(row['auroc']):.6f} |"
        )
    all_evidence = next(row for row in evidence if row["group"] == "all")
    return "\n".join([
        "# CF005 AA-018 context information-gain evaluation", "", "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Verification Status: exploratory, AI-provisional relation, OOF validation only",
        f"- Decision: **{gate['status']}**", "", "## Macro-case comparison", "",
        "| Method | AUPRC | Delta vs raw P2 | AUROC |", "|---|---:|---:|---:|", *table, "",
        "## Evidence availability", "",
        f"- AA-018 observable: `{all_evidence['evidence_available']}/{all_evidence['components']}` components.",
        f"- Safety gate on: `{all_evidence['gate_on']}/{all_evidence['components']}` components.",
        "- Missing class-6 partner is UNKNOWN and contributes no violation.",
        "- Station filename/one-hot appears only in K5 oracle and is non-deployable.", "",
        "No result here authorizes clinical deployment, automatic correction, or ontology-loss training.", "",
    ])


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    dense_manifest = json.loads((args.dense_oof_root / "manifest.json").read_text())
    if dense_manifest.get("test_data_used") is not False:
        raise ValueError("AA-018 evaluation requires test-free OOF predictions")
    rows = read_csv(args.pair_cohort)
    if len(rows) != 744 or sum(parse_bool(row["pair_swap_target"]) for row in rows) != 99:
        raise ValueError("Expected frozen CF005 cohort with 744 components and 99 positives")
    envelopes, envelope_audit = load_aa018_envelopes(args.kg_root)
    rows = build_feature_rows(
        rows, args.dense_oof_root, envelopes, args.tool_threshold, args.reliability_threshold,
    )
    methods = method_features(rows)
    coverages = [0.05, 0.10, 0.20, 0.30]
    y, case_ids, risks, metrics, operational, predictions, coefficients = evaluate(
        rows, methods, coverages, args.seed,
    )
    operational_summary = summarize_operational(operational)
    subgroups = subgroup_metrics(rows, y, risks, case_ids)
    evidence = evidence_summary(rows)
    bootstrap = []
    for reference in ["K2_P2_DIRECTION", "P2_DENSE_RAW"]:
        bootstrap.extend(bootstrap_deltas(
            metrics,
            [method for method in methods if method != reference],
            reference, args.bootstrap_replicates, args.seed,
        ))
    gate = gate_decision(metrics, subgroups, bootstrap, operational_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "component_features.csv", rows)
    write_csv(args.output_root / "aa018_fold_envelopes.csv", envelope_audit)
    write_csv(args.output_root / "metrics.csv", metrics)
    write_csv(args.output_root / "operational_metrics.csv", operational)
    write_csv(args.output_root / "operational_summary.csv", operational_summary)
    write_csv(args.output_root / "subgroup_metrics.csv", subgroups)
    write_csv(args.output_root / "evidence_availability.csv", evidence)
    write_csv(args.output_root / "case_bootstrap_deltas.csv", bootstrap)
    write_csv(args.output_root / "coefficient_stability.csv", coefficients)
    write_csv(args.output_root / "oof_risk_predictions.csv", predictions)
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    manifest = {
        "artifact": "CF005 AA-018 anatomical context information-gain evaluation",
        "version": args.output_root.name, **gate,
        "relation": {
            "id": "AA-018", "subject": "7 Fatty tissue esophagus",
            "predicate": "attached_to_or_near", "object": "6 Esophagus",
            "status": "annotation-protocol relation; pending clinical review",
        },
        "source_pair_cohort": str(args.pair_cohort),
        "source_oof": str(args.dense_oof_root),
        "source_kg": str(args.kg_root),
        "methods": methods,
        "outer_evaluation": "leave-one-case-out across five OOF validation cases",
        "missing_partner_policy": "UNKNOWN; never violation",
        "tool_gate": f"mean component tool probability < {args.tool_threshold}",
        "uncertainty_gate": f"entropy/tool reliability >= {args.reliability_threshold}",
        "station_oracle_policy": "K5 only; non-deployable; never used by K3",
        "test_data_used": False,
        "clinical_expert_complete": False,
        "clinical_warning_ready": False,
        "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "README.md").write_text(render_readme(macro, gate, evidence))
    print(json.dumps({
        "status": gate["status"],
        "macro_case_auprc": {method: float(row["auprc"]) for method, row in macro.items()},
        "primary_comparisons": gate["comparisons"],
        "evidence_all": evidence[0],
    }, indent=2))


if __name__ == "__main__":
    main()
