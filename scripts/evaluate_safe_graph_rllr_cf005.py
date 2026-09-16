#!/usr/bin/env python3
"""Evaluate fold-safe dual-hypothesis relational LLR beyond P2-DENSE."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.analyze_safe_graph_p2_cf005 import (  # noqa: E402
    bootstrap_deltas, macro_metrics, parse_bool, summarize_operational,
)
from scripts.evaluate_safe_graph_aa018_context_cf005 import (  # noqa: E402
    reliability, station_from_frame,
)
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.evaluate_safe_graph_q3_ai_provisional import recover_global_components  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


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


def load_profiles(kg_root: Path, minimum_observations: int, minimum_station_rows: int):
    profiles = {}
    audit = []
    for fold in range(5):
        path = kg_root / f"outer_fold_{fold}_train/anatomy_pair_relations.csv"
        grouped = {7: defaultdict(list), 19: defaultdict(list)}
        for row in read_csv(path):
            first, second = int(row["anatomy_i_id"]), int(row["anatomy_j_id"])
            for hypothesis in [7, 19]:
                if hypothesis not in (first, second):
                    continue
                partner = second if first == hypothesis else first
                if partner in {7, 19} or int(row["relation_observations"]) <= 0:
                    continue
                grouped[hypothesis][partner].append(row)
        eligible = []
        for partner in sorted(set(grouped[7]) & set(grouped[19])):
            support = {}
            for hypothesis in [7, 19]:
                values = grouped[hypothesis][partner]
                support[hypothesis] = {
                    "observations": sum(int(row["relation_observations"]) for row in values),
                    "station_rows": len(values),
                }
            if all(
                support[hypothesis]["observations"] >= minimum_observations
                and support[hypothesis]["station_rows"] >= minimum_station_rows
                for hypothesis in [7, 19]
            ):
                eligible.append(partner)
                audit.append({
                    "fold": fold, "partner_class_id": partner,
                    "h7_observations": support[7]["observations"],
                    "h19_observations": support[19]["observations"],
                    "h7_station_rows": support[7]["station_rows"],
                    "h19_station_rows": support[19]["station_rows"],
                })
        profiles[fold] = {
            "eligible": eligible,
            "rows": {hypothesis: dict(grouped[hypothesis]) for hypothesis in [7, 19]},
        }
    return profiles, audit


def station_mixture_log_compatibility(
    distance: float, touch: float, rows: list[dict[str, str]], diagonal: float,
) -> float:
    observations = np.asarray([int(row["relation_observations"]) for row in rows], dtype=float)
    log_weights = np.log(observations / observations.sum())
    log_compatibilities = []
    for row in rows:
        low = float(row["boundary_distance_norm_q05"])
        high = float(row["boundary_distance_norm_q95"])
        outside = max(low - distance, distance - high, 0.0)
        width = max(high - low, 2.0 / diagonal)
        log_distance = max(-0.5 * (outside / width) ** 2, -30.0)
        count = float(row["relation_observations"])
        probability = (float(row["touch_probability_when_covisible"]) * count + 1.0) / (count + 2.0)
        touch_probability = probability if touch else 1.0 - probability
        log_touch = math.log(max(touch_probability, 1e-6))
        log_compatibilities.append(0.5 * (log_distance + log_touch))
    result = float(logsumexp(log_weights + np.asarray(log_compatibilities)))
    return max(result, math.log(1e-6))


def score_relations(
    prediction: np.ndarray, focus: np.ndarray, predicted_class: int, profile: dict,
    shrinkage_observations: float, partner_map: dict[int, int] | None = None,
    distance_maps: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    diagonal = math.hypot(*prediction.shape)
    predicted_hypothesis = predicted_class
    counterpart_hypothesis = 19 if predicted_class == 7 else 7
    contributions = []
    weights = []
    visible = 0
    for observed_partner in profile["eligible"]:
        if distance_maps is not None:
            distance_map = distance_maps.get(observed_partner)
        else:
            mask = prediction == observed_partner
            distance_map = ndimage.distance_transform_edt(~mask) if mask.any() else None
        if distance_map is None:
            continue  # Missing anatomy is UNKNOWN.
        visible += 1
        distance = float(distance_map[focus].min() / diagonal)
        touch = float(distance <= math.sqrt(2.0) / diagonal)
        profile_partner = partner_map.get(observed_partner, observed_partner) if partner_map else observed_partner
        predicted_rows = profile["rows"][predicted_hypothesis][profile_partner]
        counterpart_rows = profile["rows"][counterpart_hypothesis][profile_partner]
        predicted_log = station_mixture_log_compatibility(
            distance, touch, predicted_rows, diagonal,
        )
        counterpart_log = station_mixture_log_compatibility(
            distance, touch, counterpart_rows, diagonal,
        )
        minimum_support = min(
            sum(int(row["relation_observations"]) for row in predicted_rows),
            sum(int(row["relation_observations"]) for row in counterpart_rows),
        )
        weight = minimum_support / (minimum_support + shrinkage_observations)
        contributions.append(counterpart_log - predicted_log)
        weights.append(weight)
    if not weights:
        return {"rllr": 0.0, "rllr_visible_partners": 0.0, "rllr_evidence_available": 0.0}
    return {
        "rllr": float(np.average(contributions, weights=weights)),
        "rllr_visible_partners": float(visible),
        "rllr_evidence_available": 1.0,
    }


def shuffled_partner_map(eligible: list[int], fold: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed + fold)
    shuffled = np.asarray(eligible).copy()
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and np.all(shuffled == np.asarray(eligible)):
        shuffled = np.roll(shuffled, 1)
    return dict(zip(eligible, shuffled.tolist()))


def build_feature_rows(rows, dense_root, profiles, shrinkage_observations, seed):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[(int(row["fold"]), row["frame"])].append(row)
    output = []
    partner_maps = {
        fold: shuffled_partner_map(profiles[fold]["eligible"], fold, seed) for fold in range(5)
    }
    for (fold, frame), frame_rows in sorted(by_frame.items()):
        path = dense_root / f"fold_{fold}/frame_arrays/{Path(frame).stem}.npz"
        with np.load(path) as arrays:
            prediction = arrays["prediction"]
        components = recover_global_components(prediction)
        distance_maps = {
            partner: ndimage.distance_transform_edt(prediction != partner)
            for partner in profiles[fold]["eligible"] if np.any(prediction == partner)
        }
        presence = {class_id: float(np.any(prediction == class_id)) for class_id in range(1, 31)}
        for row in frame_rows:
            component = components[int(row["component_id"])]
            predicted = int(row["predicted_class_id"])
            relation = score_relations(
                prediction, component, predicted, profiles[fold], shrinkage_observations,
                distance_maps=distance_maps,
            )
            shuffled = score_relations(
                prediction, component, predicted, profiles[fold], shrinkage_observations,
                partner_maps[fold], distance_maps,
            )
            output.append({
                **row,
                "station_oracle": station_from_frame(row["case_id"], frame),
                "rllr_reliability": reliability(row),
                **relation,
                "rllr_shuffled": shuffled["rllr"],
                **{f"presence_{class_id}": value for class_id, value in presence.items()},
            })
    return output, partner_maps


def method_features(rows):
    stations = sorted({row["station_oracle"] for row in rows})
    for row in rows:
        for station in stations:
            row[f"station_{station}"] = float(row["station_oracle"] == station)
    presence = [f"presence_{class_id}" for class_id in range(1, 31) if class_id not in {7, 19}]
    return {
        "R0_P2_DENSE_RAW": {"kind": "raw", "features": ["dense_counterpart_probability"]},
        "R1_RLLR_RAW": {"kind": "raw", "features": ["rllr"]},
        "R2_P2_LEARNED": {"kind": "learned", "features": ["dense_counterpart_probability"]},
        "R3_P2_RLLR": {
            "kind": "learned", "features": ["dense_counterpart_probability", "rllr"],
        },
        "R4_P2_SHUFFLED_RLLR": {
            "kind": "learned", "features": ["dense_counterpart_probability", "rllr_shuffled"],
        },
        "R5_P2_PRESENCE": {
            "kind": "learned", "features": ["dense_counterpart_probability", *presence],
        },
        "R6_P2_STATION_ORACLE": {
            "kind": "learned",
            "features": ["dense_counterpart_probability", *[f"station_{s}" for s in stations]],
        },
    }


def evaluate(rows, methods, coverages, seed):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["pair_swap_target"]) for row in rows], dtype=np.uint8)
    risks, thresholds = {}, {}
    coefficients = []
    for method, specification in methods.items():
        names = specification["features"]
        matrix = np.asarray([[float(row[name]) for name in names] for row in rows])
        risk = np.zeros(len(rows), dtype=float)
        for held_case in cases:
            train, test = case_ids != held_case, case_ids == held_case
            if specification["kind"] == "raw":
                train_risk = matrix[train, 0]
                risk[test] = matrix[test, 0]
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=seed,
                        solver="liblinear",
                    ),
                ).fit(matrix[train], y[train])
                train_risk = model.predict_proba(matrix[train])[:, 1]
                risk[test] = model.predict_proba(matrix[test])[:, 1]
                for name, coefficient in zip(
                    names, model.named_steps["logisticregression"].coef_[0],
                ):
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
    operational, predictions = [], []
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
                "rllr_evidence_available": row["rllr_evidence_available"],
                "rllr_visible_partners": row["rllr_visible_partners"],
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
        "rllr_available": np.asarray([float(row["rllr_evidence_available"]) > 0 for row in rows]),
        "rllr_unknown": np.asarray([float(row["rllr_evidence_available"]) == 0 for row in rows]),
        "reliability_low": np.asarray([float(row["rllr_reliability"]) < 0.5 for row in rows]),
        "reliability_high": np.asarray([float(row["rllr_reliability"]) >= 0.5 for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        if not selected.any():
            continue
        for method, risk in risks.items():
            pooled = binary_metrics(y[selected], risk[selected])
            case_values = []
            for case in sorted(set(case_ids[selected])):
                result = binary_metrics(y[selected & (case_ids == case)], risk[selected & (case_ids == case)])
                if result["auprc"] is not None:
                    case_values.append(result["auprc"])
            output.append({
                "group": group, "method": method, **pooled,
                "evaluable_cases": len(case_values),
                "macro_case_auprc": float(np.mean(case_values)) if case_values else None,
            })
    return output


def evidence_summary(rows):
    groups = {"all": rows}
    for case in sorted({row["case_id"] for row in rows}):
        groups[f"case:{case}"] = [row for row in rows if row["case_id"] == case]
    return [{
        "group": group, "components": len(selected),
        "positives": sum(parse_bool(row["pair_swap_target"]) for row in selected),
        "evidence_available": sum(float(row["rllr_evidence_available"]) > 0 for row in selected),
        "mean_visible_partners": float(np.mean([float(row["rllr_visible_partners"]) for row in selected])),
        "median_visible_partners": float(np.median([float(row["rllr_visible_partners"]) for row in selected])),
    } for group, selected in groups.items()]


def gate_decision(metrics, subgroups, bootstrap, operational_summary):
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    boot = {(row["method"], row["reference"]): row for row in bootstrap}
    subgroup = {(row["group"], row["method"]): row for row in subgroups}
    operational = {(row["method"], float(row["target_coverage"])): row for row in operational_summary}
    primary = "R3_P2_RLLR"
    comparisons = {}
    core = []
    for reference in ["R0_P2_DENSE_RAW", "R2_P2_LEARNED"]:
        result = boot[(primary, reference)]
        delta = float(macro[primary]["auprc"]) - float(macro[reference]["auprc"])
        comparisons[reference] = {
            "delta": delta, "cases_improved": int(result["cases_improved"]),
            "ci_low": float(result["case_bootstrap_ci_low"]),
            "ci_high": float(result["case_bootstrap_ci_high"]),
        }
        core.append(delta >= 0.02 and int(result["cases_improved"]) >= 4 and float(result["case_bootstrap_ci_low"]) > 0)
    non_tiny_delta = (
        float(subgroup[("area_non_tiny", primary)]["macro_case_auprc"])
        - float(subgroup[("area_non_tiny", "R0_P2_DENSE_RAW")]["macro_case_auprc"])
    )
    ppv_deltas = {}
    for coverage in [0.05, 0.10]:
        current = operational[(primary, coverage)]["min_case_ppv"]
        baseline = operational[("R0_P2_DENSE_RAW", coverage)]["min_case_ppv"]
        ppv_deltas[str(coverage)] = float(current) - float(baseline) if current is not None and baseline is not None else None
    controls_reproduce = []
    for control in ["R4_P2_SHUFFLED_RLLR", "R5_P2_PRESENCE"]:
        result = boot[(control, "R0_P2_DENSE_RAW")]
        delta = float(macro[control]["auprc"]) - float(macro["R0_P2_DENSE_RAW"]["auprc"])
        if delta >= 0.02 and int(result["cases_improved"]) >= 4 and float(result["case_bootstrap_ci_low"]) > 0:
            controls_reproduce.append(control)
    passed = (
        all(core) and non_tiny_delta >= -0.01 and not controls_reproduce
        and all(value is not None and value >= -0.05 for value in ppv_deltas.values())
    )
    return {
        "status": "RLLR_INFORMATION_GAIN_GO" if passed else "RLLR_INFORMATION_GAIN_NO_GO",
        "primary": primary, "comparisons": comparisons,
        "non_tiny_delta_vs_raw_p2": non_tiny_delta,
        "min_case_ppv_deltas_vs_raw_p2": ppv_deltas,
        "controls_reproducing_gate": controls_reproduce,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    source_manifest = json.loads((args.dense_oof_root / "manifest.json").read_text())
    if source_manifest.get("test_data_used") is not False:
        raise ValueError("R-LLR requires test-free OOF source")
    cohort = read_csv(args.pair_cohort)
    if len(cohort) != 744 or sum(parse_bool(row["pair_swap_target"]) for row in cohort) != 99:
        raise ValueError("Expected frozen 744/99 CF005 cohort")
    profiles, profile_audit = load_profiles(
        args.kg_root, args.minimum_observations, args.minimum_station_rows,
    )
    rows, partner_maps = build_feature_rows(
        cohort, args.dense_oof_root, profiles, args.shrinkage_observations, args.seed,
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
    for reference in ["R0_P2_DENSE_RAW", "R2_P2_LEARNED"]:
        bootstrap.extend(bootstrap_deltas(
            metrics, [method for method in methods if method != reference],
            reference, args.bootstrap_replicates, args.seed,
        ))
    decision = gate_decision(metrics, subgroups, bootstrap, operational_summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "component_features.csv", rows)
    write_csv(args.output_root / "profile_support_audit.csv", profile_audit)
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
        "artifact": "CF005 dual-hypothesis relational likelihood ratio",
        "version": args.output_root.name, **decision,
        "frozen_protocol": "docs/13.08__RLLR_FROZEN_PROTOCOL.md",
        "minimum_observations_per_hypothesis": args.minimum_observations,
        "minimum_station_rows_per_hypothesis": args.minimum_station_rows,
        "shrinkage_observations": args.shrinkage_observations,
        "eligible_partner_counts": {str(fold): len(profiles[fold]["eligible"]) for fold in range(5)},
        "shuffled_partner_maps": {str(fold): mapping for fold, mapping in partner_maps.items()},
        "methods": methods,
        "missing_partner_policy": "UNKNOWN; zero contribution",
        "station_policy": "latent mixture in R-LLR; filename station only in non-deployable R6",
        "outer_evaluation": "leave-one-case-out over five OOF validation cases",
        "test_data_used": False,
        "clinical_expert_complete": False,
        "clinical_warning_ready": False,
        "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": decision["status"],
        "macro_case_auprc": {method: float(row["auprc"]) for method, row in macro.items()},
        "comparisons": decision["comparisons"],
        "evidence": evidence[0],
    }, indent=2))


if __name__ == "__main__":
    main()
