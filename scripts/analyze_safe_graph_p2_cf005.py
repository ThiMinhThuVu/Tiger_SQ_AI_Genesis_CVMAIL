#!/usr/bin/env python3
"""Deep-dive the CF005 P2 confusion signal without touching the held-out test set."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


METHODS = {
    "P0_confidence_raw": {"kind": "raw", "features": ["local_confidence_risk"]},
    "D0_direction_only": {"kind": "learned", "features": ["pair_class_indicator"]},
    "P1_local_plus_pair_class": {
        "kind": "learned",
        "features": [
            "local_confidence_risk", "local_top1_risk", "local_margin_risk",
            "local_entropy", "local_log_area", "pair_class_indicator",
        ],
    },
    "P2a_counterpart_raw": {"kind": "raw", "features": ["counterpart_probability"]},
    "P2b_confusion_raw": {"kind": "raw", "features": ["fold_excluded_confusion_score"]},
    "P2c_counterpart_lr": {"kind": "learned", "features": ["counterpart_probability"]},
    "P2d_confusion_lr": {"kind": "learned", "features": ["fold_excluded_confusion_score"]},
    "P2e_counterpart_plus_confusion_lr": {
        "kind": "learned",
        "features": ["counterpart_probability", "fold_excluded_confusion_score"],
    },
    "P2f_counterpart_plus_direction_lr": {
        "kind": "learned",
        "features": ["counterpart_probability", "pair_class_indicator"],
    },
    "P2g_pair_share_plus_direction_lr": {
        "kind": "learned",
        "features": ["pair_probability_share", "pair_class_indicator"],
    },
    "P2h_counterpart_current_direction_lr": {
        "kind": "learned",
        "features": [
            "counterpart_probability", "current_class_probability", "pair_class_indicator",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/pair_specific_cf005_v1_r1/pair_cohort.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
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


def parse_bool(value: str | bool) -> bool:
    return value is True or str(value).strip().lower() == "true"


def enrich_rows(rows: list[dict[str, str]]) -> list[dict]:
    output = []
    for row in rows:
        current = float(1.0 - float(row["local_confidence_risk"]))
        counterpart = float(row["counterpart_probability"])
        denominator = current + counterpart
        output.append({
            **row,
            "current_class_probability": current,
            "pair_probability_share": counterpart / denominator if denominator > 0 else 0.0,
            "counterpart_available": counterpart > 0,
            "direction": (
                "pred_fatty_esophagus_gt_lymph_node"
                if int(row["predicted_class_id"]) == 7
                else "pred_lymph_node_gt_fatty_esophagus"
            ),
        })
    return output


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def macro_metrics(y: np.ndarray, risks: dict[str, np.ndarray], case_ids: np.ndarray):
    rows = []
    unique_cases = sorted(set(case_ids))
    for method, risk in risks.items():
        per_case = []
        for case_id in unique_cases:
            selected = case_ids == case_id
            result = binary_metrics(y[selected], risk[selected])
            per_case.append(result)
            rows.append({"method": method, "scope": "per_case", "case_id": case_id, **result})
        rows.append({
            "method": method,
            "scope": "pooled_components",
            "case_id": "ALL",
            **binary_metrics(y, risk),
        })
        rows.append({
            "method": method,
            "scope": "macro_case_mean",
            "case_id": "ALL",
            "components": len(y),
            "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "auprc": float(np.mean([item["auprc"] for item in per_case if item["auprc"] is not None])),
            "auroc": float(np.mean([item["auroc"] for item in per_case if item["auroc"] is not None])),
            "brier": float(np.mean([item["brier"] for item in per_case])),
        })
    return rows


def evaluate(rows: list[dict], coverages: list[float], seed: int):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["pair_swap_target"]) for row in rows], dtype=np.uint8)
    risks: dict[str, np.ndarray] = {}
    thresholds: dict[tuple[str, str, float], float] = {}
    coefficient_rows = []

    for method, specification in METHODS.items():
        names = specification["features"]
        x = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
        risk = np.zeros(len(rows), dtype=float)
        for held_case in cases:
            train = case_ids != held_case
            test = ~train
            if specification["kind"] == "raw":
                train_risk = x[train, 0]
                risk[test] = x[test, 0]
            else:
                model = make_model(seed)
                model.fit(x[train], y[train])
                train_risk = model.predict_proba(x[train])[:, 1]
                risk[test] = model.predict_proba(x[test])[:, 1]
                classifier = model.named_steps["logisticregression"]
                for feature, coefficient in zip(names, classifier.coef_[0]):
                    coefficient_rows.append({
                        "method": method,
                        "held_case": held_case,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                        "intercept": float(classifier.intercept_[0]),
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
                    "method": method,
                    "target_coverage": coverage,
                    "case_id": held_case,
                    **threshold_metrics(
                        y[selected], risk[selected], thresholds[(method, held_case, coverage)]
                    ),
                })
        for index, row in enumerate(rows):
            predictions.append({
                "method": method,
                "fold": row["fold"],
                "case_id": row["case_id"],
                "frame": row["frame"],
                "component_id": row["component_id"],
                "predicted_class_id": row["predicted_class_id"],
                "direction": row["direction"],
                "area_bin": row["area_bin"],
                "counterpart_available": row["counterpart_available"],
                "true_label": int(y[index]),
                "risk": float(risk[index]),
            })
    return y, case_ids, risks, metrics, operational, coefficient_rows, predictions


def summarize_operational(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["method"], float(row["target_coverage"])) for row in rows})
    for method, coverage in keys:
        selected = [
            row for row in rows
            if row["method"] == method and float(row["target_coverage"]) == coverage
        ]
        ppv = [row["warning_ppv"] for row in selected if row["warning_ppv"] is not None]
        recall = [row["warning_recall"] for row in selected if row["warning_recall"] is not None]
        output.append({
            "method": method,
            "target_coverage": coverage,
            "cases": len(selected),
            "mean_actual_coverage": float(np.mean([row["warning_coverage"] for row in selected])),
            "mean_ppv": float(np.mean(ppv)) if ppv else None,
            "mean_recall": float(np.mean(recall)) if recall else None,
            "min_case_ppv": float(np.min(ppv)) if ppv else None,
            "cases_ppv_at_least_0_5": sum(
                row["warning_ppv"] is not None and row["warning_ppv"] >= 0.5
                for row in selected
            ),
            "cases_zero_true_positive_warning": sum(
                row["warning_count"] > 0 and row["warning_ppv"] == 0 for row in selected
            ),
        })
    return output


def grouped_metrics(rows: list[dict], y: np.ndarray, risks: dict[str, np.ndarray]) -> list[dict]:
    case_ids = np.asarray([row["case_id"] for row in rows])
    groups = {
        "direction_pred_7": np.asarray([int(row["predicted_class_id"]) == 7 for row in rows]),
        "direction_pred_19": np.asarray([int(row["predicted_class_id"]) == 19 for row in rows]),
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in rows]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in rows]),
        "counterpart_available": np.asarray([row["counterpart_available"] for row in rows]),
        "counterpart_unavailable": np.asarray([not row["counterpart_available"] for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        for method, risk in risks.items():
            pooled = binary_metrics(y[selected], risk[selected])
            case_auprc = []
            for case_id in sorted(set(case_ids[selected])):
                subset = selected & (case_ids == case_id)
                metric = binary_metrics(y[subset], risk[subset])
                if metric["auprc"] is not None:
                    case_auprc.append(metric["auprc"])
            output.append({
                "group": group,
                "method": method,
                **pooled,
                "evaluable_cases": len(case_auprc),
                "macro_case_auprc": float(np.mean(case_auprc)) if case_auprc else None,
            })
    return output


def availability_summary(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {"all": rows}
    for case_id in sorted({row["case_id"] for row in rows}):
        groups[f"case:{case_id}"] = [row for row in rows if row["case_id"] == case_id]
    groups["direction:pred_7"] = [row for row in rows if int(row["predicted_class_id"]) == 7]
    groups["direction:pred_19"] = [row for row in rows if int(row["predicted_class_id"]) == 19]
    groups["area:tiny"] = [row for row in rows if row["area_bin"] == "tiny"]
    groups["area:non_tiny"] = [row for row in rows if row["area_bin"] != "tiny"]
    output = []
    for group, selected in groups.items():
        positives = [row for row in selected if parse_bool(row["pair_swap_target"])]
        available = [row for row in selected if row["counterpart_available"]]
        positive_available = [row for row in positives if row["counterpart_available"]]
        output.append({
            "group": group,
            "components": len(selected),
            "positives": len(positives),
            "prevalence": len(positives) / len(selected) if selected else None,
            "counterpart_available": len(available),
            "counterpart_availability": len(available) / len(selected) if selected else None,
            "positive_counterpart_available": len(positive_available),
            "positive_availability_recall_ceiling": (
                len(positive_available) / len(positives) if positives else None
            ),
        })
    return output


def confusion_scaling_diagnostic(rows: list[dict]) -> list[dict]:
    output = []
    for fold in sorted({row["fold"] for row in rows}):
        selected = [row for row in rows if row["fold"] == fold]
        nonzero = [row for row in selected if float(row["counterpart_probability"]) > 0]
        ratios = np.asarray([
            float(row["fold_excluded_confusion_score"]) / float(row["counterpart_probability"])
            for row in nonzero
        ])
        counterpart = np.asarray([float(row["counterpart_probability"]) for row in selected])
        confusion = np.asarray([float(row["fold_excluded_confusion_score"]) for row in selected])
        output.append({
            "fold": fold,
            "components": len(selected),
            "nonzero_counterpart": len(nonzero),
            "confusion_to_counterpart_ratio_min": float(ratios.min()) if len(ratios) else None,
            "confusion_to_counterpart_ratio_max": float(ratios.max()) if len(ratios) else None,
            "pearson_correlation": float(np.corrcoef(counterpart, confusion)[0, 1]),
        })
    return output


def score_bins(rows: list[dict]) -> list[dict]:
    scores = np.asarray([float(row["counterpart_probability"]) for row in rows])
    nonzero = scores[scores > 0]
    q25, q50, q75 = np.quantile(nonzero, [0.25, 0.50, 0.75])
    bins = [
        ("zero", lambda value: value == 0),
        ("nonzero_q1", lambda value: 0 < value <= q25),
        ("nonzero_q2", lambda value: q25 < value <= q50),
        ("nonzero_q3", lambda value: q50 < value <= q75),
        ("nonzero_q4", lambda value: value > q75),
    ]
    output = []
    for name, predicate in bins:
        selected = [row for row in rows if predicate(float(row["counterpart_probability"]))]
        positives = sum(parse_bool(row["pair_swap_target"]) for row in selected)
        output.append({
            "bin": name,
            "lower_context_q25": float(q25),
            "median_context_q50": float(q50),
            "upper_context_q75": float(q75),
            "components": len(selected),
            "positives": positives,
            "observed_error_rate": positives / len(selected) if selected else None,
            "predicted_7_components": sum(int(row["predicted_class_id"]) == 7 for row in selected),
            "predicted_19_components": sum(int(row["predicted_class_id"]) == 19 for row in selected),
        })
    return output


def bootstrap_deltas(
    metrics: list[dict], methods: list[str], reference: str, replicates: int, seed: int,
) -> list[dict]:
    per_case = defaultdict(dict)
    for row in metrics:
        if row["scope"] == "per_case" and row["auprc"] is not None:
            per_case[row["case_id"]][row["method"]] = float(row["auprc"])
    rng = np.random.default_rng(seed)
    output = []
    for method in methods:
        deltas = np.asarray([
            values[method] - values[reference]
            for values in per_case.values() if method in values and reference in values
        ])
        indices = rng.integers(0, len(deltas), size=(replicates, len(deltas)))
        bootstrap = deltas[indices].mean(axis=1)
        output.append({
            "method": method,
            "reference": reference,
            "cases": len(deltas),
            "mean_case_delta_auprc": float(deltas.mean()),
            "cases_improved": int((deltas > 0).sum()),
            "case_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "case_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_probability_delta_positive": float((bootstrap > 0).mean()),
            "inference_scope": "descriptive_exploratory_selection_leaky",
        })
    return output


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    rows = enrich_rows(read_csv(args.cohort))
    if len(rows) != 744 or sum(parse_bool(row["pair_swap_target"]) for row in rows) != 99:
        raise ValueError("Expected frozen CF005 cohort with 744 components and 99 positives")
    coverages = [0.05, 0.10, 0.20, 0.30]
    y, case_ids, risks, metrics, operational, coefficients, predictions = evaluate(
        rows, coverages, args.seed
    )
    operational_summary = summarize_operational(operational)
    grouped = grouped_metrics(rows, y, risks)
    availability = availability_summary(rows)
    scaling = confusion_scaling_diagnostic(rows)
    bins = score_bins(rows)
    bootstrap = bootstrap_deltas(
        metrics,
        [
            "P2a_counterpart_raw", "P2e_counterpart_plus_confusion_lr",
            "P2f_counterpart_plus_direction_lr", "P2g_pair_share_plus_direction_lr",
            "P2h_counterpart_current_direction_lr",
        ],
        "P1_local_plus_pair_class",
        args.bootstrap_replicates,
        args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metrics.csv", metrics)
    write_csv(args.output_root / "operational_metrics.csv", operational)
    write_csv(args.output_root / "operational_summary.csv", operational_summary)
    write_csv(args.output_root / "grouped_metrics.csv", grouped)
    write_csv(args.output_root / "counterpart_availability.csv", availability)
    write_csv(args.output_root / "confusion_scaling_diagnostic.csv", scaling)
    write_csv(args.output_root / "counterpart_score_bins.csv", bins)
    write_csv(args.output_root / "coefficient_stability.csv", coefficients)
    write_csv(args.output_root / "case_bootstrap_deltas.csv", bootstrap)
    write_csv(args.output_root / "oof_risk_predictions.csv", predictions)

    macro = {
        row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"
    }
    p2_reproduced = abs(
        float(macro["P2e_counterpart_plus_confusion_lr"]["auprc"]) - 0.34778401141458154
    ) < 1e-12
    best = max(macro.values(), key=lambda row: float(row["auprc"]))
    manifest = {
        "artifact": "CF005 P2 confusion-signal deep dive",
        "version": args.output_root.name,
        "status": "EXPLORATORY_ANALYZED_NOT_CLINICALLY_READY",
        "source_cohort": str(args.cohort),
        "endpoint": "Exact class-7/class-19 predicted-component swap",
        "evaluation": "leave-one-case-out across five OOF validation cases",
        "methods": METHODS,
        "warning_target_coverages": coverages,
        "bootstrap_replicates": args.bootstrap_replicates,
        "test_data_used": False,
        "selection_leakage": "CF005 selected from the complete OOF aggregate; all inference is exploratory",
        "p2_original_reproduced": p2_reproduced,
        "best_macro_case_method": best,
        "limitations": [
            "Only five independent cases are available",
            "CF005 pair selection used the complete OOF aggregate",
            "Counterpart probability is truncated by the exported alternative-class list",
            "Balanced logistic scores are not calibrated clinical probabilities",
            "Component targets are algorithmic and lack clinician adjudication",
        ],
        "clinical_warning_ready": False,
        "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "p2_original_reproduced": p2_reproduced,
        "macro_case_results": sorted(
            ({"method": key, "auprc": value["auprc"], "auroc": value["auroc"], "brier": value["brier"]}
             for key, value in macro.items()),
            key=lambda item: item["auprc"], reverse=True,
        ),
        "best_method": best["method"],
    }, indent=2))


if __name__ == "__main__":
    main()

