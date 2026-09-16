#!/usr/bin/env python3
"""Evaluate dense class-7/class-19 probabilities for the CF005 verifier."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.analyze_safe_graph_p2_cf005 import (  # noqa: E402
    bootstrap_deltas,
    macro_metrics,
    make_model,
    parse_bool,
    summarize_operational,
)
from scripts.evaluate_safe_graph_pair_specific import threshold_metrics  # noqa: E402
from scripts.train_safe_graph_warning_baselines import binary_metrics  # noqa: E402


METHODS = {
    "P1_local_plus_pair_class": {
        "kind": "learned",
        "features": [
            "local_confidence_risk", "local_top1_risk", "local_margin_risk",
            "local_entropy", "local_log_area", "pair_class_indicator",
        ],
    },
    "P2_topk_counterpart_raw": {
        "kind": "raw", "features": ["topk_counterpart_probability"],
    },
    "P2_dense_counterpart_raw": {
        "kind": "raw", "features": ["dense_counterpart_probability"],
    },
    "P2_dense_pair_share_raw": {
        "kind": "raw", "features": ["dense_pair_probability_share"],
    },
    "P2_dense_counterpart_lr": {
        "kind": "learned", "features": ["dense_counterpart_probability"],
    },
    "P2_dense_counterpart_plus_direction_lr": {
        "kind": "learned",
        "features": ["dense_counterpart_probability", "pair_class_indicator"],
    },
    "P2_dense_counterpart_current_direction_lr": {
        "kind": "learned",
        "features": [
            "dense_counterpart_probability", "dense_current_probability",
            "pair_class_indicator",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-cohort", type=Path,
        default=ROOT / "artifacts/safe_graph_verifier/pair_specific_cf005_v1_r1/pair_cohort.csv",
    )
    parser.add_argument(
        "--dense-oof-root", type=Path,
        default=ROOT / "artifacts/safe_graph_oof/improved_v2_validation_p2_dense_cf005_v1",
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


def explicit_probability_lookup(row: dict[str, str]) -> dict[int, float]:
    ids = [int(value) for value in row["explicit_probability_class_ids"].split("|") if value]
    probabilities = [
        float(value) for value in row["explicit_class_mean_probabilities"].split("|") if value
    ]
    if len(ids) != len(probabilities):
        raise ValueError("Explicit class/probability length mismatch")
    result = dict(zip(ids, probabilities))
    if set(result) != {7, 19}:
        raise ValueError(f"Expected explicit classes 7 and 19, got {sorted(result)}")
    return result


def load_dense_rows(root: Path) -> tuple[list[dict[str, str]], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("split") != "out_of_fold_validation_only":
        raise ValueError("Dense P2 export must be OOF validation only")
    if manifest.get("test_data_used") is not False:
        raise ValueError("Dense P2 export must not use test data")
    if manifest.get("explicit_probability_class_ids") != [7, 19]:
        raise ValueError("Dense P2 export must contain exactly class 7 and 19 probabilities")
    rows = []
    for fold in range(5):
        rows.extend(read_csv(root / f"fold_{fold}/predicted_components_raw.csv"))
    return rows, manifest


def merge_dense_cohort(pair_rows: list[dict[str, str]], dense_rows: list[dict[str, str]]) -> list[dict]:
    lookup = {
        (row["fold"], row["case_id"], row["frame"], row["component_id"]): row
        for row in dense_rows
    }
    output = []
    for row in pair_rows:
        key = (row["fold"], row["case_id"], row["frame"], row["component_id"])
        if key not in lookup:
            raise ValueError(f"Dense export missing component {key}")
        dense = lookup[key]
        if dense["predicted_class_id"] != row["predicted_class_id"]:
            raise ValueError(f"Prediction mismatch for component {key}")
        probability = explicit_probability_lookup(dense)
        predicted = int(row["predicted_class_id"])
        counterpart = 19 if predicted == 7 else 7
        current = probability[predicted]
        alternative = probability[counterpart]
        denominator = current + alternative
        output.append({
            **row,
            "topk_counterpart_probability": float(row["counterpart_probability"]),
            "dense_class_7_probability": probability[7],
            "dense_class_19_probability": probability[19],
            "dense_current_probability": current,
            "dense_counterpart_probability": alternative,
            "dense_pair_probability_share": alternative / denominator if denominator > 0 else 0.0,
            "topk_counterpart_available": float(row["counterpart_probability"]) > 0,
            "recovered_from_topk_zero": float(row["counterpart_probability"]) == 0,
        })
    if len(output) != len(pair_rows):
        raise ValueError("Dense cohort row count changed")
    return output


def evaluate(rows: list[dict], coverages: list[float], seed: int):
    case_ids = np.asarray([row["case_id"] for row in rows])
    cases = sorted(set(case_ids))
    y = np.asarray([parse_bool(row["pair_swap_target"]) for row in rows], dtype=np.uint8)
    risks = {}
    thresholds = {}
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
                "area_bin": row["area_bin"],
                "topk_counterpart_available": row["topk_counterpart_available"],
                "true_label": int(y[index]),
                "risk": float(risk[index]),
            })
    return y, case_ids, risks, metrics, operational, predictions


def subgroup_metrics(rows: list[dict], y: np.ndarray, risks: dict[str, np.ndarray]) -> list[dict]:
    groups = {
        "all": np.ones(len(rows), dtype=bool),
        "topk_available": np.asarray([row["topk_counterpart_available"] for row in rows]),
        "topk_missing": np.asarray([not row["topk_counterpart_available"] for row in rows]),
        "direction_pred_7": np.asarray([int(row["predicted_class_id"]) == 7 for row in rows]),
        "direction_pred_19": np.asarray([int(row["predicted_class_id"]) == 19 for row in rows]),
        "area_tiny": np.asarray([row["area_bin"] == "tiny" for row in rows]),
        "area_non_tiny": np.asarray([row["area_bin"] != "tiny" for row in rows]),
    }
    output = []
    for group, selected in groups.items():
        for method, risk in risks.items():
            output.append({"group": group, "method": method, **binary_metrics(y[selected], risk[selected])})
    return output


def recovery_summary(rows: list[dict]) -> list[dict]:
    groups = {
        "all": rows,
        "direction_pred_7": [row for row in rows if int(row["predicted_class_id"]) == 7],
        "direction_pred_19": [row for row in rows if int(row["predicted_class_id"]) == 19],
        "area_tiny": [row for row in rows if row["area_bin"] == "tiny"],
        "area_non_tiny": [row for row in rows if row["area_bin"] != "tiny"],
    }
    output = []
    for group, selected in groups.items():
        positives = [row for row in selected if parse_bool(row["pair_swap_target"])]
        missing = [row for row in selected if not row["topk_counterpart_available"]]
        missing_positive = [row for row in missing if parse_bool(row["pair_swap_target"])]
        output.append({
            "group": group,
            "components": len(selected),
            "positives": len(positives),
            "topk_missing_components": len(missing),
            "topk_missing_positives": len(missing_positive),
            "dense_nonzero_components": sum(float(row["dense_counterpart_probability"]) > 0 for row in selected),
            "dense_nonzero_positives": sum(
                parse_bool(row["pair_swap_target"]) and float(row["dense_counterpart_probability"]) > 0
                for row in selected
            ),
            "median_recovered_probability": (
                float(np.median([float(row["dense_counterpart_probability"]) for row in missing]))
                if missing else None
            ),
            "median_recovered_positive_probability": (
                float(np.median([
                    float(row["dense_counterpart_probability"]) for row in missing_positive
                ])) if missing_positive else None
            ),
        })
    return output


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    pair_rows = read_csv(args.pair_cohort)
    dense_rows, dense_manifest = load_dense_rows(args.dense_oof_root)
    rows = merge_dense_cohort(pair_rows, dense_rows)
    coverages = [0.05, 0.10, 0.20, 0.30]
    y, case_ids, risks, metrics, operational, predictions = evaluate(rows, coverages, args.seed)
    operational_summary = summarize_operational(operational)
    subgroups = subgroup_metrics(rows, y, risks)
    recovery = recovery_summary(rows)
    bootstrap = bootstrap_deltas(
        metrics,
        [
            "P2_dense_counterpart_raw", "P2_dense_pair_share_raw",
            "P2_dense_counterpart_plus_direction_lr",
            "P2_dense_counterpart_current_direction_lr",
        ],
        "P2_topk_counterpart_raw",
        args.bootstrap_replicates,
        args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "dense_pair_cohort.csv", rows)
    write_csv(args.output_root / "metrics.csv", metrics)
    write_csv(args.output_root / "operational_metrics.csv", operational)
    write_csv(args.output_root / "operational_summary.csv", operational_summary)
    write_csv(args.output_root / "subgroup_metrics.csv", subgroups)
    write_csv(args.output_root / "recovery_summary.csv", recovery)
    write_csv(args.output_root / "case_bootstrap_deltas.csv", bootstrap)
    write_csv(args.output_root / "oof_risk_predictions.csv", predictions)
    macro = {row["method"]: row for row in metrics if row["scope"] == "macro_case_mean"}
    topk = float(macro["P2_topk_counterpart_raw"]["auprc"])
    dense = float(macro["P2_dense_counterpart_raw"]["auprc"])
    per_case = {
        row["case_id"]: {} for row in metrics if row["scope"] == "per_case"
    }
    for row in metrics:
        if row["scope"] == "per_case":
            per_case[row["case_id"]][row["method"]] = row["auprc"]
    improved = sum(
        values["P2_dense_counterpart_raw"] > values["P2_topk_counterpart_raw"]
        for values in per_case.values()
    )
    status = (
        "DENSE_P2_EXPLORATORY_GO"
        if dense - topk >= 0.02 and improved >= 3
        else "DENSE_P2_NO_INCREMENTAL_GO"
    )
    manifest = {
        "artifact": "CF005 dense counterpart-probability verifier",
        "version": args.output_root.name,
        "status": status,
        "source_pair_cohort": str(args.pair_cohort),
        "source_dense_oof": str(args.dense_oof_root),
        "source_dense_oof_manifest": dense_manifest,
        "evaluation": "leave-one-case-out across five OOF validation cases",
        "endpoint": "Exact class-7/class-19 predicted-component swap",
        "methods": METHODS,
        "topk_macro_case_auprc": topk,
        "dense_macro_case_auprc": dense,
        "dense_delta_auprc": dense - topk,
        "cases_improved": improved,
        "test_data_used": False,
        "selection_leakage": "CF005 was selected from the complete OOF aggregate; exploratory only",
        "clinical_warning_ready": False,
        "correction_authorized": False,
        "ontology_loss_authorized": False,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "topk_macro_case_auprc": topk,
        "dense_macro_case_auprc": dense,
        "dense_delta_auprc": dense - topk,
        "cases_improved": improved,
        "recovery": recovery[0],
    }, indent=2))


if __name__ == "__main__":
    main()

