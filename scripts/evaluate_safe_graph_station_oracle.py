#!/usr/bin/env python3
"""Evaluate fold-safe oracle-station support as a SAFE-Graph upper bound."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from scripts.train_safe_graph_warning_baselines import (  # noqa: E402
    binary_metrics,
    features,
    parse_bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-corpus", type=Path,
        default=ROOT / "artifacts/safe_graph_error_corpus/improved_v2_validation_error_v1_r1",
    )
    parser.add_argument(
        "--kg-root", type=Path,
        default=ROOT / "artifacts/safe_graph_knowledge/kg_v1_20260808",
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


def load_fold_support(kg_root: Path, fold: int) -> dict[tuple[str, int], float]:
    rows = read_csv(kg_root / f"outer_fold_{fold}_train/station_anatomy_support.csv")
    output = {
        (row["station"], int(row["anatomy_id"])): float(row["beta11_frame_probability"])
        for row in rows if row["context_basis"] == "visible_station"
    }
    if len(output) != 14 * 30:
        raise ValueError(f"Fold {fold} support must contain 14*30 non-background entries, got {len(output)}")
    return output


def oracle_station_support(row: dict[str, str], fold_support: dict[tuple[str, int], float]) -> float:
    stations = [station for station in row["visible_station_targets"].split("|") if station]
    if not stations:
        raise ValueError(f"Missing station target for {row['frame']}")
    class_id = int(row["predicted_class_id"])
    return max(fold_support[(station, class_id)] for station in stations)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    corpus_manifest = json.loads((args.error_corpus / "manifest.json").read_text())
    if corpus_manifest.get("test_data_used") is not False:
        raise ValueError("Station upper bound requires test-free error corpus")
    rows = read_csv(args.error_corpus / "predicted_component_errors.csv")
    fold_support = {fold: load_fold_support(args.kg_root, fold) for fold in range(5)}
    support = np.asarray([
        oracle_station_support(row, fold_support[int(row["fold"])]) for row in rows
    ])
    station_risk = 1.0 - support
    local = np.asarray([features(row) for row in rows], dtype=np.float64)
    combined = np.column_stack([local, station_risk])
    case_ids = np.asarray([row["case_id"] for row in rows])
    unique_cases = sorted(set(case_ids))
    targets = {
        "dangerous_error": np.asarray([parse_bool(row["dangerous_error_candidate"]) for row in rows], dtype=np.uint8),
        "any_component_error": np.asarray([row["error_label"] != "CORRECT_COMPONENT" for row in rows], dtype=np.uint8),
    }
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for target_name, y in targets.items():
        combined_risk = np.zeros(len(rows), dtype=np.float64)
        for held_case in unique_cases:
            train = case_ids != held_case
            test = ~train
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026),
            )
            model.fit(combined[train], y[train])
            combined_risk[test] = model.predict_proba(combined[test])[:, 1]
        for method, risk in (
            ("Q2_oracle_station_only", station_risk),
            ("Q2_local_plus_oracle_station", combined_risk),
        ):
            case_metrics = []
            for held_case in unique_cases:
                selected = case_ids == held_case
                result = binary_metrics(y[selected], risk[selected])
                case_metrics.append(result)
                metric_rows.append({
                    "target": target_name, "method": method, "scope": "per_case",
                    "case_id": held_case, **result,
                })
            pooled = binary_metrics(y, risk)
            metric_rows.append({
                "target": target_name, "method": method, "scope": "pooled_components",
                "case_id": "ALL", **pooled,
            })
            metric_rows.append({
                "target": target_name, "method": method, "scope": "macro_case_mean",
                "case_id": "ALL", "components": len(rows), "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "auprc": float(np.mean([item["auprc"] for item in case_metrics if item["auprc"] is not None])),
                "auroc": float(np.mean([item["auroc"] for item in case_metrics if item["auroc"] is not None])),
                "brier": float(np.mean([item["brier"] for item in case_metrics])),
            })
        for index, row in enumerate(rows):
            prediction_rows.append({
                "target": target_name,
                "fold": row["fold"],
                "case_id": row["case_id"],
                "frame": row["frame"],
                "component_id": row["component_id"],
                "true_label": int(y[index]),
                "oracle_station_support": support[index],
                "q2_oracle_station_risk": station_risk[index],
                "q2_combined_risk": combined_risk[index],
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "metrics.csv", metric_rows)
    write_csv(args.output_root / "oof_risk_predictions.csv", prediction_rows)
    primary = [
        row for row in metric_rows
        if row["target"] == "dangerous_error" and row["scope"] == "macro_case_mean"
    ]
    manifest = {
        "artifact": "SAFE-Graph Q2 oracle-station upper bound",
        "version": "station_oracle_upper_bound_v1",
        "source_error_corpus": str(args.error_corpus),
        "source_kg": str(args.kg_root),
        "source_split": corpus_manifest["source_split"],
        "test_data_used": False,
        "station_source": "ground-truth visible station labels (oracle upper bound)",
        "fold_safety": "Each validation case uses station-anatomy support from its corresponding 7-case outer-fold training atlas.",
        "station_mixture": "maximum Beta(1,1)-smoothed support over all visible stations",
        "primary_macro_case_results": primary,
        "deployment_status": "NOT_DEPLOYABLE; predicted station probabilities are not available from Improved v2",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
