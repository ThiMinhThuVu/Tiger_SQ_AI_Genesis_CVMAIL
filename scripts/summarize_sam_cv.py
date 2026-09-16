#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiger_models.data import STATIONS, case_from_name
from tiger_models.metrics import selection_score, visibility_metrics
from tiger_models.runtime import atomic_json


METRICS = (
    "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd",
    "visibility_macro_f1", "visibility_macro_auroc", "selection_score",
)


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "sample_standard_deviation": float(array.std(ddof=1)),
        "median": float(np.median(array)), "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "artifacts/new_baselines/sam2_hiera_tiny_partial")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/SAM_BASELINE_RESULTS.md")
    args = parser.parse_args()
    fold_rows = []
    case_segmentation = {}
    oof_visibility = {}
    class_rows_by_key = {}
    station_rows_by_name = {}
    for fold in range(5):
        directory = args.experiment_dir / f"fold_{fold}"
        if not (directory / "completion.json").is_file():
            raise FileNotFoundError(f"Fold {fold} is not complete")
        metrics = json.loads((directory / "best_metrics.json").read_text())
        resource = json.loads((directory / "resource_profile.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        row = {"fold": fold, **{key: metrics[key] for key in METRICS},
               "best_epoch": metrics["best_epoch"],
               "peak_reserved_vram_gb": resource["peak_reserved_vram_gb"],
               "peak_allocated_vram_gb": resource["peak_allocated_vram_gb"],
               "train_seconds_per_epoch": resource["train_seconds_per_epoch"],
               "inference_ms_per_image": resource["inference_ms_per_image"],
               "trainable_parameters": config["parameter_counts"]["trainable"],
               "total_parameters": config["parameter_counts"]["total"]}
        fold_rows.append(row)
        with (directory / "per_case_metrics.csv").open(newline="") as handle:
            for case_row in csv.DictReader(handle):
                case_segmentation[case_row["case_id"]] = {
                    key: float(case_row[key]) for key in
                    ("fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd")
                }
        with (directory / "per_class_metrics.csv").open(newline="") as handle:
            for class_row in csv.DictReader(handle):
                key = (class_row["task"], int(class_row["class_id"]))
                class_rows_by_key.setdefault(key, []).append(class_row)
        with (directory / "per_station_metrics.csv").open(newline="") as handle:
            for station_row in csv.DictReader(handle):
                station_rows_by_name.setdefault(station_row["station"], []).append(station_row)
        probabilities = np.load(directory / "validation_probabilities.npz")
        for name, target, probability in zip(
            probabilities["names"], probabilities["visibility_target"],
            probabilities["visibility_probability"],
        ):
            oof_visibility[str(name)] = (target, probability)
    if len(case_segmentation) != 10 or len(oof_visibility) != 140:
        raise RuntimeError("OOF aggregation does not cover exactly 10 cases and 140 frames")

    summary = {key: describe([float(row[key]) for row in fold_rows]) for key in METRICS}
    rng = np.random.default_rng(args.seed)
    cases = sorted(case_segmentation)
    visibility_by_case = {
        case: [name for name in sorted(oof_visibility) if case_from_name(name) == case]
        for case in cases
    }
    bootstrap = {key: [] for key in METRICS}
    for _ in range(args.bootstrap_samples):
        sampled_cases = rng.choice(cases, size=len(cases), replace=True)
        seg = {
            key: float(np.mean([case_segmentation[case][key] for case in sampled_cases]))
            for key in ("fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd")
        }
        names = [name for case in sampled_cases for name in visibility_by_case[case]]
        target = np.stack([oof_visibility[name][0] for name in names])
        probability = np.stack([oof_visibility[name][1] for name in names])
        visibility = visibility_metrics(target, probability, STATIONS)
        values = {
            **seg,
            "visibility_macro_f1": visibility["macro_f1"],
            "visibility_macro_auroc": visibility["macro_auroc"],
        }
        values["selection_score"] = selection_score(
            values["fine_dice"], values["fine_nhd"], values["coarse_dice"],
            values["coarse_nhd"], values["visibility_macro_f1"],
            values["visibility_macro_auroc"],
        )
        for key in METRICS:
            if values[key] is not None:
                bootstrap[key].append(float(values[key]))
    for key in METRICS:
        summary[key]["bootstrap_95_percentile_ci"] = [
            float(np.percentile(bootstrap[key], 2.5)), float(np.percentile(bootstrap[key], 97.5))
        ]
    result = {
        "experiment": "sam2_hiera_tiny_partial", "completed_folds": [0, 1, 2, 3, 4],
        "fold_metrics": fold_rows, "summary": summary,
        "bootstrap": {
            "resampling_unit": "case", "samples": args.bootstrap_samples,
            "seed": args.seed, "confidence_interval_method": "percentile",
        },
    }
    atomic_json(args.experiment_dir / "aggregated_out_of_fold_metrics.json", result)
    with (args.experiment_dir / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader(); writer.writerows(fold_rows)
    with (args.experiment_dir / "per_case_metrics.csv").open("w", newline="") as handle:
        rows = [{"case_id": case, **case_segmentation[case]} for case in sorted(case_segmentation)]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    combined_classes = []
    for (task, class_id), rows in sorted(class_rows_by_key.items()):
        observable_rows = [row for row in rows if row["observable"].lower() == "true"]
        combined_classes.append({
            "task": task, "class_id": class_id, "class_name": rows[0]["class_name"],
            "observable_folds": len(observable_rows),
            "dice_mean_over_observable_folds": (
                float(np.mean([float(row["dice"]) for row in observable_rows])) if observable_rows else "not observable"
            ),
            "nhd_mean_over_observable_folds": (
                float(np.mean([float(row["nhd"]) for row in observable_rows])) if observable_rows else "not observable"
            ),
        })
    with (args.experiment_dir / "per_class_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_classes[0])); writer.writeheader(); writer.writerows(combined_classes)
    combined_stations = []
    for station in STATIONS:
        rows = station_rows_by_name[station]
        combined_stations.append({
            "station": station,
            **{metric: float(np.mean([float(row[metric]) for row in rows if row[metric] not in {"", "None"}]))
               for metric in ("f1", "precision", "recall", "auroc")},
        })
    with (args.experiment_dir / "per_station_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_stations[0])); writer.writeheader(); writer.writerows(combined_stations)
    lines = [
        "# SAM baseline results", "", "## Completed primary benchmark", "",
        "`sam2_hiera_tiny_partial`: five case-level folds × 80 epochs, 512×896, seed 2026.", "",
        "| Metric | Mean | Sample SD | Median | Min | Max | Case-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in METRICS:
        item = summary[key]; ci = item["bootstrap_95_percentile_ci"]
        lines.append(
            f"| {key} | {item['mean']:.4f} | {item['sample_standard_deviation']:.4f} | "
            f"{item['median']:.4f} | {item['minimum']:.4f} | {item['maximum']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines.extend(["", "## Fold details", "", "| Fold | Selection | Best epoch | Peak reserved GB | Inference ms/image |", "|---:|---:|---:|---:|---:|"])
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['selection_score']:.4f} | {row['best_epoch']} | "
            f"{row['peak_reserved_vram_gb']:.2f} | {row['inference_ms_per_image']:.2f} |"
        )
    lines.extend([
        "", "Bootstrap resamples 10 cases with replacement, 1,000 times, seed 2026, "
        "and reports percentile intervals. No ground-truth prompt is used.", "",
    ])
    args.report.write_text("\n".join(lines))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
