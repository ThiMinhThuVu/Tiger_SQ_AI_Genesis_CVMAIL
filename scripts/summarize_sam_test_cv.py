#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd",
    "fine_pixel_accuracy", "coarse_pixel_accuracy", "visibility_accuracy",
    "visibility_macro_f1", "visibility_macro_auroc", "selection_score",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for fold in range(5):
        fold_dir = args.experiment_dir / f"fold_{fold}"
        metrics = json.loads((fold_dir / "quantitative_metrics.json").read_text())
        early = json.loads((fold_dir / "early_stopping.json").read_text())
        rows.append({
            "fold": fold, "epochs_completed": early["completed_epochs"],
            "stopped_early": early["stopped_early"],
            **{name: metrics[name] for name in METRICS},
        })
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "sample_standard_deviation": float(np.std([row[name] for row in rows], ddof=1)),
        }
        for name in METRICS
    }
    result = {"experiment_dir": str(args.experiment_dir), "folds": rows, "summary": summary}
    (args.experiment_dir / "test_cv_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.experiment_dir / "test_fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Five-fold test summary", "",
        "Protocol: 7 train / 1 validation / 2 test cases, maximum 100 epochs with early stopping.", "",
        "| Metric | Mean | Sample SD |", "|---|---:|---:|",
    ]
    for name in METRICS:
        lines.append(f"| {name} | {summary[name]['mean']:.4f} | {summary[name]['sample_standard_deviation']:.4f} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
