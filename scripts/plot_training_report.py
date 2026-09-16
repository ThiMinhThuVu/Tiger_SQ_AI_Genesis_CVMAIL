#!/usr/bin/env python3
"""Plot training curves and build a cross-model score report.

The training loop records epoch-level metrics per fold.  This script aggregates
those files without mixing folds: curves show mean +/- sample standard
deviation across the five folds, while the score table is computed from each
fold's saved best_metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "sam2_hiera_tiny": "SAM2 Hiera Tiny",
    "sam2_hiera_small": "SAM2 Hiera Small",
    "sam2_hiera_base_plus": "SAM2 Hiera Base+",
}
SCORE_METRICS = (
    "selection_score", "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd",
    "visibility_macro_f1", "visibility_macro_auroc",
)
TEST_METRICS = (
    "selection_score", "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd",
    "fine_pixel_accuracy", "coarse_pixel_accuracy", "visibility_accuracy",
    "visibility_macro_f1", "visibility_macro_auroc",
)
METRIC_LABELS = {
    "selection_score": "Selection score", "fine_dice": "Fine Dice", "fine_nhd": "Fine NHD",
    "coarse_dice": "Coarse Dice", "coarse_nhd": "Coarse NHD",
    "fine_pixel_accuracy": "Fine pixel accuracy", "coarse_pixel_accuracy": "Coarse pixel accuracy",
    "visibility_accuracy": "Visibility accuracy", "visibility_macro_f1": "Visibility macro F1",
    "visibility_macro_auroc": "Visibility macro AUROC",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path,
                        default=root / "artifacts/sam_7_1_2_boundary_100")
    parser.add_argument("--output", type=Path,
                        default=root / "reports/training_analysis")
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate_curves(model_dir: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(model_dir.glob("fold_*")):
        path = fold_dir / "epoch_metrics.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["fold"] = int(fold_dir.name.split("_")[-1])
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No epoch_metrics.csv found under {model_dir}")
    data = pd.concat(frames, ignore_index=True)
    numeric = [column for column in data.columns
               if column not in {"epoch", "fold", "learning_rates"}
               and pd.api.types.is_numeric_dtype(data[column])]
    grouped = data.groupby("epoch", as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = ["epoch"] + [f"{metric}_{stat}" for metric, stat in grouped.columns[1:]]
    return grouped


def plot_curves(curves: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = [
        ("train_loss", "val_loss", "Loss", "Loss (lower is better)"),
        (None, "selection_score", "Validation selection score", "Score (higher is better)"),
        ("train_fine_pixel_accuracy", "val_fine_pixel_accuracy", "Fine pixel accuracy", "Accuracy"),
        ("train_visibility_accuracy", "val_visibility_accuracy", "Visibility accuracy", "Accuracy"),
    ]
    colors = {name: color for name, color in zip(MODELS, ("#2563eb", "#dc2626", "#059669"))}
    for axis, (train_metric, val_metric, title, ylabel) in zip(axes.flat, panels):
        for model, frame in curves.items():
            color = colors[model]
            x = frame["epoch"] + 1
            if train_metric:
                axis.plot(x, frame[f"{train_metric}_mean"], color=color,
                          label=f"{MODELS[model]} train")
                axis.fill_between(x,
                                  frame[f"{train_metric}_mean"] - frame[f"{train_metric}_std"],
                                  frame[f"{train_metric}_mean"] + frame[f"{train_metric}_std"],
                                  color=color, alpha=0.10)
            axis.plot(x, frame[f"{val_metric}_mean"], color=color, linestyle="--",
                      label=f"{MODELS[model]} val")
            axis.fill_between(x,
                              frame[f"{val_metric}_mean"] - frame[f"{val_metric}_std"],
                              frame[f"{val_metric}_mean"] + frame[f"{val_metric}_std"],
                              color=color, alpha=0.06)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("TIGER-SQ-AI: training curves across 5 folds", fontsize=16)
    fig.savefig(output / "training_curves_3_models.png", dpi=180)
    plt.close(fig)


def load_test_summary(artifacts: Path, model: str) -> dict:
    path = artifacts / f"{model}_partial" / "test_cv_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test summary: {path}")
    return json.loads(path.read_text())


def build_score_table(artifacts: Path, output: Path) -> pd.DataFrame:
    rows = []
    for model, label in MODELS.items():
        model_dir = artifacts / f"{model}_partial"
        for fold_dir in sorted(model_dir.glob("fold_*")):
            metrics_path = fold_dir / "best_metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text())
            row = {"model": label, "model_id": model,
                   "fold": int(fold_dir.name.split("_")[-1]),
                   "best_epoch": metrics.get("best_epoch")}
            row.update({metric: metrics.get(metric) for metric in SCORE_METRICS})
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No best_metrics.json found for the requested models")
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(output / "score_by_fold.csv", index=False, float_format="%.6f")
    summary_rows = []
    for (model, model_id), group in fold_table.groupby(["model", "model_id"], sort=False):
        test = load_test_summary(artifacts, model_id)["summary"]
        row = {"model": model, "model_id": model_id, "folds": len(group),
               # The training loop only computes the official score during validation.
               # Keep this explicit instead of inventing a non-comparable train score.
               "train_selection_score_mean": np.nan,
               "train_selection_score_std": np.nan,
               "val_selection_score_mean": group["selection_score"].mean(),
               "val_selection_score_std": group["selection_score"].std(ddof=1),
               "test_selection_score_mean": test["selection_score"]["mean"],
               "test_selection_score_std": test["selection_score"]["sample_standard_deviation"]}
        for metric in TEST_METRICS:
            row[f"test_{metric}_mean"] = test[metric]["mean"]
            row[f"test_{metric}_std"] = test[metric]["sample_standard_deviation"]
        for metric in SCORE_METRICS:
            mean, std = mean_std(group[metric].dropna().tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        row["best_epoch_mean"] = group["best_epoch"].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("test_selection_score_mean", ascending=False)
    summary.to_csv(output / "score_summary.csv", index=False, float_format="%.6f")
    return summary


def write_report(summary: pd.DataFrame, curves: dict[str, pd.DataFrame], output: Path) -> None:
    winner = summary.iloc[0]
    lines = [
        "# Training curves and score comparison",
        "",
        "## Scope",
        "",
        "- Three SAM2 models, partial fine-tuning, 5 folds per model.",
        "- Curves show fold mean ± sample standard deviation by epoch.",
        "- The official `selection_score` is a validation metric; the training loop does not define a train selection score.",
        "- `score_by_fold.csv` keeps the individual fold values for auditability.",
        "",
        f"## Result: {winner['model']} ranks first",
        "",
        f"Test selection score = **{winner['test_selection_score_mean']:.4f} ± {winner['test_selection_score_std']:.4f}** "
        f"over {int(winner['folds'])} folds; mean best epoch = {winner['best_epoch_mean']:.1f}.",
        "",
        "## Validation metrics",
        "",
        "| Model | Folds | Selection score | Fine Dice | Fine NHD | Coarse Dice | Coarse NHD | Visibility F1 | Visibility AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    val_order = summary["val_selection_score_mean"].rank(method="min", ascending=False).astype(int)
    for index, (_, row) in enumerate(summary.iterrows()):
        def fmt(metric: str, rank: int | None = None, prefix: str = "") -> str:
            value = f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            if rank == 1:
                return f"**{value}**"
            if rank == 2:
                return f"<u>{value}</u>"
            return value
        values = []
        for metric in SCORE_METRICS:
            if metric == "selection_score":
                rank = int(val_order.iloc[index])
            else:
                values_metric = summary[f"{metric}_mean"]
                rank = int(values_metric.rank(method="min", ascending=metric.endswith("nhd")).iloc[index])
            values.append(fmt(metric, rank))
        lines.append(f"| {row['model']} | {int(row['folds'])} | " + " | ".join(values) + " |")
    lines += ["", "## Test metrics", "",
              "| Model | Selection score | Fine Dice | Fine NHD | Coarse Dice | Coarse NHD | Fine pixel accuracy | Coarse pixel accuracy | Visibility accuracy | Visibility macro F1 | Visibility macro AUROC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index, (_, row) in enumerate(summary.iterrows()):
        values = []
        for metric in TEST_METRICS:
            values_metric = summary[f"test_{metric}_mean"]
            rank = int(values_metric.rank(method="min", ascending=metric.endswith("nhd")).iloc[index])
            value = f"{row[f'test_{metric}_mean']:.4f} ± {row[f'test_{metric}_std']:.4f}"
            values.append(f"**{value}**" if rank == 1 else f"<u>{value}</u>" if rank == 2 else value)
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    lines += ["", "## Validation vs test: important score components", "",
              "| Model | Selection score Val | Test | Fine Dice Val | Test | Fine NHD Val | Test | Coarse Dice Val | Test | Coarse NHD Val | Test | Visibility F1 Val | Test | AUROC Val | Test |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    comparison_metrics = ("selection_score", "fine_dice", "fine_nhd", "coarse_dice", "coarse_nhd",
                          "visibility_macro_f1", "visibility_macro_auroc")
    for _, row in summary.iterrows():
        values = [row["model"]]
        for metric in comparison_metrics:
            values.extend([
                f"{row[f'{metric}_mean']:.4f}",
                f"{row[f'test_{metric}_mean']:.4f}",
            ])
        lines.append("| " + " | ".join(values) + " |")
    m2f_models = {
        "mask2former_swin_tiny": "Mask2Former Swin-Tiny",
        "mask2former_resnet50": "Mask2Former R50 (random)",
        "mask2former_resnet50_official": "Mask2Former R50 (official pretrained)",
    }
    lines += ["", "## Mask2Former baseline validation quantitative", "",
              "| Baseline | Val folds | Val loss | Fine Dice | Fine NHD | Fine pixel accuracy | Coarse Dice | Coarse NHD | Visibility accuracy | Selection score |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id, label in m2f_models.items():
        histories=[]
        for path in sorted((Path(__file__).resolve().parents[1]/"artifacts"/"mask2former"/model_id).glob("fold_*/epoch_metrics.json")):
            data=json.loads(path.read_text())
            if data: histories.append(min(data,key=lambda x:x.get("val_loss",float("inf"))))
        if histories:
            def avg(key):
                vals=[x[key] for x in histories if x.get(key) is not None]
                return f"{np.mean(vals):.4f} ± {np.std(vals,ddof=1):.4f}" if len(vals)>1 else "N/A"
            values=[label,str(len(histories)),avg("val_loss"),avg("val_fine_dice"),avg("val_fine_nhd"),avg("val_fine_pixel_accuracy"),"N/A","N/A","N/A","N/A"]
        else: values=[label,"0/5"]+["N/A"]*8
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Mask2Former validation currently reports fine segmentation metrics only; coarse, visibility and official selection score are N/A in this runner.", "",
              "## Mask2Former baseline test quantitative", "",
              "| Baseline | Test folds | Fine Dice | Fine NHD | Coarse Dice | Coarse NHD | Fine pixel accuracy | Coarse pixel accuracy | Visibility accuracy | Visibility macro F1 | Visibility macro AUROC | Selection score |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    m2f_models = {
        "mask2former_swin_tiny": "Mask2Former Swin-Tiny",
        "mask2former_resnet50": "Mask2Former R50 (random)",
        "mask2former_resnet50_official": "Mask2Former R50 (official pretrained)",
    }
    for model_id, label in m2f_models.items():
        path = Path(__file__).resolve().parents[1] / "artifacts" / "mask2former" / model_id / "test_quantitative.json"
        if path.is_file():
            result = json.loads(path.read_text()); s = result["summary"]
            def m(metric): return f"{s[metric]['mean']:.4f} ± {s[metric]['sample_standard_deviation']:.4f}"
            values = [label, str(result["completed_test_folds"]), m("fine_dice"), m("fine_nhd"),
                      m("coarse_dice"), m("coarse_nhd"), m("fine_pixel_accuracy"),
                      "N/A", "N/A", "N/A", "N/A", "N/A"]
        else:
            values = [label, "0/5", *(["N/A"] * 10)]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Mask2Former currently has no visibility head; therefore visibility metrics and official `selection_score` are N/A.", "",
              "## Model potential by task", "",
              "| Task | Recommended model | Evidence from test |",
              "|---|---|---|",
              "| Task 1 — Fine segmentation | **Mask2Former Swin-Tiny** | Fine Dice 0.7061 ± 0.0435 |",
              "| Task 2 — Coarse segmentation | **Mask2Former Swin-Tiny** | Coarse Dice 0.6401 ± 0.0624 |",
              "| Task 3 — Visibility classification | **SAM2 Hiera Small/Tiny** | Small: AUROC 0.8936, accuracy 0.8133; Tiny: macro F1 0.5862 |",
              "", "Overall recommendation: use Mask2Former Swin-Tiny for segmentation; use SAM2 Hiera Small when all three tasks must be supported. A combined Mask2Former segmentation model plus visibility head is a promising next experiment.", "",
              "**Bold** = best result; <u>underline</u> = second best. For NHD, lower is better; for all other metrics, higher is better.", "",
              "## Generated files", "", "- `training_curves_3_models.png` — loss, validation score, and train/validation accuracy.",
              "- `score_summary.csv` — mean ± SD summary used for ranking.",
              "- `score_by_fold.csv` — raw validation fold-level score table.",
              "- `test_cv_summary.json` — test score source for each model.", ""]
    (output / "TRAINING_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = {model: aggregate_curves(args.artifacts / f"{model}_partial") for model in MODELS}
    plot_curves(curves, args.output)
    summary = build_score_table(args.artifacts, args.output)
    write_report(summary, curves, args.output)
    print(summary[["model", "folds", "val_selection_score_mean", "test_selection_score_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
