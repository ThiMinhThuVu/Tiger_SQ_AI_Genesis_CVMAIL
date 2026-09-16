#!/usr/bin/env python3
"""Build out-of-fold confusion-graph matrices for the Directed Sink-Margin Loss (DSML).

Leakage guard: for a given target fold, the graph is estimated ONLY from the
other folds' held-out test confusion matrices (already-trained A0 checkpoints,
`artifacts/mask2former_research/mask2former_swin_small_c4_a0_nocontext_bs4_task1/`).
Each TIGER case appears as a test case in exactly one fold under case-level
5-fold CV, so folds != target_fold cover cases that are disjoint from
target_fold's own test cases -- this never reads target_fold's test
predictions. See reports/A_STAR_2027_NOVELTY_SCAN_BROADENED.md section 4.2.

Produces three [num_labels, num_labels] weight matrices (row=donor/GT class,
col=sink/predicted class), used by the B2/B3/B4 ablation configs:

  B2 uniform   -- every off-diagonal pair present in the raw confusion matrix
                  above a minimum-mass threshold gets equal weight 1.0,
                  symmetric. Encodes "some pair exists" only, no structure.
  B3 symmetric -- top-K confusable pairs by raw (donor+sink) mass get equal
                  weight 1.0, symmetric, but restricted to genuinely
                  confusable pairs (pair detection, no direction).
  B4 directed  -- net-flow-derived directed weight: only donor->sink edges
                  where the sink is a net receiver from that donor, weighted
                  by the out-of-fold pixel mass, normalized to [0, 1].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A0_ARTIFACT = (ROOT / "artifacts/mask2former_research"
                / "mask2former_swin_small_c4_a0_nocontext_bs4_task1"
                / "test_task1_quantitative.json")


def load_confusion(path: Path, exclude_fold: int) -> tuple[np.ndarray, dict[int, str]]:
    data = json.loads(path.read_text())
    class_names: dict[int, str] = {}
    conf_sum: np.ndarray | None = None
    used_folds = []
    for fold_entry in data["folds"]:
        fold_id = int(fold_entry["fold"])
        if fold_id == exclude_fold:
            continue
        used_folds.append(fold_id)
        for cls in fold_entry["fine"]["per_class"]:
            class_names[cls["class_id"]] = cls["class_name"]
        cp = np.array(fold_entry["failure_metrics"]["confusion_pixels"], dtype=np.int64)
        conf_sum = cp if conf_sum is None else conf_sum + cp
    if conf_sum is None:
        raise ValueError(f"No folds left after excluding fold {exclude_fold}")
    print(f"Out-of-fold graph for target fold {exclude_fold} built from folds {used_folds}")
    return conf_sum, class_names


def load_recall_reweight_vector(path: Path, exclude_fold: int, num_labels: int) -> np.ndarray:
    """Per-class out-of-fold complete-miss rate, present-image-weighted (Recall-Loss style, B1)."""
    data = json.loads(path.read_text())
    miss_sum = np.zeros(num_labels, dtype=np.float64)
    present_sum = np.zeros(num_labels, dtype=np.float64)
    for fold_entry in data["folds"]:
        if int(fold_entry["fold"]) == exclude_fold:
            continue
        for cls in fold_entry["failure_metrics"]["per_class"]:
            cid = cls["class_id"]
            present = cls.get("present_images", 0) or 0
            miss_rate = cls.get("complete_miss_rate") or 0.0
            miss_sum[cid] += miss_rate * present
            present_sum[cid] += present
    present_sum[present_sum == 0] = 1.0
    return (miss_sum / present_sum).astype(np.float32)


def build_directed(conf: np.ndarray, min_mass: int) -> np.ndarray:
    n = conf.shape[0]
    diag = np.diag(conf)
    matrix = np.zeros((n, n), dtype=np.float64)
    for donor in range(n):
        if donor == 0:  # background is never a meaningful donor/sink for the margin
            continue
        row = conf[donor, :].astype(np.float64)
        row[donor] = 0.0
        row[0] = 0.0  # exclude background as a sink target too
        mislabeled = row.sum()
        if mislabeled < min_mass:
            continue
        matrix[donor, :] = row
    row_sum = matrix.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    normalized = matrix / row_sum
    # Scale so the strongest edge in the whole graph is 1.0 (relative weighting,
    # not a probability distribution -- this is a margin weight, not a plan).
    peak = normalized.max()
    if peak > 0:
        normalized = normalized / peak
    return normalized


def build_symmetric_topk(conf: np.ndarray, top_k_per_class: int, min_mass: int) -> np.ndarray:
    n = conf.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for a in range(1, n):
        row = conf[a, :].astype(np.float64)
        row[a] = 0.0
        row[0] = 0.0
        if row.sum() < min_mass:
            continue
        top = np.argsort(-row)[:top_k_per_class]
        for b in top:
            if row[b] <= 0:
                continue
            matrix[a, b] = 1.0
            matrix[b, a] = 1.0
    return matrix


def build_uniform(conf: np.ndarray, min_mass: int) -> np.ndarray:
    n = conf.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for a in range(1, n):
        for b in range(1, n):
            if a == b:
                continue
            mass = conf[a, b] + conf[b, a]
            if mass >= min_mass:
                matrix[a, b] = 1.0
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="Target fold to build a graph for")
    parser.add_argument("--min-mass", type=int, default=5000,
                         help="Minimum out-of-fold mislabeled pixel mass to trust an edge")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K confusion partners per class for B3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/dsml_graphs")
    args = parser.parse_args()

    conf, class_names = load_confusion(A0_ARTIFACT, args.fold)
    n = conf.shape[0]
    names = [class_names.get(i, f"id{i}") for i in range(n)]

    uniform = build_uniform(conf, args.min_mass)
    symmetric = build_symmetric_topk(conf, args.top_k, args.min_mass)
    directed = build_directed(conf, args.min_mass)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, matrix in [("uniform", uniform), ("symmetric", symmetric), ("directed", directed)]:
        out = args.output_dir / f"fold{args.fold}_{label}.json"
        out.write_text(json.dumps({
            "fold": args.fold,
            "variant": label,
            "num_labels": n,
            "class_names": names,
            "min_mass": args.min_mass,
            "top_k": args.top_k if label == "symmetric" else None,
            "matrix": matrix.tolist(),
        }, indent=2) + "\n")
        n_edges = int((matrix > 0).sum())
        print(f"  wrote {out} ({n_edges} nonzero edges, max weight {matrix.max():.3f})")

    miss_rate = load_recall_reweight_vector(A0_ARTIFACT, args.fold, n)
    recall_out = args.output_dir / f"fold{args.fold}_recall_reweight.json"
    recall_out.write_text(json.dumps({
        "fold": args.fold,
        "num_labels": n,
        "class_names": names,
        "miss_rate": miss_rate.tolist(),
    }, indent=2) + "\n")
    print(f"  wrote {recall_out} (mean miss_rate {miss_rate.mean():.3f})")


if __name__ == "__main__":
    main()
