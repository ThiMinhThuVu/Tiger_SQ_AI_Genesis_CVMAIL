#!/usr/bin/env python3
"""Build per-fold out-of-fold complete-miss-rate vectors for Recall Loss (B1).

Leakage guard: the vector for target fold k is estimated ONLY from the held-out
OOF test metrics of folds != k. Under case-level 5-fold CV each case is a test
case in exactly one fold, so folds != k cover cases disjoint from fold k's own
test cases; fold k never sees its own held-out predictions. Mirrors
scripts/build_dsml_confusion_graph.py, sourced from the full40 OOF fold JSONs
instead of the 10-case A0 research artifact.

Output: full_version/task2_fine_31cls/artifacts/recall_vectors/foldK.json with
key "miss_rate" of length 31, read by
scripts/train_mask2former_improved.load_recall_reweight_vector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS_DIR = ROOT / "full_version/task2_fine_31cls/artifacts/seed_2026/test_oof/folds"
NUM_LABELS = 31


def build_vector(folds_dir: Path, exclude_fold: int) -> list:
    miss = [0.0] * NUM_LABELS
    present = [0.0] * NUM_LABELS
    used = []
    for path in sorted(folds_dir.glob("fold_*.json")):
        data = json.loads(path.read_text())
        if int(data["fold"]) == exclude_fold:
            continue
        used.append(int(data["fold"]))
        for cls in data["failure_metrics"]["per_class"]:
            cid = int(cls["class_id"])
            p = float(cls.get("present_images", 0) or 0)
            r = float(cls.get("complete_miss_rate") or 0.0)
            miss[cid] += r * p
            present[cid] += p
    if not used:
        raise ValueError("No OOF fold JSONs left after excluding fold %d" % exclude_fold)
    vector = [(miss[c] / present[c]) if present[c] > 0 else 0.0 for c in range(NUM_LABELS)]
    peak = max(vector)
    print("fold %d: built from OOF folds %s; mean miss_rate %.4f; max %.3f at class %d"
          % (exclude_fold, used, sum(vector) / NUM_LABELS, peak, vector.index(peak)))
    return vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "full_version/task2_fine_31cls/artifacts/recall_vectors")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        vector = build_vector(args.folds_dir, k)
        out = args.output_dir / ("fold%d.json" % k)
        out.write_text(json.dumps({
            "target_fold": k,
            "source": "full40 OOF test metrics, folds != target",
            "num_labels": NUM_LABELS,
            "miss_rate": vector,
        }, indent=2) + "\n")
        print("  wrote %s" % out)


if __name__ == "__main__":
    main()
