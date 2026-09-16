#!/usr/bin/env python3
"""Emit the 5 per-fold Recall-Loss (B1) configs into full_version + this patch.

Each config == the frozen C4-A0 baseline (same backbone / OT matching / lr /
schedule / split / selection_metric) with exactly two additions:
  recall_reweight_weight: 2.0
  recall_reweight_vector_path: <per-fold OOF miss-rate vector>
Single-factor change. Shared experiment_id so folds land in one artifact dir.
The array sbatch maps SLURM_ARRAY_TASK_ID -> the matching fold config.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATCH = ROOT / "patch_hieu_recall_loss/configs"
BASE = (ROOT / "full_version/task2_fine_31cls/configs/full40_fine31_c4_a0_seed2026.yaml").read_text()

EXPERIMENT_ID = "full40_fine31_recall_b1_seed2026"

for k in range(5):
    text = BASE.replace(
        "experiment_id: full40_fine31_c4_a0_seed2026",
        "experiment_id: %s" % EXPERIMENT_ID,
    ).replace(
        "research_stage: full40_fine31_clean_baseline",
        "research_stage: recall_loss_b1_clean_rerun_40case_fold%d" % k,
    )
    vector = "full_version/task2_fine_31cls/artifacts/recall_vectors/fold%d.json" % k
    text += (
        "\n# Recall Loss (Tian et al., arXiv:2106.14917): scale per-class CE by\n"
        "# 1 + weight * out-of-fold complete-miss-rate. Targets complete-miss\n"
        "# (39.6%% of the nHD budget per P1). Vector for fold %d is estimated\n"
        "# ONLY from OOF folds != %d. No confusion-graph / pairwise structure.\n"
        "recall_reweight_weight: 2.0\n"
        "recall_reweight_vector_path: %s\n" % (k, k, vector)
    )
    OUT_PATCH.mkdir(parents=True, exist_ok=True)
    p = OUT_PATCH / ("full40_fine31_recall_b1_fold%d_seed2026.yaml" % k)
    p.write_text(text)
    print("wrote", p)
