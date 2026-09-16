# Patch: Recall Loss (B1) clean rerun — 40-case Task 2

**Why:** P1 error decomposition — complete-miss is **39.6%** of the nHD budget
(absent-FP another 35.7%; boundary only 24.7%). Recall Loss (Tian et al.,
arXiv:2106.14917) reweights per-class CE by `1 + weight * complete_miss_rate`,
aimed straight at that 39.6%. In the round-3 DSML Phase-0 it scored **0.7944**
on fold 0 — the best of B1–B4 — but under `torch 2.11.0+cu130`, not the repo's
pinned `torch 2.7.1+cu128`, so it was **not comparable to C4-A0**. This rerun on
the correct `.venv` is what makes it interpretable.

**Single-factor:** each per-fold config == frozen C4-A0 baseline
(`full40_fine31_c4_a0_seed2026.yaml`) with only:
```yaml
recall_reweight_weight: 2.0
recall_reweight_vector_path: .../recall_vectors/foldK.json
```
Same backbone / OT matching / lr / schedule / split / selection_metric.
Implementation already in `tiger_models/asymmetric_partial_ot_mask2former.py:304`.

**Leakage guard:** the miss-rate vector for fold k is built ONLY from OOF folds
!= k (`build_recall_reweight_40case.py`). Each case is a test case in exactly
one fold, so folds != k are disjoint from fold k's test cases.

**Known caveat:** ultra-rare classes get extreme miss rates (class 21 Left
subclavian artery = 1.0, present in 1 frame). The loss clamps the vector to
[0,1] and scales `1 + 2.0*rate` -> up to 3x CE weight. Report present-only Dice
and complete-miss rate per class, not just the mean, so a "win" driven only by
the absent-class convention is caught.

## Files
| File | Copy to |
|---|---|
| `build_recall_reweight_40case.py` | run in place (stdlib only, already run — vectors in `artifacts/recall_vectors/`) |
| `gen_recall_configs.py` | run in place (already run) |
| `configs/full40_fine31_recall_b1_fold{0..4}_seed2026.yaml` | `full_version/task2_fine_31cls/configs/` |
| `jobs/train_recall_b1_5fold.sbatch` | `full_version/task2_fine_31cls/jobs/` |
| `jobs/evaluate_recall_b1_test_5fold.sbatch` | `full_version/task2_fine_31cls/jobs/` |
| `jobs/aggregate_recall_b1_test.sbatch` | `full_version/task2_fine_31cls/jobs/` |

## RUNBOOK (thu.vtm)

```bash
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI

# 0. deploy
cp patch_hieu_recall_loss/configs/*.yaml full_version/task2_fine_31cls/configs/
cp patch_hieu_recall_loss/jobs/*.sbatch  full_version/task2_fine_31cls/jobs/
# (recall_vectors/ already written by build_recall_reweight_40case.py)

# 1. train 5-fold, 2-GPU concurrent. WATCH FOLD 0 FIRST.
sbatch full_version/task2_fine_31cls/jobs/train_recall_b1_5fold.sbatch
#   -> artifacts/seed_2026/full40_fine31_recall_b1_seed2026/fold_{0..4}/best.pt

# 2. GATE on fold 0: evaluate just fold 0, compare to C4-A0 fold-0 OOF (0.783868).
sbatch full_version/task2_fine_31cls/jobs/evaluate_recall_b1_test_5fold.sbatch
#   read test_oof_recall_b1/folds/fold_0.json :: task1_score
#   if fold-0 regresses clearly vs 0.783868 -> scancel folds 1-4, stop.

# 3. aggregate (after all 5 eval folds done)
sbatch --dependency=afterok:<eval_jobid> \
  full_version/task2_fine_31cls/jobs/aggregate_recall_b1_test.sbatch
#   -> test_oof_recall_b1/oof_test_metrics.json  (compare to C4-A0 0.774027)
```

## Gate promote
Win only if Task-2 OOF >= C4-A0 (0.774027) on >= 3/5 paired folds AND mean
+>= 0.005 AND present-only Dice / complete-miss rate improve (not just the
absent-class-convention rows). Report paired fold deltas.

## Compose with Improved-v2?
Recall Loss and the Improved-v2 heads (patch_hieu_method_tuning) are orthogonal
(class-CE reweight vs added heads). If both pass their own gate, a combined
config is the natural next single-factor step — do NOT bundle them before each
clears alone.
