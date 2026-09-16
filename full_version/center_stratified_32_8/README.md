# Center-stratified 5-fold rerun (32 train / 8 validation)

## Dataset-517 retrain (2026-09-12)

The current active dataset follows the challenge correction: seven named
frames are retired and both masks for `center_7_case_5_10L.png` are replaced
with the corrected Synapse entities. The new immutable manifest is
`splits/case_folds_d517.json`; it preserves the reviewed case-level fold
assignment and recomputes frame counts (101--106 validation frames per fold).

The retrain reproduces the recipes used by the previous submissions without
overwriting them: Task 1 P0 (ver1), Task 2 P0 (ver1 and the ver2 Task-3
encoder), Task 2 P2 (ver2), followed by Task 3 using the retrained P0 encoders.
All 15 segmentation folds share one two-GPU allocation with one sequential
worker pinned to physical GPU 1 and GPU 3, so the rerun never uses more than
two GPUs. Task 3 subsequently uses GPU 1:

```bash
seg_job=$(sbatch --parsable jobs/retrain_d517_segmentation_2gpu.sbatch)
sbatch --dependency=afterok:${seg_job} jobs/train_d517_task3.sbatch
sbatch --dependency=afterok:${seg_job} jobs/train_d517_task3_p2.sbatch
```

This is the no-local-test rerun of the strongest available official-task
recipes in this repository:

- Task 1: independent 16-class coarse-only Mask2Former Swin-Small.
- Task 2: 31-class Mask2Former Swin-Small with the Improved-v2 coarse,
  hierarchy, and presence heads plus asymmetric partial OT matching.
- Task 3: frozen Task-2 encoder plus the tuned visibility head. Thresholds
  are cross-fitted: each validation fold uses thresholds fitted on the other
  four folds.

The immutable assignment is case-grouped. Every case is validation once.
Centers 2, 3, and 7 occur in all five validation folds. Centers 4 and 6 have
only three cases each and are therefore spread across three distinct folds;
requiring either center in all five folds is mathematically impossible without
reusing cases. Every validation fold covers 5 or 6 centers.

Checkpoint selection is center-macro rather than case-macro: Task 1 monitors
the mean coarse task score across represented centers; Task 2 monitors the mean
four-component fine/coarse surrogate across centers. Thus center 1 cannot win a
checkpoint merely through its larger case count.

## Prepare and run

```bash
python full_version/center_stratified_32_8/scripts/prepare_manifest.py
sbatch full_version/center_stratified_32_8/jobs/train_task1.sbatch
sbatch full_version/center_stratified_32_8/jobs/train_task2.sbatch
```

Both training arrays use one GPU per fold and permit at most two concurrent
folds (`--array=0-4%2`), so no more than two GPUs are used. After all Task-1
and Task-2 folds finish:

```bash
sbatch full_version/center_stratified_32_8/jobs/evaluate_task1.sbatch
sbatch full_version/center_stratified_32_8/jobs/evaluate_task2.sbatch
sbatch full_version/center_stratified_32_8/jobs/train_task3.sbatch
```

The evaluators write validation-only OOF reports containing `per_center`,
`worst_center`, and macro-over-centers metrics. No command evaluates a local
test split and no output is named `test_quantitative`.

## P2 high-resolution comparison

`task2_fine_p2_hr_best.yaml` is a controlled P2 rerun: it keeps the current
Improved-v2 recipe and center-macro checkpoint selection, changing only the
input from 512x896 to 640x1120 and the physical/accumulated batch from 4/1 to
2/2. Its artifacts, report, and retrained Task-3 probes use `_p2` paths and do
not overwrite the submitted ver1/P0 artifacts.
