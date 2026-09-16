# Experiment log: full40 fine segmentation (31 IDs)

## Material Passport

- Origin Skill: experiment-agent / code-runner-agent
- Origin Mode: execution preparation
- Started: 2026-08-26 UTC
- Dataset: 40 cases, 524 paired frames, 6 centers
- Split manifest: `splits/case_folds_v1.json`
- Split seed: 2026
- Model RNG: base seed 2026; effective seed is `2026 + fold` (historical
  trainer convention).
- Current status: SEED-2026 FIVE-FOLD TRAINING COMPLETED

## Historical reference

The former 10-case A0 and A1 runs both use 31 output IDs. A0 is the clean
no-context baseline and has local fine task score 0.770824339; it is used as a
configuration reference only, never as initialization for these folds.

## Run 00 — data and split audit

- Goal: isolate all full-data artifacts and verify case-level 28/4/8 splits.
- Expected: 40 cases, 524 paired frames, every case in test once, validation
  cases unique across folds, no frame leakage within a fold.
- Result: PASS.
- Verified: 40 cases, 524/524 paired frames, matching dimensions, valid RGB
  values, disjoint 28/4/8 case splits, every case appears in outer test once.
- Dataset caveat: 30 of 31 label colors occur. Fine ID 22 (`Right bronchial
  artery`) is absent from the public training masks, but its output slot is
  retained for evaluation compatibility.
- Verification: 9 focused tests passed; Python entry points compile/import and
  both Slurm scripts pass `bash -n`.

## Run 01 — three-epoch fold-0 smoke

- Experiment: `full40_fine31_c4_a0_seed2026`
- Effective model seed: 2026 (`seed + fold`).
- Train/validation/test: 28/4/8 cases from the immutable manifest.
- Purpose: verify loading, forward/backward, validation metric, checkpoint and
  output isolation before allocating the five-fold run.
- Exact submit command: `sbatch full_version/task2_fine_31cls/jobs/smoke_fold0.sbatch`
- Submitted: 2026-08-26 UTC as Slurm job `400`.
- Initial state at 2026-08-26 09:27 UTC: `RUNNING` on `vishc-server-2`.
- GPU selection: physical GPU 2, 24114 MiB free at startup.
- Initial log: pretrained weights loaded; the 133-to-31 class-head
  reinitialization warning is expected for downstream fine-tuning.
- Log: `logs/smoke_400.log`.

## Run 02 — seed-2026 five-fold baseline

- Slurm array: job `401`, indices 0..4. Initially submitted at maximum
  concurrency 1; updated in-place to concurrency 2 on 2026-08-26 at the user's
  request via `scontrol update JobId=401 ArrayTaskThrottle=2`.
- Submit command: `sbatch --dependency=afterok:400 full_version/task2_fine_31cls/jobs/train_5fold_seed2026.sbatch`.
- Dependency: `afterok:400`; no array element can start unless smoke job 400
  completes successfully.
- Initial state at 2026-08-26 09:27 UTC: `PENDING (Dependency)`.
- After smoke completion: fold 0 (`401_0`, numeric job 402) runs on physical
  GPU 2; fold 1 (`401_1`, numeric job 403) runs concurrently on physical GPU
  5. Folds 2--4 remain gated by `JobArrayTaskLimit` until one active fold ends.
- Expected logs: `logs/train_s2026_401_<fold>.log`.
- Final state checked 2026-08-27: all five folds wrote `completion.json`; job
  `401` is no longer active in `squeue`. Slurm accounting storage is disabled,
  so completion is verified from the run artifacts and logs rather than
  `sacct`.
- Best-validation summary: mean fine task score 0.772363 (sample SD 0.023195),
  mean fine Dice 0.733470 (sample SD 0.027374), and mean fine nHD 0.188743
  (sample SD 0.019290).
- Full fold table: `SCORE_TABLE.md`.

## Promotion rule

Only submit the seed-2026 five-fold array after smoke completion is inspected.
Do not tune hyperparameters from outer-test results; select checkpoints and
candidate changes from validation metrics, then evaluate the frozen candidate.

## Run 03 — 40-case held-out OOF test

- Submitted: 2026-08-27 UTC.
- Inference array: Slurm job `407`, folds 0..4, maximum concurrency 2.
- Initial active folds: fold 0 on physical GPU 2 and fold 1 on physical GPU 5.
- Remaining folds: 2..4 gated by `JobArrayTaskLimit` and start automatically as
  an active fold completes.
- Aggregation job: Slurm job `410`, dependency `afterok:407`.
- Fold outputs: `artifacts/seed_2026/test_oof/folds/fold_<fold>.json`.
- Final output: `artifacts/seed_2026/test_oof/oof_test_metrics.json`.
- Aggregation contract: every one of the 524 frames and 40 cases must occur
  exactly once; report exact case-level OOF score, per-center score and the
  worst center.
- Initial status: folds 0 and 1 `RUNNING`; model/checkpoint initialization
  succeeded with no error. Aggregator `PENDING (Dependency)`.
- Final inference status: all five fold JSON files completed successfully.
- Slurm aggregator `410` failed immediately on `vishc-server-1` with exit code
  1 and produced no log/output. Per the no-auto-retry rule, it was not retried
  until the user explicitly requested it.
- Retry: deterministic aggregation ran directly in the workspace on 2026-08-27
  and completed successfully without repeating inference.
- Final 40-case/524-frame OOF result: fine task score 0.774027, fine Dice
  0.734709, fine nHD 0.186656. Worst center: center 4, score 0.725984.
- Final artifact: `artifacts/seed_2026/test_oof/oof_test_metrics.json`.
