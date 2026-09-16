# Full40 fine-31 baseline: seed-2026 validation scores

## Material Passport

- Origin Skill: experiment-agent / validation
- Generated: 2026-08-27 UTC
- Experiment: `full40_fine31_c4_a0_seed2026`
- Slurm array: `401`
- Split manifest: `splits/case_folds_v1.json`
- Verification Status: COMPLETED on five validation folds

## Best-checkpoint table

Checkpoint selection uses the maximum validation fine task score:
`(weighted Dice + 1 - weighted nHD) / 2`. Epoch indices are zero-based.

| Fold | Effective seed | GPU | Train/val/test frames | Epochs run | Best epoch | Fine task score | Fine Dice | Fine nHD | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 | 2026 | 2 | 366 / 52 / 106 | 60 | 44 | 0.746502 | 0.706273 | 0.213269 | Completed |
| 1 | 2027 | 5 | 371 / 51 / 102 | 45 | 29 | 0.809433 | 0.778363 | 0.159498 | Completed |
| 2 | 2028 | 4 | 368 / 50 / 106 | 46 | 30 | 0.762427 | 0.719160 | 0.194306 | Completed |
| 3 | 2029 | 2 | 367 / 53 / 104 | 37 | 21 | 0.773910 | 0.735756 | 0.187936 | Completed |
| 4 | 2030 | 4 | 368 / 50 / 106 | 37 | 21 | 0.769543 | 0.727796 | 0.188709 | Completed |
| **Mean** | — | — | — | — | — | **0.772363** | **0.733470** | **0.188743** | **5/5 completed** |
| **Sample SD** | — | — | — | — | — | **0.023195** | **0.027374** | **0.019290** | — |

## Progress and interpretation boundary

- Training progress: 5/5 folds completed; every fold wrote `best.pt`,
  `best_validation_metrics.json`, `epoch_metrics.json`, `early_stopping.json`
  and `completion.json`.
- Best fold: fold 1, task score 0.809433.
- Lowest fold: fold 0, task score 0.746502.
- Fold spread is material, so the mean must be reported with its fold-level
  variation rather than only the best fold.
- These are validation scores used for checkpoint selection. They are not yet
  the held-out local-test score and are not the hidden challenge evaluation.
- The historical 10-case score 0.770824 uses a different dataset/split and is
  therefore not a like-for-like improvement claim.

## Next evaluation gate

Run inference from each fold's frozen `best.pt` on that fold's eight held-out
test cases, aggregate the 40 out-of-fold predictions once, and report overall,
per-center and worst-center Dice/nHD before choosing a tuned candidate.

## Held-out OOF test result

| Fold | Fine task score | Fine Dice | Fine nHD |
|---:|---:|---:|---:|
| 0 | 0.783868 | 0.743661 | 0.175925 |
| 1 | 0.756271 | 0.716866 | 0.204323 |
| 2 | 0.778470 | 0.742946 | 0.186007 |
| 3 | 0.783829 | 0.744826 | 0.177167 |
| 4 | 0.767696 | 0.725247 | 0.189856 |
| **40-case OOF** | **0.774027** | **0.734709** | **0.186656** |
| **Fold sample SD** | **0.011917** | **0.012828** | **0.011489** |

The exact aggregate covers 40 cases and 524 frames once each. Center 4 is the
worst center (three cases, task score 0.725984). Full machine-readable metrics
are stored in `artifacts/seed_2026/test_oof/oof_test_metrics.json`.
