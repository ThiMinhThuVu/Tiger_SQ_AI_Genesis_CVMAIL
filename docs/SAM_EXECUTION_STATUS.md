# SAM baseline execution status

Updated: 2026-07-17 UTC

| Phase | State | Evidence |
|---|---|---|
| Mandatory source review | Partial due to missing named legacy files | Gap documented in `SAM_BASELINE_PLAN.md`; available protocols and official repositories read |
| Full dataset/fold audit | Passed | `artifacts/new_baselines/data_audit.json` |
| SAM 2.1 Tiny implementation | Completed | Prompt-free encoder/decoder/training/prediction modules |
| Unit tests | Passed | 11/11, including real-checkpoint integration |
| Official evaluator conformance | Passed | Unmodified commit `f0b701c`, all three tasks on generated fixture |
| Login-node GPU probe | No device, as expected | Slurm GPU jobs used |
| Fold-0 smoke | Passed | Job 75; 1.04 GB peak reserved |
| Fold-0 stability | Passed | Job 76; resume at epoch 5; all stability gates passed |
| Fold-0 primary training | Completed | Job 77; 100/100 epochs; best epoch 71, local score 0.7716 |
| Fold-0 prediction validation | Passed | 28 frames, all three tasks, derived coarse masks, no oracle prompt |
| Fold-0 qualitative analysis | Completed | Eight required selections and figures |
| Fold-0 native official evaluation | Failed: timeout | Return code 124 after 300 seconds in native Task 1 |
| Folds 1–4 | Not started | Correctly stopped by failed Fold-0 completion gate |
| Later SAM variants | Blocked | Hiera Tiny five-fold baseline is not complete |

The registry status is `failed`, with `completed_folds: []`, because an
official-evaluator pass is mandatory before a fold or baseline can be called
complete. The trained checkpoint and native predictions are preserved. Resume
from the evaluator stage after an approved scalable nHD evaluation path is
available; no retraining is required.
