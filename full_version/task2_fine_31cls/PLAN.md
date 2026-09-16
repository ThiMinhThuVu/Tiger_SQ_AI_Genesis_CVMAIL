# Full-version fine segmentation plan (31 IDs)

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-26
- Verification Status: UNVERIFIED on the 40-case split
- Version Label: full40_fine31_plan_v1

## Scope

- Official task: Task 2, fine-grained semantic segmentation.
- Output space: IDs 0..30 (background plus 30 foreground classes). ID 22 is
  unobserved in this training snapshot but remains in the output space because
  the evaluation labelmap requires all 31 IDs.
- Data snapshot: 40 cases, 524 images, 6 centers.
- Isolation rule: do not write new artifacts into historical 10-case output
  directories.

## Output layout

```text
full_version/task2_fine_31cls/
  configs/
  splits/
  jobs/
  artifacts/<experiment>/seed_<seed>/fold_<fold>/
  logs/
  reports/
  submission/
```

Directories are created by the preparation/training stage when they receive
their first tracked artifact.

## First baseline

- Experiment ID: `full40_fine31_c4_a0_seed2026`
- Architecture: Mask2Former Swin-Small.
- Initialization: `facebook/mask2former-swin-small-coco-panoptic` only.
- Labels: 31 fine IDs decoded from `data/masks_fine/`.
- Matching: asymmetric partial Sinkhorn, fixed target mass.
- Clinical weighting: enabled.
- Station context: disabled.
- Direct coarse, hierarchy and presence heads: disabled.
- Input: 512 x 896.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4.
- Batch: physical 4, gradient accumulation 1.
- Maximum epochs: 100.
- Early stopping: patience 15, min delta 1e-4.
- Checkpoint metric: `(fine_weighted_dice + 1 - fine_weighted_nHD) / 2`.

The historical 10-case checkpoint must not initialize cross-validation runs,
because it has already seen cases that may appear in a new validation/test
fold. It remains a reference result only.

## Split protocol

- Unit of grouping: `center_X_case_Y`; frames from a case never cross splits.
- Five deterministic folds, stratified by center and approximately balanced by
  frame count/resolution.
- Per fold: 28 train cases, 4 validation cases, 8 held-out local test cases.
- Every case appears in held-out test exactly once across the five folds.
- Split seed: 2026; the generated case manifest is immutable once training
  starts.
- Validation selects checkpoints and tuning decisions. Local test is evaluated
  only after a configuration is frozen for that run.

## Seed protocol

1. Primary comparison: base seed 2026 on all five folds. To preserve the
   historical trainer convention, effective RNG seed is `base_seed + fold`,
   hence 2026, 2027, 2028, 2029 and 2030 for folds 0..4.
2. Stability confirmation: repeat the winning configuration with base seeds
   2027 and 2028 on all five folds only.
3. Report fold and seed variation separately; do not treat 15 runs as 15
   independent patient cohorts.
4. Final submission refit: train the frozen winner on all 40 cases with seeds
   2026, 2027 and 2028, then ensemble probabilities.

## Execution gates

1. Data/parser audit passes on 524 image/fine-mask pairs and all 40 cases.
2. One-fold, three-epoch training smoke completes and writes its checkpoint,
   validation metrics and provenance. Native-size RGB output is checked at the
   later inference/submission gate.
3. Seed-2026 five-fold baseline completes before any new ablation is promoted.
4. A tuned candidate must improve the mean fine task score and must not create a
   material regression in either Dice or nHD or in worst-center performance.
5. Ensemble weights and post-processing thresholds are fit from validation/OOF
   predictions only, never from hidden evaluation data.
