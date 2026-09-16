# TIGER SQ-AI Foundation Model Baseline Implementation Prompt

You are a **Senior Machine Learning Research Engineer** responsible for designing, implementing, and running new baselines for the **TIGER SQ-AI Challenge**.

The objective is to evaluate foundation-model-based approaches for all three challenge tasks, with **SAM-based models as the highest priority**.

Do not reproduce, rerun, or audit existing methods such as:

```text
affine_full
Tversky–Focal ResNet-34
gdf_affine_frozen_bn
monai_gdf_affine_no_boundary
```

Existing results may only be used as reference points in the final report. Do not allocate compute resources to reproducing them.

The experiment server has the following hardware:

```yaml
gpu:
  model: NVIDIA RTX A5000
  vram: 24 GB
  count: 1
```

Design the complete pipeline so that it runs reliably on a single RTX A5000, avoids out-of-memory errors, does not overuse system resources, and never runs multiple training processes concurrently.

# 1. Mandatory Sources to Read

Before implementing anything, read the following files in full:

```text
REPRODUCTION_AND_FAIR_COMPARISON_PROTOCOL.md
REPORT_TASKS_EDA_CURRENT_APPROACH.md
readme.md
train.py
crossval_setups.py
calibrate_visibility.py
generate_qualitative_results.py
tiger_baseline/data.py
tiger_baseline/model.py
tiger_baseline/losses.py
tiger_baseline/metrics.py
```

Also cross-reference the official challenge resources:

```text
https://www.synapse.org/Synapse:syn74209386/wiki/639932
https://www.synapse.org/Synapse:syn74209386/wiki/639938
https://gitlab.com/nct_tso_public/challenges/miccai2026/tigersqai_challenge
```

Do not spend time analyzing or reproducing previous models.

Only perform the minimum checks required to guarantee that the new baselines run correctly:

1. Confirm that the dataset contains exactly 140 frames from 10 cases.
2. Confirm that image, fine-mask, and coarse-mask filenames match.
3. Confirm that RGB masks are decoded correctly using `labelmap.csv`.
4. Confirm that the fine-to-coarse class mapping is correct.
5. Confirm that visibility labels are loaded as 14-dimensional multi-hot vectors.
6. Confirm that the five case-level folds match the required protocol.
7. Confirm that the official evaluator can process generated predictions.
8. Confirm that there is no case leakage or frame leakage.

If any of these checks fail, fix the data or evaluation pipeline before training.

Do not guess class mappings, metric definitions, evaluator behavior, model module names, or checkpoint interfaces. Verify them from the repository, official code, and installed model implementation.

# 2. Three Required Tasks

Each input is one RGB surgical video frame.

The model must produce three outputs:

```text
Task 1:
Fine semantic segmentation
31 logit channels

Task 2:
Coarse semantic segmentation
16 logit channels, or a coarse prediction derived from the fine prediction

Task 3:
Lymph-node station visibility
14 independent logits
```

The required Task 3 order is:

```text
6L, 6R, 7L, 7R, 8, 9, 10L, 10R,
11L, 11R, 12L, 12R, 13L, 13R
```

Task 3 is a multi-label classification task, not a single-label classification task.

# 3. Fixed Training and Validation Protocol

All new baselines must follow the protocol defined in the Markdown files.

## 3.1 Case-Level Split

Use five-fold cross-validation at the case level:

```yaml
fold_0_validation:
  - center_1_case_10
  - center_1_case_15

fold_1_validation:
  - center_1_case_11
  - center_1_case_6

fold_2_validation:
  - center_1_case_12
  - center_1_case_7

fold_3_validation:
  - center_1_case_13
  - center_1_case_8

fold_4_validation:
  - center_1_case_14
  - center_1_case_9
```

Each fold must contain:

```text
Training: 8 cases, 112 frames
Validation: 2 cases, 28 frames
```

Do not randomly split individual frames.

No frame from the same case may appear in both the training and validation partitions.

## 3.2 Input Configuration

Use the following primary comparable setting:

```yaml
height: 512
width: 896
image_resize: bilinear
mask_resize: nearest
crop: false
normalization: ImageNet
```

Do not crop images because Task 3 requires full-image context.

If an encoder requires dimensions divisible by its patch size:

1. Pad the tensor only on the right and/or bottom.
2. Run the encoder.
3. Remove the padding from the output.
4. Resize the prediction to `512×896`.
5. Preserve the original field of view.

Do not reduce the resolution in the primary experiment.

A reduced-resolution run may only be used as an engineering feasibility test. It must be clearly marked as not directly comparable with the primary protocol.

## 3.3 Augmentation

Use the primary augmentation configuration exactly as follows:

```yaml
affine:
  probability: 0.75
  rotation: [-10, 10]
  translation_fraction: [-0.05, 0.05]
  scale: [0.90, 1.10]

brightness:
  probability: 0.75
  factor: [0.82, 1.18]

contrast:
  probability: 0.75
  factor: [0.85, 1.15]

color:
  probability: 0.50
  factor: [0.88, 1.12]

horizontal_flip:
  probability: 0.0
```

Do not use cropping.

Do not use horizontal flipping unless a complete left-right label remapping is implemented for both segmentation classes and station visibility labels.

Horizontal flipping is not part of the primary baseline.

## 3.4 Training Budget

Use the following primary comparable configuration:

```yaml
epochs: 100
seed: 2026
optimizer: AdamW
weight_decay: 1.0e-4
scheduler: cosine
gradient_clip_norm: 1.0
validation_frequency: every_epoch
checkpoint_selection: validation_selection_score
```

The target effective batch size is:

```text
4
```

On the RTX A5000, prefer:

```text
physical batch size: 1
gradient accumulation: 4
```

If the model is sufficiently lightweight and VRAM usage permits:

```text
physical batch size: 2
gradient accumulation: 2
```

Each epoch must remain equivalent to:

```text
112 training samples
28 optimizer steps
```

Do not increase gradient accumulation to 8 if doing so changes the optimizer-step budget.

If gradient accumulation is changed, adjust the sampler or update logic so that each epoch still contains exactly 112 training samples and 28 optimizer steps.

Document precisely how incomplete accumulation windows, if any, are handled.

# 4. RTX A5000 Resource Strategy

Before running each model, perform a resource probe:

```bash
nvidia-smi
```

Then run the following sequence:

1. One training forward pass.
2. One backward pass.
3. One optimizer step.
4. One validation inference pass.
5. Record peak allocated and peak reserved CUDA memory.

Target resource limits:

```yaml
maximum_target_vram: 22 GB
reserved_headroom: approximately 2 GB
concurrent_training_processes: 1
```

Never run two folds or two models concurrently on the same GPU.

## 4.1 Mixed Precision

Use the following priority order:

```text
1. bfloat16 autocast, if the runtime supports it reliably
2. float16 autocast with GradScaler
3. float32 only for debugging
```

Check for NaN and Inf values in:

```text
loss
gradient norm
logits
probabilities
```

The selected precision mode and the reason for any fallback must be recorded in the experiment artifacts.

## 4.2 VRAM-Saving Measures

Apply the following measures when necessary:

```text
gradient checkpointing
physical batch size 1
gradient accumulation 4
channels_last for convolutional decoders, if stable
zero_grad(set_to_none=True)
do not retain full-epoch logits on the GPU
move metric tensors to CPU
run validation sequentially
do not cache image tensors on the GPU
```

For large SAM encoders:

```text
freeze most of the encoder
unfreeze only the final blocks
use a lower encoder learning rate
```

Do not automatically use an 8-bit optimizer or CPU offloading in the primary run.

If either is strictly required, treat it as a separate experimental setting and declare it explicitly.

## 4.3 DataLoader

Use the following defaults:

```yaml
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 2
```

The number of workers may be increased to 8 only if CPU and RAM availability justify it.

Do not increase worker count merely to maximize resource usage.

Do not preload the entire dataset onto the GPU.

RGB-mask decoding results may be cached in CPU RAM if the implementation is safe and deterministic.

# 5. Model Priority Order

## Priority 1 — SAM 2.1 Hiera Tiny

This is the first new baseline that must be implemented and run.

Experiment ID:

```text
sam2_hiera_tiny_multitask
```

SAM 2 must not receive ground-truth prompts during validation or testing.

Primary adaptation:

```text
RGB image
→ SAM 2.1 Hiera Tiny image encoder
→ multi-scale feature adapter
→ semantic decoder
    ├── fine logits: 31 channels
    ├── optional coarse logits: 16 channels
    └── optional boundary logits
→ globally pooled encoder features
    └── visibility MLP: 14 logits
```

Evaluate Task 2 in the following order:

```text
Primary:
Derive the coarse prediction from the fine prediction using fine_id → merged_id.

Secondary ablation:
Train a separate coarse segmentation head.
```

Do not change the loss, augmentation, and coarse-prediction strategy simultaneously.

### SAM 2 Tiny Fine-Tuning Modes

Evaluate the modes in this order:

```text
A. Frozen encoder
B. Partial fine-tuning
C. Full fine-tuning only if VRAM and runtime permit
```

The primary mode should be partial fine-tuning:

```text
freeze early encoder stages
unfreeze the feature neck
unfreeze the final encoder blocks
train the entire semantic decoder
train the visibility head
```

Record the exact module names and parameter groups that are frozen and unfrozen.

Initial learning rates:

```yaml
decoder_lr: 3.0e-4
visibility_head_lr: 3.0e-4
encoder_lr: 2.0e-5
```

An encoder learning rate of `4.5e-5` may be used if training is stable, but it must be registered in the configuration before the run begins.

Do not silently alter learning rates after examining validation performance.

## Priority 2 — SAM 2.1 Hiera Small

Experiment ID:

```text
sam2_hiera_small_multitask
```

Do not begin this model until the Hiera Tiny pipeline has:

1. Passed unit tests.
2. Passed the resource smoke test.
3. Completed at least Fold 0.
4. Produced predictions accepted by the evaluator.
5. Remained within the VRAM limit.

On the RTX A5000, use the following defaults:

```yaml
physical_batch_size: 1
gradient_accumulation: 4
gradient_checkpointing: true
fine_tuning: partial
```

Do not begin with full fine-tuning of Hiera Small.

If peak VRAM exceeds 22 GB:

1. Check for tensors unintentionally retained by the computation graph.
2. Enable gradient checkpointing.
3. Freeze additional encoder blocks.
4. Reduce decoder channel widths.
5. Do not reduce resolution in the primary run.

If the model still encounters OOM at physical batch size 1, record it as:

```text
not feasible under the primary A5000 protocol
```

A reduced-resolution engineering test may then be run only to confirm technical feasibility.

## Priority 3 — MedSAM

Experiment ID:

```text
medsam_vit_b_multitask
```

Do not use ground-truth bounding boxes during validation or testing.

Primary adaptation:

```text
MedSAM image encoder
→ semantic FPN or U-Net-style decoder
→ 31-class fine segmentation
→ coarse prediction derived from the fine prediction
→ pooled encoder features
→ 14-label visibility head
```

If the original MedSAM mask decoder is used, its prompt must be one of the following:

```text
fixed full-image box
learned prompt
predicted prompt generated by a model trained only on the current training fold
```

Never derive a bounding box from validation or test ground truth.

Prefer a frozen or partially fine-tuned encoder because ViT-B is more expensive on the RTX A5000.

## Priority 4 — Original SAM ViT-B

Experiment ID:

```text
sam_vit_b_multitask
```

Use SAM ViT-B as an image encoder with an attached semantic decoder.

Do not use ground-truth points or ground-truth bounding boxes.

Do not run SAM ViT-H or SAM ViT-L on the RTX A5000 during the primary baseline phase.

List SAM ViT-H and SAM ViT-L only as future experiments.

# 6. Models After SAM

Only implement the following models after completing the SAM baseline matrix:

```text
1. Mask2Former
2. DINOv2 ViT-S/14
3. nnU-Net v2 2D
4. DINOv2 ViT-B/14, if resources permit
```

Do not begin these models before at least one SAM baseline has completed all five folds, unless SAM is shown to be technically infeasible.

Do not implement CONCH during the current experimental cycle.

# 7. Loss Function

For a SAM encoder with a semantic decoder, use:

```text
L =
  0.50 × weighted fine cross-entropy
  + 1.00 × weighted fine soft Dice
  + 0.40 × BCEWithLogits visibility
  + optional 0.30 × coarse loss
  + optional 0.08 × boundary loss
```

Obtain clinical segmentation weights from `labelmap.csv`.

For Task 3, calculate `pos_weight` only from the training partition of the current fold:

```text
clip(number_negative / max(number_positive, 1), 1, 8)
```

Do not calculate `pos_weight` using the complete dataset.

The primary SAM baseline must use:

```text
fine segmentation loss
visibility loss
coarse prediction derived from the fine prediction
```

A coarse auxiliary head and a boundary head are secondary ablations.

Do not enable either optional head in the primary run unless explicitly registered as part of a separate experiment.

# 8. SAM Execution Plan

## Phase 1 — Implementation

Create the following modules:

```text
tiger_models/sam2_encoder.py
tiger_models/sam_semantic_decoder.py
tiger_models/multitask_sam.py
configs/sam/
scripts/train_sam.py
scripts/predict_sam.py
scripts/validate_predictions.py
scripts/generate_sam_qualitative.py
```

The exact file structure may be adapted to the existing repository, but maintain a clear separation between:

```text
encoder
decoder
task heads
training loop
prediction
evaluation
visualization
```

Reuse verified repository utilities where appropriate instead of duplicating reliable logic.

## Phase 2 — Unit Tests

Test the following:

```text
input shape
fine-logit shape
coarse-output shape
visibility-logit shape
loss backward pass
RGB-mask encoding
fine-to-coarse mapping
case split
checkpoint save and load
deterministic inference
```

Expected shapes:

```text
input: [B,3,512,896]
fine_logits: [B,31,512,896]
visibility_logits: [B,14]
```

If a coarse head is enabled:

```text
coarse_logits: [B,16,512,896]
```

Tests must use small synthetic or minimal real-data examples and must not allocate unnecessary GPU memory.

## Phase 3 — Resource Smoke Test

Run Fold 0 for one epoch with:

```text
SAM 2.1 Hiera Tiny
physical batch size 1
gradient accumulation 4
mixed precision
```

The smoke test must record:

```text
peak VRAM
time per training iteration
time per validation image
loss values
gradient norm
prediction samples
```

Smoke-test results must not be included in the primary benchmark table.

Store smoke-test artifacts separately from full-run artifacts so that later runs do not overwrite them.

## Phase 4 — Short Stability Test

Run Fold 0 for 10 epochs.

The objectives are to verify that:

```text
the loss decreases
no NaN values occur
no gradient explosion occurs
segmentation does not collapse to background
Task 3 probabilities do not collapse to one constant value
checkpoint resume works
```

The 10-epoch run is only a stability check and must not be presented as a benchmark result.

Store stability-test artifacts separately from the 100-epoch run.

## Phase 5 — Full Fold 0 Run

Run:

```text
SAM 2.1 Hiera Tiny
Fold 0
100 epochs
```

After training:

1. Select the best checkpoint using `selection_score`.
2. Generate predictions.
3. Run the local evaluator.
4. Run the official evaluator.
5. Generate the qualitative report.
6. Review VRAM usage and runtime.
7. Confirm that no oracle or ground-truth prompt was used.

## Phase 6 — Full Cross-Validation

If Fold 0 succeeds, run the remaining folds sequentially:

```text
Fold 1
Fold 2
Fold 3
Fold 4
```

Do not run folds in parallel.

After all five folds are complete, calculate:

```text
aggregated out-of-fold metrics
mean ± sample standard deviation
per-case metrics
per-class metrics
per-station metrics
qualitative error analysis
```

## Phase 7 — SAM Hiera Small

After Hiera Tiny, evaluate Hiera Small in the following order:

```text
1-epoch smoke test
10-epoch stability test
Fold 0 × 100 epochs
five-fold evaluation only if Fold 0 is sufficiently competitive and VRAM usage is acceptable
```

Before starting Hiera Small, define and record a concrete go/no-go criterion in the plan.

The criterion must use available reference metrics, runtime, and VRAM measurements. Do not decide retrospectively using an undocumented subjective judgment.

Do not automatically run all five Hiera Small folds if any of the following occurs:

```text
OOM
runtime exceeds the pre-registered feasibility limit
Fold 0 is clearly below the pre-registered competitiveness threshold
training is unstable
```

# 9. Checkpoint Selection

At every validation epoch, save the following six metrics:

```text
Fine Dice
Fine nHD
Coarse Dice
Coarse nHD
Visibility macro F1
Visibility macro AUROC
```

Use the following selection score:

```text
mean(
  Fine Dice,
  1 - Fine nHD,
  Coarse Dice,
  1 - Coarse nHD,
  Visibility macro F1,
  Visibility macro AUROC
)
```

Use `1 - nHD` only if the evaluator defines nHD on a normalized `[0,1]` scale.

If the official nHD range differs, use the official normalized equivalent and document the exact transformation. Do not apply `1 - nHD` blindly.

Select `best.pt` from the epoch with the highest selection score.

Do not use the final epoch metrics as a substitute for the best checkpoint.

Save:

```text
best.pt
last.pt
best_epoch
best_metrics.json
history.json
```

Handle undefined AUROC values explicitly when a station has only one ground-truth class in a validation fold. Follow the official evaluator behavior and document aggregation rules.

# 10. Prediction Requirements

## Task 1

Generate Task 1 predictions as follows:

```text
fine logits
→ channel-wise argmax
→ fine-ID mask
→ nearest-neighbor resize to the original resolution, when required
→ encode with the correct RGB values from labelmap.csv
→ save as RGB PNG
```

Do not output:

```text
palette PNG
grayscale ID PNG
JPEG
probability heatmap instead of a segmentation mask
```

## Task 2

Primary strategy:

```text
predicted fine ID
→ map fine_id to merged_id
→ encode as coarse RGB
```

Verify that the mapping is applied to predicted class IDs before RGB encoding.

## Task 3

Generate Task 3 output as follows:

```text
visibility logits
→ sigmoid
→ 14 raw probabilities
```

Use the thresholding behavior implemented by the official evaluator for the primary result.

Out-of-fold threshold calibration may only be reported as a secondary analysis.

Do not fit thresholds on the hidden test set.

# 11. Quantitative Report

For every fold, report:

```text
Fine weighted Dice
Fine weighted nHD
Coarse weighted Dice
Coarse weighted nHD
Task 3 macro F1
Task 3 macro AUROC
selection score
best epoch
peak VRAM
training time
inference latency
trainable parameter count
total parameter count
```

After all five folds, report:

```text
mean
sample standard deviation
median
minimum fold
maximum fold
95% bootstrap confidence interval
```

Also report:

```text
per-class Dice
per-class nHD
per-station F1
per-station precision
per-station recall
per-station AUROC
per-case score
```

Classes that are not present in the ground truth of a fold must be marked as:

```text
not observable in this fold
```

Do not automatically assign Dice 0 to an absent class unless the official evaluator explicitly specifies that behavior.

The bootstrap procedure must document:

```text
resampling unit
number of bootstrap samples
random seed
confidence-interval method
```

Prefer case-level resampling when reporting uncertainty across cases, unless the official protocol specifies otherwise.

# 12. Qualitative Evaluation and Error Analysis

For every fold, automatically select:

```text
best sample
median sample
worst sample
sample with the highest nHD
sample containing a class with clinical weight 3
sample containing a rare class
sample with many visibility labels
sample with many Task 3 errors
```

Do not select only visually successful predictions.

Every visualization must include:

```text
A. Original image
B. Fine ground truth
C. Fine prediction
D. Fine semantic error map
E. Fine boundary overlay
F. Coarse ground truth
G. Coarse prediction
H. Coarse error map
I. Prediction entropy or confidence
J. Task 3 ground truth and all 14 predicted probabilities
```

The error map must distinguish:

```text
correct
wrong semantic class
false positive
false negative
boundary disagreement
```

For segmentation overlays:

```text
ground-truth boundary: solid line
prediction boundary: dashed line
```

Generate:

```text
reports/qualitative/<method>/fold_<n>/
reports/SAM_ERROR_ANALYSIS.md
reports/SAM_BASELINE_RESULTS.md
```

The report must clearly separate:

```text
smoke-test observations
stability-test observations
completed primary benchmark results
incomplete or failed runs
```

# 13. Artifact Layout

For each completed primary experiment fold, create:

```text
artifacts/new_baselines/<method>/fold_<n>/
├── config.json
├── command.txt
├── environment.json
├── resource_profile.json
├── git_state.json
├── history.json
├── best.pt
├── last.pt
├── best_metrics.json
├── predictions/
│   ├── task1/
│   ├── task2/
│   └── task3.csv
├── validation_probabilities.npz
├── per_class_metrics.csv
├── per_station_metrics.csv
├── per_case_metrics.csv
└── qualitative/
```

Use separate directories for engineering runs:

```text
artifacts/new_baselines/<method>/engineering/smoke/fold_<n>/
artifacts/new_baselines/<method>/engineering/stability/fold_<n>/
```

`resource_profile.json` must contain:

```json
{
  "gpu": "NVIDIA RTX A5000",
  "gpu_count": 1,
  "peak_allocated_vram_gb": null,
  "peak_reserved_vram_gb": null,
  "physical_batch_size": null,
  "gradient_accumulation": null,
  "effective_batch_size": 4,
  "mixed_precision": null,
  "gradient_checkpointing": null,
  "train_seconds_per_epoch": null,
  "validation_seconds": null,
  "inference_ms_per_image": null
}
```

Replace null values only with measurements that were actually collected.

Do not invent missing measurements.

# 14. Experiment Registry

Create:

```text
configs/experiment_registry.yaml
```

Register at least the following experiments:

```yaml
sam2_hiera_tiny_partial:
  priority: 1
  status: planned
  architecture: SAM2 Hiera Tiny
  adaptation: semantic encoder-decoder
  fine_tuning: partial
  fine_output_classes: 31
  coarse_strategy: derived_from_fine
  visibility_classes: 14
  resolution: [512, 896]
  folds: [0, 1, 2, 3, 4]
  epochs: 100
  physical_batch_size: 1
  gradient_accumulation: 4
  effective_batch_size: 4
  gradient_checkpointing: true
  gpu: RTX A5000 24GB
  max_target_vram_gb: 22

sam2_hiera_small_partial:
  priority: 2
  status: blocked
  blocked_by: sam2_hiera_tiny_partial
  architecture: SAM2 Hiera Small
  adaptation: semantic encoder-decoder
  fine_tuning: partial
  fine_output_classes: 31
  coarse_strategy: derived_from_fine
  visibility_classes: 14
  resolution: [512, 896]
  physical_batch_size: 1
  gradient_accumulation: 4
  effective_batch_size: 4
  gradient_checkpointing: true

medsam_vit_b:
  priority: 3
  status: blocked
  blocked_by: sam2_hiera_tiny_partial
  architecture: MedSAM ViT-B
  oracle_prompt: false

sam_vit_b:
  priority: 4
  status: blocked
  blocked_by: sam2_hiera_tiny_partial
  architecture: SAM ViT-B
  oracle_prompt: false
```

Valid status values are:

```text
planned
implemented
smoke_tested
running
completed
failed
not_feasible
blocked
```

Use additional fields such as `blocked_by`, `failure_reason`, `last_completed_phase`, and `completed_folds` instead of creating undocumented status values.

Registry updates must be atomic and must reflect the actual execution state.

# 15. Execution Commands

Create commands that match the actual repository and installed model APIs.

The desired command format is:

```bash
python scripts/train_sam.py \
  --model sam2_hiera_tiny \
  --fold 0 \
  --n-folds 5 \
  --epochs 100 \
  --height 512 \
  --width 896 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --gradient-checkpointing \
  --finetune partial \
  --mixed-precision bf16 \
  --num-workers 4 \
  --seed 2026 \
  --output-dir artifacts/new_baselines/sam2_hiera_tiny_partial/fold_0
```

Do not implement command-line arguments merely to imitate this example. Ensure that every argument is used correctly by the training code.

If BF16 is unsupported or unstable, use:

```bash
--mixed-precision fp16
```

Create a sequential five-fold script:

```text
scripts/run_sam2_tiny_5fold.sh
```

The script must:

1. Run one fold at a time.
2. Stop when a fold fails.
3. Save stdout and stderr.
4. Never overwrite completed artifacts.
5. Support resume.
6. Record the exit code.
7. Run prediction and evaluation after training.
8. Update the experiment registry.
9. Detect whether a fold is already complete before starting it.
10. Use a lock or equivalent safeguard to prevent two copies from training concurrently.

Before downloading model checkpoints or installing dependencies:

1. Check whether the required files or packages already exist.
2. Record exact package and checkpoint versions.
3. Verify checkpoint compatibility with the selected architecture.
4. Fail with a clear message if network access or credentials are required but unavailable.
5. Do not silently switch to a different architecture or random initialization.

# 16. Completion Criteria

A SAM baseline may only be marked as `completed` when:

```text
all required 100-epoch runs have completed, including valid resumes
the best checkpoint exists
predictions for all three tasks exist
the prediction validator passes
the local evaluator passes
the official evaluator passes
quantitative metrics are saved
the qualitative report is generated
the resource profile is saved
no ground-truth prompt was used during validation or testing
```

Do not fabricate results.

Do not present placeholders as completed measurements.

If there is insufficient execution time to finish all five folds, complete as much real work as possible and report the exact state, for example:

```text
implementation completed
smoke test passed
Fold 0 completed
Fold 1 running
Fold 2 not started
```

Do not use Fold 0 alone to claim that a model is superior to all existing baselines.

A process that was started but did not finish must not be described as completed.

# 17. Required Final Execution Order

Execute the work in the following order:

```text
1. Read the protocol and existing code
2. Perform the minimum dataset, fold, and evaluator checks
3. Design the SAM semantic adaptation
4. Implement SAM 2.1 Hiera Tiny
5. Run unit tests
6. Run the RTX A5000 VRAM probe
7. Run the Fold 0 smoke test
8. Run the Fold 0 stability test
9. Run Fold 0 for 100 epochs
10. Generate predictions and quantitative/qualitative analyses
11. Complete all five SAM 2.1 Hiera Tiny folds
12. Evaluate SAM 2.1 Hiera Small
13. Evaluate MedSAM
14. Evaluate original SAM ViT-B
15. Evaluate Mask2Former, DINOv2, and nnU-Net after SAM is complete
```

Do not perform a reproduction or audit tier for previous methods.

Do not retrain `affine_full`.

Do not retrain Tversky–Focal ResNet-34.

Do not spend compute resources validating old artifacts.

Begin by creating:

```text
docs/SAM_BASELINE_PLAN.md
configs/experiment_registry.yaml
```

The plan must include:

```text
verified repository structure
verified dataset assumptions
model architecture
freeze and unfreeze policy
parameter groups
memory strategy
test strategy
execution commands
go/no-go gates
expected artifacts
known risks
```

After writing the plan, continue with implementation and run the smoke test.

Do not stop after planning alone.

When an operation cannot be completed because of missing data, dependencies, checkpoints, network access, permissions, evaluator credentials, or hardware availability:

1. Complete all independent work that remains possible.
2. Record the exact blocking command and error.
3. Do not fabricate an execution result.
4. Mark the affected experiment with the correct registry status.
5. Provide concrete recovery instructions.
