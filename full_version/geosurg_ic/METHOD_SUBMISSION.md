# Genesis_CVMAIL — Task 2 D517 decoder adaptation

This technical addendum describes two Task-2 candidates, matched appearance
augmentation and GeoSurg-IC. Team: Genesis_CVMAIL, Synapse team 3602194.
Authorship, affiliations and public-archive consent remain the team's existing
declarations; this addendum does not alter them.

## Background and architecture

Both candidates extend the team's P2 D517 Mask2Former Swin-Small system.
Each model predicts 31 fine semantic IDs, including background. Auxiliary
coarse (16 classes), fine/coarse consistency and fine-class presence objectives
are used in training. Fine-class presence modulates the final fine probability.
The model has approximately 68.7 million parameters; 19,900,431 decoder/head
parameters are updated during the final adaptation. The Swin encoder is frozen
during this adaptation, after the original full-model P2 training.

No lymph-node-station visibility annotations, filename tokens or test-set
metadata are used by the Task-2 inference algorithm.

## Data and original training

Training uses the corrected TIGER snapshot with 517 frames from 40 cases and
six centers. Seven retired frames are excluded and the challenge-provided
corrected masks replace the superseded masks. The fixed, center-stratified
case split has five folds, each containing 32 training and 8 validation cases.
Every case is validation exactly once. Each adapted model starts from the P2
checkpoint trained on its corresponding training cases, never another fold's
checkpoint. The validation sets previously selected the P2 checkpoints; they
are development validation, not independent test data.

RGB input is resized to 640 x 1120 and ImageNet-normalized. The original P2
uses physical batch 2, gradient accumulation 2, AdamW LR 1e-4, weight decay
1e-4, up to 100 epochs, validation early stopping with patience 15 and
minimum delta 1e-4. Checkpoint selection uses the center-macro four-component
fine/coarse Dice and normalized-Hausdorff surrogate. Its supervised objective
retains clinically weighted asymmetric partial OT matching and Mask2Former
losses, plus coarse CE weight 0.6, hierarchical consistency weight 0.2 and
fine-class presence BCE weight 0.2. The original geometric/photometric
augmentation recipe is described in the team's P2 configuration.

## Final adaptation: common recipe

Run 192 optimizer steps separately on each fold's training set, physical batch
one, AdamW LR 1e-5 and weight decay 1e-4, global gradient clipping at 1.0. The
Swin encoder is frozen; the model is in eval mode with autograd enabled to
disable stochastic dropout in the response comparisons. Keep the existing
supervised losses. The final step is used, without best-epoch selection during
adaptation. All runs execute sequentially on one RTX A5000 GPU.

At a 160 x 280 GT-label grid, sample one touching non-background semantic class
pair, requiring at least eight neighbor transitions. Independently perturb
each class's appearance with per-channel RGB gains in [0.8, 1.2] and offsets in
[-0.05, 0.05], clipped to [0, 1]. The two masks are disjoint; all disconnected
instances of a selected class share its perturbation. This yields four views:
clean, side A perturbed, side B perturbed, both perturbed. Supervised training
cycles evenly through these four views. No geometric augmentation is applied
during the final adaptation, retaining alignment with cached pseudo-depth.

### Matched augmentation candidate

Only the common supervised adaptation is applied. There is no IC gradient.
Pseudo-depth used by shared experiment diagnostics does not select images,
class pairs or perturbations, and does not influence this candidate's updates.

### GeoSurg-IC candidate

Frozen Depth Anything V2 Small predicts relative pseudo-depth locally from
training images. Depth normalized by the image's 10th--90th percentile range
provides geometry evidence. Compare local within-class depth means in 9 x 9
neighborhoods around a two-pixel GT boundary band. Select up to the top 25% of
band pixels with normalized depth difference at least 0.03. This evidence only
routes the regularizer; it never prescribes semantic feature affinity.

For semantic logit margin m = log(p_a) - log(p_b), compute

    delta = m(both) - m(A) - m(B) + m(clean)
    L_IC = sum_x w(x) delta(x)^2, with sum_x w(x) = 1

IC weight is 0.02, applied every second optimizer step, fixed from a training-
only gradient-scale audit. Samples with no eligible geometry route have no IC
loss. All four views receive the exact first derivative via deterministic
replay, retaining one view graph in memory. Independent additive responses
from the two sides are allowed. Depth and perturbation probes are training-
only and are not packaged as inference dependencies.

## Inference and outputs

Five distinct fold models form an equal-weight ensemble. Load all model weights
once, then iterate over every RGB PNG in the mounted input directory. Average
fine probabilities after soft class-presence modulation. Resize the averaged
probabilities to the original frame dimensions using bilinear interpolation,
then argmax. Export an RGB PNG with the exact official 31-ID palette, unchanged
filename and native dimensions, under /output/task2 only. There is no test-time
augmentation, external service, spatial metadata use or morphology cleanup.
All inference code and weights are in the image. The container needs no
network access and receives no command-line arguments.

## Validation and limitations

All five adapted checkpoints must have completed their fixed schedule and
passed the native-size/palette output smoke test before upload. Validation
aggregation is frame to case to center. The attached local five-fold summary
is development evidence, not hidden challenge performance. The initial
single-fold pilot did not show GeoSurg-IC beating matched augmentation; these
remain experimental candidates. No claim of established geometry-specific
generalization or statistical significance is made.

The host cannot access a Docker daemon. Validation uses the extracted final
application layer, exact packaged weights and the pinned host runtime; it does
not claim a completed `docker run`. OCI manifests and content hashes are
audited separately.

## External data and pretrained checkpoints

- Mask2Former Swin-Small COCO-panoptic checkpoint:
  https://huggingface.co/facebook/mask2former-swin-small-coco-panoptic
  (public COCO/ImageNet pretrained backbone lineage).
- Depth Anything V2 Small, training-only geometry/diagnostics:
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
  revision `5426e4f0f36572d16453bbda7a8389317b1bef99`.
  Its model card describes pretraining on synthetic depth images and public
  real-image collections. Those upstream images were not separately downloaded
  or annotated for this challenge experiment.
- No private external dataset, private pretrained model, added private labels
  or hidden challenge-test labels were used for this adaptation.

## Contributions and disclosure

This addendum covers implementation and validation of the two candidate
variants. AI coding assistance was used to implement the user-provided
GeoSurg-IC idea, orchestrate controlled runs, summarize artifacts and prepare
submission assets. Team authorship and approval declarations are unchanged.
