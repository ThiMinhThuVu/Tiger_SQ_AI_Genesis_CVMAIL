# Additional baseline plan: DINOv2, nnU-Net, MedSAM

This plan keeps the existing case-level five-fold protocol (`512 x 896`, seed
2026, held-out validation and test cases) and records the important scope
difference between the methods.

| Baseline | Primary setup | Tasks | Prompt / leakage rule | First executable milestone |
|---|---|---:|---|---|
| DINOv2 | DINOv2 ViT-S/14 encoder + 31-class semantic decoder; coarse derived from fine; visibility head added only if the multitask runner is implemented | 1, 2, 3 when the head is present | frozen or partial encoder; no test-time adaptation | one-fold smoke, then five-fold train/val/test |
| nnU-Net | nnU-Net v2 2D segmentation pipeline | 1, 2 | no visibility head; no overall three-task score | dataset conversion audit and one-fold pilot |
| MedSAM | MedSAM ViT-B image encoder + semantic decoder; coarse derived from fine; visibility head | 1, 2, 3 | no GT boxes; fixed full-image box, learned prompt, or train-fold-only predicted prompt | checkpoint/model import smoke, then one-fold pilot |

## Execution order

1. DINOv2 ViT-S/14, because `transformers` is already available in the
   project environment and it can reuse the current TIGER dataset loader.
2. nnU-Net v2, after the nnU-Net package and its dataset conversion pass. Its
   native preprocessing and training budget will be reported explicitly rather
   than mixed into the multitask overall score.
3. MedSAM ViT-B, after verifying the official checkpoint import. The primary
   run will use a fixed full-image box if the original prompt decoder is used;
   no validation/test ground-truth geometry may enter the prompt.

## Reporting

Every completed baseline must produce train/validation/test metrics and test
qualitative figures. DINOv2 and MedSAM will be compared on fine Dice, fine
NHD, coarse Dice, coarse NHD, pixel accuracy, visibility F1 and the existing
selection score when all heads are present. nnU-Net will report fine/coarse
segmentation metrics only and will be marked `N/A` for visibility,
selection_score and the overall three-task score.

The current Mask2Former all-test qualitative job is left running; these new
jobs should be queued after the smoke/import checks so the GPU is not
oversubscribed.
