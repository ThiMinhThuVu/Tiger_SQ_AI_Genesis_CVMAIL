# Mask2Former baseline plan

## Variants

| ID | Backbone | Initialization | Purpose |
|---|---|---|---|
| `mask2former_swin_tiny` | Swin-Tiny | `facebook/mask2former-swin-tiny-coco-panoptic` | compact pretrained baseline |
| `mask2former_resnet50` | ResNet-50 | Mask2Former R50 configuration, ImageNet-pretrained backbone | CNN comparison |

Both variants use the existing TIGER protocol: 512×896 input, case-level 5-fold split, seed 2026, maximum 100 epochs, checkpoint selected on validation selection score, and no ground-truth prompts.

## Execution order

1. Install `transformers` and `accelerate` on the GPU environment.
2. Run a one-epoch smoke test for each backbone.
3. Run fold 0 stability test.
4. Run all five primary folds for each variant.
5. Generate validation/test summaries and add both variants to the comparison report.

The ResNet-50 variant must be verified against the installed Transformers version before training: unlike Swin-Tiny, a universally available official Hugging Face Mask2Former R50 checkpoint is not assumed.

An official Facebook/Meta COCO panoptic R50 checkpoint was downloaded for the additional comparison:
`checkpoints/mask2former_official/maskformer2_R50_coco_panoptic_model_final_94dc52.pkl` (168 MB). It is a Detectron2 pickle and requires a Detectron2/official-model adapter; it cannot be loaded directly by the current Hugging Face runner.
