# TIGER SQ-AI Training Report

**Cập nhật:** 2026-07-22  
**Protocol:** local case-level 5-fold, split 7:1:2, ảnh 512×896  
**Lưu ý:** đây là local test score, không phải hidden-test hoặc leaderboard score.

## Kết luận hiện tại

- **Baseline tốt nhất cho đủ 3 task:** ConvNeXt-V2 Tiny + Mask2Former, test `selection_score` **0,6902 ± 0,0347**.
- **Baseline segmentation-only tốt nhất:** Mask2Former Swin-Tiny, Fine Dice **0,7061 ± 0,0435** và Coarse Dice **0,6401 ± 0,0624**. Model này không có visibility head nên không có `selection_score`.
- ConvNeXt-V2 Base + Mask2Former vẫn đang train; chưa đưa vào xếp hạng test cho tới khi hoàn tất đủ năm fold.

## Baseline đủ ba task

Các giá trị là mean ± sample standard deviation trên năm test fold.

| Model | Selection score | Fine Dice | Fine nHD | Coarse Dice | Coarse nHD | Visibility F1 | Visibility AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ConvNeXt-V2 Tiny + Mask2Former** | **0,6902 ± 0,0347** | **0,6691 ± 0,0522** | 0,2505 ± 0,0399 | **0,5717 ± 0,0708** | 0,3103 ± 0,0522 | 0,5659 ± 0,0723 | **0,8952 ± 0,0300** |
| ConvNeXt-V2 Tiny + FPN | 0,6862 ± 0,0348 | 0,6470 ± 0,0333 | **0,2472 ± 0,0338** | 0,5624 ± 0,0376 | **0,2992 ± 0,0417** | 0,5673 ± 0,0461 | 0,8871 ± 0,0335 |
| SAM2 Hiera Small | 0,6646 ± 0,0403 | 0,6052 ± 0,0437 | 0,2832 ± 0,0459 | 0,5204 ± 0,0598 | 0,3222 ± 0,0676 | 0,5735 ± 0,0688 | 0,8936 ± 0,0404 |
| DeepLabV3-ResNet50 | 0,6630 ± 0,0604 | 0,6266 ± 0,0272 | 0,2604 ± 0,0339 | 0,5426 ± 0,0452 | 0,3027 ± 0,0508 | 0,5166 ± 0,1444 | 0,8552 ± 0,0757 |
| SAM2 Hiera Tiny | 0,6589 ± 0,0443 | 0,5984 ± 0,0252 | 0,2863 ± 0,0354 | 0,5026 ± 0,0451 | 0,3352 ± 0,0500 | **0,5862 ± 0,0946** | 0,8880 ± 0,0364 |
| SAM2 Hiera Base+ | 0,6087 ± 0,0296 | 0,5951 ± 0,0291 | 0,2926 ± 0,0297 | 0,4932 ± 0,0279 | 0,3391 ± 0,0429 | 0,4459 ± 0,0572 | 0,7499 ± 0,0389 |

### ConvNeXt-V2 Tiny + Mask2Former theo fold

| Fold | Best epoch | Selection score | Fine Dice | Coarse Dice | Visibility F1 | AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29 | 0,6947 | 0,6283 | 0,5248 | 0,6836 | 0,9320 |
| 1 | 57 | 0,6471 | 0,6322 | 0,5185 | 0,5093 | 0,8825 |
| 2 | 77 | **0,7284** | **0,7421** | **0,6762** | 0,5050 | 0,8869 |
| 3 | 60 | 0,7176 | 0,7070 | 0,6149 | 0,5576 | 0,9178 |
| 4 | 91 | 0,6631 | 0,6361 | 0,5241 | 0,5742 | 0,8565 |

## Baseline segmentation-only

| Model | Fine Dice | Fine nHD | Coarse Dice | Coarse nHD |
|---|---:|---:|---:|---:|
| **Mask2Former Swin-Tiny** | **0,7061 ± 0,0435** | **0,2177 ± 0,0414** | **0,6401 ± 0,0624** | **0,2564 ± 0,0577** |
| DINOv2 ViT-S/14 | 0,6533 ± 0,0188 | 0,2407 ± 0,0244 | — | — |
| Mask2Former ResNet-50 official | 0,6375 ± 0,0153 | 0,2624 ± 0,0232 | 0,5234 ± 0,0338 | 0,3238 ± 0,0421 |
| Mask2Former ResNet-50 | 0,6267 ± 0,0203 | 0,2663 ± 0,0198 | 0,4987 ± 0,0291 | 0,3386 ± 0,0445 |
| MedSAM ViT-B | 0,5049 ± 0,0135 | 0,3512 ± 0,0208 | — | — |
| nnU-Net 2D | 0,4682 ± 0,0646 | 0,3769 ± 0,0664 | — | — |

## Thí nghiệm đang chạy

### ConvNeXt-V2 Base + Mask2Former

- Slurm job: `136`.
- Fold 0 và fold 1 đã hoàn tất tại thời điểm cập nhật.
- Fold 2 đang train; chưa có test score 5-fold hợp lệ.
- Chỉ cập nhật vào bảng xếp hạng chính sau khi hoàn tất đủ năm fold và chạy test aggregation.

## Nguồn kết quả

- `artifacts/new_baselines/convnextv2_tiny_mask2former/test_quantitative.json`
- `artifacts/new_baselines/convnextv2_tiny_multitask/test_quantitative.json`
- `artifacts/new_baselines/deeplabv3_resnet50_multitask/test_quantitative.json`
- `artifacts/mask2former/mask2former_swin_tiny/test_quantitative.json`
- `training.md`
