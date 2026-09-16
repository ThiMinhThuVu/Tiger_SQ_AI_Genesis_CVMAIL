# Báo cáo score training hiện tại

**Ngày kiểm tra:** 2026-07-22  
**Phạm vi:** các artifact/log đang có trong repository, chủ yếu dùng protocol case-level 5-fold và split 7:1:2.

## Tóm tắt nhanh

- Trong các model có đủ cả ba task, **ConvNeXt-V2 Tiny + Mask2Former** đang có test `selection_score` cao nhất: **0,6902 ± 0,0347**.
- ConvNeXt-V2 Tiny + FPN đạt **0,6862 ± 0,0348**; DeepLabV3-ResNet50 đạt **0,6630 ± 0,0604**.
- SAM2 Hiera Small cũng tốt nhất trong nhóm SAM2 về fine Dice, coarse Dice và nHD trên test.
- **Mask2Former Swin-Tiny** có segmentation test tốt nhất trong các artifact hiện tại: fine Dice **0,7061 ± 0,0435**, coarse Dice **0,6401 ± 0,0624**.
- Bản hậu xử lý Task 1 tuned v1 đạt fine Dice **0,7070 ± 0,0422**, nhưng mức tăng chỉ **0,00083**, fine nHD xấu nhẹ và chỉ thắng 2/5 fold; chưa thay baseline gốc.
- Các baseline Mask2Former Swin/ResNet cũ không có visibility head; các bản ConvNeXt-V2 + Mask2Former mới đã bổ sung head 14 nhãn và có thể tính `selection_score` đầy đủ.
- Các score dưới đây là **local CV/test score**, không phải hidden-test hoặc leaderboard score.

## 1. Score tổng hợp theo 5-fold

Các số trong bảng là **mean ± sample SD**. `selection_score` tổng hợp từ ba task và được báo cáo riêng cho validation/test.

| Model | Val selection score | Test selection score | Fine Dice test | Fine nHD test | Coarse Dice test | Coarse nHD test | Visibility F1 test | Visibility AUROC test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ConvNeXt-V2 Tiny + Mask2Former** | **0,7289 ± 0,0393** | **0,6902 ± 0,0347** | **0,6691 ± 0,0522** | **0,2505 ± 0,0399** | **0,5717 ± 0,0708** | 0,3103 ± 0,0522 | 0,5659 ± 0,0723 | **0,8952 ± 0,0300** |
| ConvNeXt-V2 Tiny + FPN | 0,7182 ± 0,0589 | 0,6862 ± 0,0348 | 0,6470 ± 0,0333 | 0,2472 ± 0,0338 | 0,5624 ± 0,0376 | **0,2992 ± 0,0417** | 0,5673 ± 0,0461 | 0,8871 ± 0,0335 |
| SAM2 Hiera Small | 0,7113 ± 0,0590 | 0,6646 ± 0,0403 | 0,6052 ± 0,0437 | 0,2832 ± 0,0459 | 0,5204 ± 0,0598 | 0,3222 ± 0,0676 | 0,5735 ± 0,0688 | 0,8936 ± 0,0404 |
| DeepLabV3-ResNet50 | 0,7085 ± 0,0550 | 0,6630 ± 0,0604 | 0,6266 ± 0,0272 | 0,2604 ± 0,0339 | 0,5426 ± 0,0452 | 0,3027 ± 0,0508 | 0,5166 ± 0,1444 | 0,8552 ± 0,0757 |
| SAM2 Hiera Tiny | 0,7011 ± 0,0372 | 0,6589 ± 0,0443 | 0,5984 ± 0,0252 | 0,2863 ± 0,0354 | 0,5026 ± 0,0451 | 0,3352 ± 0,0500 | **0,5862 ± 0,0946** | 0,8880 ± 0,0364 |
| SAM2 Hiera Base+ | 0,6516 ± 0,0745 | 0,6087 ± 0,0296 | 0,5951 ± 0,0291 | 0,2926 ± 0,0297 | 0,4932 ± 0,0279 | 0,3391 ± 0,0429 | 0,4459 ± 0,0572 | 0,7499 ± 0,0389 |

### Nhận xét SAM2

- Hiera Small cao hơn Hiera Tiny khoảng **0,0056** ở test `selection_score` và cao hơn Base+ khoảng **0,0559**.
- Hiera Small có test fine Dice cao nhất (**0,6052**) và coarse Dice cao nhất (**0,5204**) trong nhóm SAM2.
- Hiera Tiny có visibility macro F1 trung bình cao hơn Small (**0,5862** so với **0,5735**), nhưng độ lệch giữa các fold lớn hơn.
- Fold 3 là fold khó nhất đối với cả ba biến thể; riêng Hiera Small có validation selection score theo fold từ **0,6152 đến 0,7718**.

## 2. Score từng fold của SAM2

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| SAM2 Hiera Tiny | 0,7234 | 0,7057 | 0,7161 | 0,6360 | 0,7245 |
| SAM2 Hiera Small | 0,7158 | 0,7108 | **0,7427** | 0,6152 | **0,7718** |
| SAM2 Hiera Base+ | 0,6632 | 0,6808 | 0,6636 | 0,5260 | 0,7244 |

Đây là validation `selection_score` tại checkpoint tốt nhất của từng fold, không phải test score. Chi tiết đầy đủ nằm trong [`reports/training_analysis/score_by_fold.csv`](reports/training_analysis/score_by_fold.csv).

## 3. Mask2Former: segmentation test

Các baseline Mask2Former Swin/ResNet trong bảng này chỉ có segmentation head, không có Task 3 visibility head. Vì vậy bảng này chỉ dùng để so sánh các metric segmentation; không gán score tổng hợp giả.

| Model | Fine Dice | Fine nHD | Coarse Dice | Coarse nHD | Fine pixel accuracy |
|---|---:|---:|---:|---:|---:|
| **Mask2Former Swin-Tiny** | **0,7061 ± 0,0435** | **0,2177 ± 0,0414** | **0,6401 ± 0,0624** | **0,2564 ± 0,0577** | **0,7622 ± 0,0780** |
| Mask2Former ResNet-50 official | 0,6375 ± 0,0153 | 0,2624 ± 0,0232 | 0,5234 ± 0,0338 | 0,3238 ± 0,0421 | 0,6144 ± 0,1179 |
| Mask2Former ResNet-50 | 0,6267 ± 0,0203 | 0,2663 ± 0,0198 | 0,4987 ± 0,0291 | 0,3386 ± 0,0445 | 0,5728 ± 0,1253 |

Swin-Tiny đang là ứng viên segmentation mạnh nhất về Dice và nHD. Tuy nhiên, cần bổ sung visibility head và chạy lại evaluation đầy đủ nếu muốn đưa model này vào bảng xếp hạng `selection_score` chung.

### Task 1 tuned v1 — kết quả 5-fold

| Version | Fine Dice | Fine nHD | Surrogate `(Dice + 1 - nHD) / 2` |
|---|---:|---:|---:|
| Swin-Tiny baseline | 0,7061 ± 0,0435 | **0,2177 ± 0,0414** | 0,74421 ± 0,04231 |
| Swin-Tiny tuned v1 | **0,7070 ± 0,0422** | 0,2178 ± 0,0405 | **0,74458 ± 0,04122** |

Tuned v1 tăng surrogate trung bình **+0,00037 ± 0,00148**; 2 fold tăng, 1 fold không đổi và 2 fold giảm. Vì mức tăng nhỏ, không ổn định và chưa đạt tiêu chí thắng 4/5 fold, báo cáo vẫn giữ Swin-Tiny gốc làm baseline chính. Surrogate là chỉ số tuning nội bộ, không phải `selection_score` hay official score.

## 4. Trạng thái các baseline khác

- **ConvNeXt-V2 Base + Mask2Former:** job 136 đang chạy; fold 0 và 1 đã hoàn tất. Chưa báo cáo test 5-fold cho tới khi đủ cả năm fold.
- **DINOv2 ViT-S/14:** test 5-fold đã hoàn tất; fine Dice **0,6533 ± 0,0188**, fine nHD **0,2407 ± 0,0244**.
- **MedSAM ViT-B:** test 5-fold đã hoàn tất; fine Dice **0,5049 ± 0,0135**, fine nHD **0,3512 ± 0,0208**.
- **nnU-Net 2D:** test 5-fold đã hoàn tất; fine Dice **0,4682 ± 0,0646**, fine nHD **0,3769 ± 0,0664**.
- Cả ba baseline trên hiện là segmentation-only nên `selection_score` là **N/A**.

## 5. Kết luận và thứ tự ưu tiên

1. **Ứng viên đa nhiệm hiện tại:** ConvNeXt-V2 Tiny + Mask2Former, test `selection_score` **0,6902 ± 0,0347**.
2. **Ứng viên segmentation hiện tại:** Mask2Former Swin-Tiny, fine Dice **0,7061 ± 0,0435** và coarse Dice **0,6401 ± 0,0624**.
3. ConvNeXt-V2 Tiny + Mask2Former đứng thứ hai về Fine Dice Task 1 trong các artifact hiện tại, sau Mask2Former Swin-Tiny segmentation-only.
4. Không gọi các số liệu local trong báo cáo này là official challenge score hoặc leaderboard score.

## Nguồn số liệu

- [`reports/training_analysis/score_summary.csv`](reports/training_analysis/score_summary.csv)
- [`reports/training_analysis/score_by_fold.csv`](reports/training_analysis/score_by_fold.csv)
- [`reports/sam2_hiera_small_partial_7-1-2_boundary_100.md`](reports/sam2_hiera_small_partial_7-1-2_boundary_100.md)
- [`artifacts/mask2former/mask2former_swin_tiny/test_quantitative.json`](artifacts/mask2former/mask2former_swin_tiny/test_quantitative.json)
- [`artifacts/mask2former/mask2former_swin_tiny_task1_tuned_v1/test_quantitative.json`](artifacts/mask2former/mask2former_swin_tiny_task1_tuned_v1/test_quantitative.json)
- [`artifacts/new_baselines/dinov2_vits14/test_quantitative.json`](artifacts/new_baselines/dinov2_vits14/test_quantitative.json)
- [`artifacts/mask2former/mask2former_resnet50_official/test_quantitative.json`](artifacts/mask2former/mask2former_resnet50_official/test_quantitative.json)
- [`artifacts/mask2former/mask2former_resnet50/test_quantitative.json`](artifacts/mask2former/mask2former_resnet50/test_quantitative.json)
- [`artifacts/new_baselines/convnextv2_tiny_multitask/test_quantitative.json`](artifacts/new_baselines/convnextv2_tiny_multitask/test_quantitative.json)
- [`artifacts/new_baselines/convnextv2_tiny_mask2former/test_quantitative.json`](artifacts/new_baselines/convnextv2_tiny_mask2former/test_quantitative.json)
- [`artifacts/new_baselines/deeplabv3_resnet50_multitask/test_quantitative.json`](artifacts/new_baselines/deeplabv3_resnet50_multitask/test_quantitative.json)
