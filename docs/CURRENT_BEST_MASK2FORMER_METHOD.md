# Tổng quan method Mask2Former tốt nhất hiện tại

**Cập nhật:** 24/07/2026 (UTC)  
**Phạm vi:** kết quả local case-level 5-fold đã hoàn tất trong repository  
**Lưu ý:** các điểm số trong tài liệu là local held-out test, không phải hidden-test hoặc leaderboard score.

## 1. Kết luận nhanh

Method tốt nhất hiện tại dựa trên **Mask2Former với backbone Swin-Small**. Có hai cấu hình mạnh nhất tùy mục tiêu:

1. **Đủ cả ba task:** Mask2Former Swin-Small + visibility probe, đạt `selection_score` **0,7115 ± 0,0315**.
2. **Segmentation tốt nhất:** Mask2Former Swin-Small với clinical weighting, direct coarse head, hierarchical consistency và fine-class presence head, đạt `task12_surrogate` **0,7363 ± 0,0363**.

Hai kết quả này chưa phải một model end-to-end duy nhất: hệ thống ba task dùng segmentation checkpoint của Swin-Small baseline và một visibility probe riêng; nhánh improved segmentation chưa có Task 3 station-visibility head.

## 2. Bài toán

Một frame RGB nội soi được dùng để giải ba task:

| Task | Output | Metric |
|---|---|---|
| Task 1 | Fine semantic segmentation, 31 lớp | clinically weighted Dice và normalized Hausdorff Distance (nHD) |
| Task 2 | Coarse semantic segmentation, 16 lớp | weighted Dice và nHD |
| Task 3 | Multi-label visibility, 14 lymph-node stations | macro F1 và macro AUROC |

Metric tổng hợp được tính bằng:

```text
selection_score = mean(
    fine Dice,
    1 - fine nHD,
    coarse Dice,
    1 - coarse nHD,
    visibility macro F1,
    visibility macro AUROC
)
```

## 3. Kiến trúc nền: Mask2Former Swin-Small

### 3.1 Backbone và decoder

- Backbone: **Swin-Small**.
- Pretrained checkpoint: `facebook/mask2former-swin-small-coco-panoptic`.
- Input được resize về **512 × 896**.
- Mask2Former biểu diễn segmentation dưới dạng một tập mask query.
- Mỗi query dự đoán một class distribution và một mask probability.
- Semantic probability được dựng lại bằng cách kết hợp class probability với mask probability của toàn bộ query.
- Fine prediction có 31 lớp; coarse prediction có thể được suy ra chính xác từ ánh xạ cố định `fine_id → merged_id`.

### 3.2 Cấu hình train chung

| Thuộc tính | Giá trị |
|---|---:|
| Số fold | 5 |
| Split | theo case, tỷ lệ 7:1:2 |
| Epoch tối đa | 100 |
| Physical batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Seed | 2026 |
| Early stopping patience | 15 |
| Checkpoint selection | validation metric, không dùng test target |

Chia dữ liệu theo case là bắt buộc vì các frame trong cùng ca có ngữ cảnh rất giống nhau và repository đã ghi nhận một số frame trùng pixel. Chia ngẫu nhiên theo ảnh sẽ gây leakage và làm score lạc quan giả tạo.

## 4. Method segmentation tốt nhất

Experiment tốt nhất đã hoàn tất là:

```text
mask2former_swin_small_clinical_hier_presence_v2
```

Method mở rộng Mask2Former Swin-Small bằng bốn thành phần chính.

### 4.1 Clinical weighting

Clinical weight trong `labelmap.csv` được đưa vào:

- chi phí matching giữa query và ground-truth mask;
- classification loss;
- mask BCE loss;
- Dice loss.

Mục tiêu là tăng ảnh hưởng của các cấu trúc nhỏ nhưng quan trọng về lâm sàng, thay vì để background và các vùng giải phẫu lớn chi phối gradient.

### 4.2 Direct coarse head

Ngoài fine Mask2Former decoder, model có một convolution head dự đoán trực tiếp 16 coarse classes từ feature của pixel decoder.

```text
L_coarse = CrossEntropy(coarse_logits, coarse_target)
```

Coarse target được suy ra từ fine ground truth bằng mapping cố định. Trọng số loss hiện tại là **0,6**.

### 4.3 Hierarchical consistency

Fine probability được gom thành coarse probability theo mapping `fine → coarse`. Model ép coarse prediction trực tiếp và coarse prediction suy ra từ fine head phải nhất quán bằng symmetric KL divergence:

```text
L_hierarchy = 0.5 × [KL(P_direct || P_derived) + KL(P_derived || P_direct)]
```

Trọng số loss là **0,2**. Thành phần này truyền supervision giữa Task 1 và Task 2, đồng thời hạn chế các dự đoán fine/coarse mâu thuẫn nhau.

### 4.4 Fine-class presence head

Global pooled pixel-decoder feature được đưa qua một linear head để dự đoán class nào xuất hiện trong frame. Đây là **presence của 31 fine segmentation classes**, không phải 14 station labels của Task 3.

- Loss: multi-label BCE with logits.
- `pos_weight` được tính chỉ từ các mask train chưa augmentation của từng fold.
- `pos_weight` được chặn tối đa ở 8.
- Trọng số loss: **0,2**.
- Khi inference, presence probability điều chỉnh mềm fine semantic probability; không hard-delete class.

### 4.5 Objective tổng

```text
L = L_fine
  + 0.6 × L_coarse
  + 0.2 × L_hierarchy
  + 0.2 × L_presence
```

Khi inference, coarse output cuối là phép trộn giữa:

- coarse probability dự đoán trực tiếp;
- coarse probability được aggregate từ fine prediction.

Hệ số fusion hiện tại là `beta = 0,5`.

## 5. Kết quả segmentation

Các giá trị dưới đây là mean ± sample standard deviation trên năm held-out test folds.

| Method | Task 1 Fine Dice ↑ | Fine nHD ↓ | Task 2 Coarse Dice ↑ | Coarse nHD ↓ | Task 1+2 surrogate ↑ |
|---|---:|---:|---:|---:|---:|
| Mask2Former Swin-Small baseline | 0,7080 ± 0,0271 | 0,2206 ± 0,0280 | 0,6432 ± 0,0329 | 0,2645 ± 0,0346 | 0,7165 |
| Clinical weighting only | 0,7209 ± 0,0444 | 0,2092 ± 0,0444 | 0,6527 ± 0,0616 | 0,2564 ± 0,0552 | 0,7270 ± 0,0508 |
| **Clinical + hierarchy + presence v2** | **0,7329 ± 0,0358** | **0,1979 ± 0,0365** | **0,6607 ± 0,0412** | **0,2507 ± 0,0404** | **0,7363 ± 0,0363** |

`task12_surrogate` được định nghĩa là:

```text
mean(fine Dice, 1 - fine nHD, coarse Dice, 1 - coarse nHD)
```

So với Swin-Small baseline, improved v2:

- tăng Fine Dice khoảng **+0,0250**;
- giảm Fine nHD khoảng **0,0227**;
- tăng Coarse Dice khoảng **+0,0175**;
- giảm Coarse nHD khoảng **0,0139**;
- tăng Task 1+2 surrogate khoảng **+0,0197**.

Đây là method segmentation mạnh nhất đã có đủ kết quả 5-fold trong repository tại thời điểm cập nhật.

## 6. Hệ thống Mask2Former đủ ba task tốt nhất

Task 3 được bổ sung bằng một visibility probe huấn luyện trên feature của Mask2Former Swin-Small đã freeze:

```text
Swin encoder feature
→ global average pooling
→ LayerNorm
→ Linear(256)
→ GELU
→ Dropout(0.2)
→ Linear(14)
→ sigmoid
```

Visibility head dùng BCE with logits và `pos_weight` tính trên training split. Checkpoint được chọn bằng validation loss. Threshold **0,5** cho mọi station cho kết quả tốt hơn bản tune threshold từ pooled out-of-fold validation trong artifact hiện tại.

### 6.1 Kết quả hệ thống ba task

| Metric | Mean ± sample SD |
|---|---:|
| Fine Dice | 0,7080 ± 0,0271 |
| Fine nHD | 0,2206 ± 0,0280 |
| Coarse Dice | 0,6432 ± 0,0329 |
| Coarse nHD | 0,2645 ± 0,0346 |
| Visibility macro F1 | 0,5484 ± 0,0467 |
| Visibility macro AUROC | 0,8543 ± 0,0270 |
| **Selection score** | **0,7115 ± 0,0315** |

### 6.2 Kết quả từng fold

| Fold | Fine Dice | Fine nHD | Coarse Dice | Coarse nHD | Visibility F1 | Visibility AUROC | Selection score |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,7067 | 0,2129 | 0,6550 | 0,2498 | 0,5963 | 0,8703 | 0,7276 |
| 1 | 0,6669 | 0,2666 | 0,5987 | 0,3214 | 0,4818 | 0,8153 | 0,6625 |
| 2 | 0,7294 | 0,2040 | 0,6746 | 0,2408 | 0,5684 | 0,8849 | 0,7354 |
| 3 | 0,7355 | 0,1948 | 0,6687 | 0,2379 | 0,5761 | 0,8598 | 0,7346 |
| 4 | 0,7014 | 0,2247 | 0,6191 | 0,2728 | 0,5197 | 0,8412 | 0,6973 |

## 7. Baseline tốt nhất hiện tại nên được hiểu thế nào

- Nếu cần **một score chung cho đủ ba task**, baseline tốt nhất là **Mask2Former Swin-Small + visibility probe**, `selection_score = 0,7115 ± 0,0315`.
- Nếu tập trung vào **Task 1 và Task 2**, method tốt nhất là **Clinical + hierarchy + presence v2**, `task12_surrogate = 0,7363 ± 0,0363`.
- Không nên gán `selection_score` cho improved v2 vì model đó chưa dự đoán 14 station visibility labels.
- Không nên ghép metric của improved v2 với visibility probe cũ rồi gọi đó là một kết quả thực nghiệm; tổ hợp này cần được chạy và đánh giá lại trên đúng từng fold.

## 8. Hạn chế và hướng tiếp theo

1. Dữ liệu chỉ có 10 case nên variance giữa folds vẫn đáng kể, đặc biệt ở Fold 1.
2. Một số fine classes không xuất hiện hoặc xuất hiện cực ít; score có thể nhạy với absent-class convention.
3. Fine-class presence head không thay thế Task 3 station visibility head.
4. Bước hợp lý tiếp theo là train visibility probe trên frozen feature của improved v2, hoặc thêm station-visibility head và joint fine-tune end-to-end.
5. Chỉ sau khi chạy đủ năm held-out folds mới có thể xác nhận improved segmentation có nâng `selection_score` chung vượt 0,7115 hay không.

## 9. Artifact và code nguồn

- Config baseline: `configs/mask2former_swin_small.yaml`
- Config improved v2: `configs/mask2former_swin_small_clinical_hier_presence_v2.yaml`
- Mask2Former baseline trainer: `scripts/train_mask2former.py`
- Improved trainer: `scripts/train_mask2former_improved.py`
- Improved model/loss: `tiger_models/improved_mask2former.py`
- Visibility probe: `scripts/train_mask2former_visibility.py`
- Baseline segmentation result: `artifacts/mask2former/mask2former_swin_small/test_quantitative.json`
- Improved v2 result: `artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2/test_quantitative_r1.json`
- Full three-task result: `artifacts/mask2former/mask2former_swin_small_task3_base/test_quantitative.json`

