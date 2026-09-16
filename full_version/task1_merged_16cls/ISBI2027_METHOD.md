# Coarse-only Mask2Former cho TIGER Task 1 — mô tả phương pháp hướng tới ISBI 2027

## Material Passport

| Trường | Giá trị |
|---|---|
| Phạm vi | Phương pháp và kết quả nội bộ hiện có của Task 1 merged segmentation |
| Dataset snapshot | 40 ca, 524 frame, 6 center |
| Experiment | `coarse_only_swin_small_seed2026` |
| Split | 5-fold theo ca, seed 2026 |
| Trạng thái bằng chứng | Hoàn tất 5/5 fold và held-out local-test OOF |
| Hidden challenge test | Chưa thực hiện |
| Mục tiêu tài liệu | Method note và nền tảng phát triển bài ISBI 2027 |
| Citation status | Chưa thực hiện literature/citation verification; không phải bản manuscript sẵn sàng nộp |

## 1. Tóm tắt phương pháp hiện tại

Phương pháp hiện tại là một baseline semantic segmentation đơn nhiệm cho Task 1 theo numbering chính thức của TIGER SQ-AI. Mô hình nhận một frame RGB và dự đoán trực tiếp mask gồm 16 lớp merged. Khác với pipeline trước, đầu ra Task 1 không được tạo bằng cách dự đoán 31 lớp fine của Task 2 rồi ánh xạ `fine → merged`. Mô hình coarse-only có class head 16 lớp riêng, được tối ưu chỉ bằng ground truth trong `data/masks_coarse`, không load checkpoint Task 2 và không sử dụng fine logits ở train hoặc inference.

![Coarse-only Mask2Former method](assets/figure_01_coarse_only_method.svg)

**Figure 1. Coarse-only single-task segmentation for official TIGER Task 1.** The network is trained directly against 16-class merged masks. Task-2 checkpoints, fine-label predictions, and fine-to-merged post-processing are explicitly excluded. The diagram is provided as vector SVG for paper editing.

## 2. Research question và vai trò trong bài ISBI

Research question có thể kiểm chứng từ thí nghiệm hiện tại là:

> Khi giữ nguyên backbone, protocol chia fold và evaluator, việc học trực tiếp ontology merged 16 lớp có tạo ra biểu diễn Task 1 tốt hơn so với suy Task 1 từ ontology fine 31 lớp hay không?

Kết quả hiện tại trả lời **không** đối với cấu hình baseline đã thử: coarse-only đạt Task-1 score `0.742988`, thấp hơn pipeline fine-derived `0.748411`. Tuy nhiên, thí nghiệm này vẫn có giá trị khoa học như một controlled comparison về mức độ phụ thuộc giữa hai ontology. Nó chưa đủ để tạo novelty cho một bài ISBI nếu chỉ được trình bày như “thay head 31 lớp bằng head 16 lớp”. Để trở thành contribution, nghiên cứu cần một cơ chế coarse-specific có giả thuyết rõ, ablation và đánh giá đa seed; các hướng được đề xuất ở Mục 10.

## 3. Dữ liệu và nhãn

Snapshot gồm 524 frame từ 40 ca tại 6 center. Mỗi frame có image RGB, fine mask 31 ID và merged mask 16 ID; coarse-only chỉ đọc image và merged mask. Các frame trong cùng một ca có tương quan mạnh, vì vậy đơn vị chia dữ liệu là `center_X_case_Y`, không phải frame.

Các ảnh có ba nhóm độ phân giải gốc: 1280×720, 1920×1080 và 3840×2160. Input được resize về 512×896 cho train và local evaluation. Mask được resize bằng nearest-neighbor để bảo toàn ID lớp. Image được chuẩn hóa bằng ImageNet mean/std.

### 3.1 Augmentation

Train-time augmentation gồm affine ngẫu nhiên với xác suất 0.75 (góc ±10°, translation ±5%, scale 0.90–1.10), brightness và contrast jitter với xác suất 0.75, cùng color jitter với xác suất 0.50. Mọi biến đổi hình học được áp dụng đồng bộ lên image và mask; mask luôn dùng nearest-neighbor interpolation.

## 4. Protocol chia fold và chống leakage

Năm outer fold được khóa bởi manifest `full40_case_folds_v1`. Mỗi fold có 28 ca train, 4 ca validation và 8 ca held-out local test. Không ca nào giao nhau giữa ba split trong cùng fold. Qua năm fold, mỗi ca xuất hiện đúng một lần trong test, tạo OOF coverage gồm 40 ca và 524 frame.

Validation chỉ được dùng để chọn checkpoint. Test fold không tham gia tối ưu tham số, early stopping hoặc threshold selection. Các metric cuối được tính theo thứ tự frame → trung bình trong ca → trung bình giữa các ca, tránh để ca có nhiều frame chi phối score.

## 5. Kiến trúc

### 5.1 Backbone

Encoder là Swin-Small khởi tạo từ checkpoint Mask2Former COCO panoptic. Các tham số backbone và decoder được fine-tune end-to-end. Việc dùng pretraining này phải được khai báo rõ trong submission và phần data/pretraining disclosure của manuscript.

### 5.2 Mask2Former decoder

Pixel decoder tạo multi-scale feature maps. Transformer decoder sử dụng learned queries và masked attention để sinh hai thành phần: class logits cho từng query và query-specific mask logits. Semantic probability của lớp `c` tại pixel `(h,w)` được tổng hợp từ class probability và mask probability của các query:

\[
p(c,h,w)=\sum_{q=1}^{Q} p_q(c)\,\sigma(m_q(h,w)).
\]

Class head có đúng 16 output classes cộng một no-object class nội bộ. Prediction cuối là `argmax` trên 16 semantic probability maps. Không tồn tại fine head 31 lớp trong mô hình này.

## 6. Hàm loss và tối ưu

Mô hình sử dụng objective chuẩn của Mask2Former:

\[
\mathcal{L}=2\mathcal{L}_{cls}+5\mathcal{L}_{mask}+5\mathcal{L}_{dice},
\]

trong đó `L_cls` là classification loss trên matched queries, `L_mask` là point-sampled sigmoid focal mask loss và `L_dice` là Dice loss. No-object weight là 0.1; mỗi bước lấy 12,544 điểm, oversample ratio 3.0 và importance-sampling ratio 0.75.

Optimizer là AdamW với learning rate `1e-4`, weight decay `1e-4`, physical batch size 4 và không gradient accumulation. Training tối đa 100 epoch. Early stopping dùng validation merged task score với patience 15 và minimum improvement `1e-4`. Effective seeds của năm fold là 2026–2030.

## 7. Inference và metric

Mỗi fold load checkpoint có validation Task-1 score cao nhất và chạy inference trên 8 held-out test cases. Hai metric thành phần là clinically weighted Dice và normalized Hausdorff distance (nHD). Task score được tính bằng:

\[
S_{T1}=\frac{Dice+(1-nHD)}{2}.
\]

Giá trị cao hơn tốt hơn. Báo cáo OOF là trung bình case-level trên 40 ca, không phải hidden challenge score.

## 8. Kết quả hiện tại

### 8.1 Held-out local-test OOF

| Fold | Test cases | Frames | Dice ↑ | nHD ↓ | Task-1 score ↑ |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 106 | 0.719905 | 0.200856 | 0.759524 |
| 1 | 8 | 102 | 0.682682 | 0.232289 | 0.725196 |
| 2 | 8 | 106 | 0.698233 | 0.221869 | 0.738182 |
| 3 | 8 | 104 | 0.695846 | 0.207040 | 0.744403 |
| 4 | 8 | 106 | 0.710989 | 0.215724 | 0.747633 |
| **OOF** | **40** | **524** | **0.701531** | **0.215556** | **0.742988** |

### 8.2 Controlled comparison với fine-derived Task 1

| Method | Task-2 dependency | Dice ↑ | nHD ↓ | Task-1 score ↑ |
|---|---|---:|---:|---:|
| Fine-derived baseline | Có | 0.705594 | 0.208773 | **0.748411** |
| **Coarse-only (current)** | **Không** | 0.701531 | 0.215556 | **0.742988** |
| Difference | — | -0.004062 | +0.006783 | **-0.005423** |

Coarse-only chỉ vượt fine-derived ở fold 2 (`+0.002049`) và thấp hơn ở bốn fold còn lại. Mức giảm lớn nhất xảy ra ở fold 3 (`-0.020024`). Kết quả cho thấy fine supervision có thể cung cấp regularization hoặc boundary detail hữu ích ngay cả khi endpoint là ontology merged. Đây là inference từ controlled result, chưa phải causal conclusion vì chưa có multi-seed uncertainty và chưa tách ảnh hưởng của capacity, matching dynamics hoặc checkpoint variance.

## 9. Điểm mạnh và giới hạn

### Điểm mạnh

- Task 1 độc lập thật sự ở cả training lẫn inference.
- Case-level split loại bỏ leakage giữa frame cùng ca.
- Five-fold OOF phủ toàn bộ 40 ca đúng một lần.
- Cùng backbone, input resolution và split manifest với comparator, giúp giảm confounding.
- Artifact ghi rõ `uses_task2_checkpoint=false` và `uses_fine_predictions=false`.

### Giới hạn cần công khai

- Chỉ có một base seed; năm effective fold seeds không thay thế multi-seed replication trên cùng fold.
- Dataset nhỏ và mất cân bằng theo center; chưa có confidence interval ở cấp ca.
- Model selection dùng cùng một metric đang báo cáo trên validation, nhưng final estimate dùng outer test nên không leakage; adaptive experiment selection qua nhiều candidate vẫn có thể gây optimistic bias.
- Evaluation hiện tại ở grid 512×896, chưa phải native-resolution evaluation.
- Chưa có hidden challenge test.
- Coarse-only hiện kém comparator và chưa cung cấp novelty thuật toán đủ mạnh cho ISBI.
- Tài liệu này chưa có verified literature review hoặc references; mọi novelty claim phải chờ systematic nearest-work search.

## 10. Lộ trình biến baseline thành đóng góp ISBI 2027

### 10.1 Hypothesis-driven method extension

Hướng ưu tiên là **ontology-aware coarse segmentation** thay vì chỉ tăng backbone hoặc resolution. Một candidate có thể kết hợp: (i) coarse query prototypes theo nhóm giải phẫu; (ii) boundary-aware loss tập trung vào interfaces giữa các merged organs; và (iii) center-robust feature normalization. Mỗi thành phần phải được thêm riêng để giữ ablation có thể diễn giải.

### 10.2 Minimum experiment matrix

| ID | Candidate | Mục đích |
|---|---|---|
| C0 | Coarse-only hiện tại | Baseline độc lập |
| C1 | C0 + class-balanced/query reweighting | Xử lý class imbalance |
| C2 | C0 + boundary-aware objective | Giảm nHD |
| C3 | C0 + ontology-aware prototypes | Tăng phân biệt merged anatomy |
| C4 | C0 + C2 + C3 | Kiểm tra tính bổ sung |
| F0 | Fine-derived baseline | Comparator phụ thuộc Task 2 |

Candidate chỉ nên được promote nếu tăng mean OOF Task-1 score, không làm xấu đồng thời Dice và nHD, không tạo center collapse, và hiệu ứng giữ được trên ít nhất ba base seeds. Cần báo cáo paired case-level bootstrap 95% CI và effect size; không diễn giải năm folds như năm mẫu bệnh nhân độc lập.

### 10.3 Paper claim boundary

Claim hiện tại có thể viết là: “Direct coarse-only learning provides a leakage-controlled independent baseline but does not outperform fine-derived supervision.” Không nên claim state of the art, superiority hoặc clinical robustness. Claim về novelty chỉ được mở sau khi literature search xác nhận nearest prior work và candidate mới vượt các gate định trước.

## 11. Reproducibility map

| Thành phần | Đường dẫn |
|---|---|
| Config | `configs/coarse_only_swin_small_seed2026.yaml` |
| Trainer/evaluator | `scripts/train_coarse_only.py` |
| Split manifest | `../task2_fine_31cls/splits/case_folds_v1.json` |
| Five-fold checkpoints | `artifacts/seed_2026/coarse_only_swin_small_seed2026/fold_*/best.pt` |
| OOF metrics | `artifacts/seed_2026/coarse_only_swin_small_seed2026/test_oof/oof_test_metrics.json` |
| Method figure | `assets/figure_01_coarse_only_method.svg` |

## 12. Candidate Methods paragraph bằng tiếng Anh

### Coarse-only segmentation

We developed a single-task Mask2Former baseline that predicts the 16 merged anatomical classes of the official TIGER Task 1 directly from an RGB frame. Unlike a fine-derived pipeline, the model neither loads a Task-2 checkpoint nor converts 31-class fine predictions into merged labels. Images were resized to 512×896 pixels and normalized using ImageNet statistics. A Swin-Small encoder initialized from a COCO-panoptic Mask2Former checkpoint was fine-tuned jointly with the pixel decoder and masked-attention transformer decoder. Training targets were read exclusively from the 16-class coarse masks. The objective combined query classification, point-sampled focal mask, and Dice losses with weights 2, 5, and 5, respectively. We used AdamW with a learning rate and weight decay of 10⁻⁴, a batch size of four, and early stopping after 15 epochs without an improvement of at least 10⁻⁴ in the validation Task-1 score.

We used five case-disjoint folds defined before model training. Each fold contained 28 training, four validation, and eight held-out test cases. Checkpoints were selected exclusively on the validation split and evaluated once on the corresponding held-out split. Across folds, every case contributed to the out-of-fold test estimate exactly once. Dice and normalized Hausdorff distance were first averaged across frames within each case and subsequently across the 40 cases. The Task-1 score was defined as the mean of Dice and one minus normalized Hausdorff distance.

---

## Artifact status

File này là method note dựa trên code và artifact hiện có. Nó chưa bao gồm Introduction, Related Work, verified citations, statistical confidence intervals, ethics/data governance statement hoặc hidden-test result cần cho manuscript ISBI hoàn chỉnh.
