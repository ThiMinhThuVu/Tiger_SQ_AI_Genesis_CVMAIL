# TIGER SQ-AI Challenge — Tasks, EDA, Label Samples và phương pháp hiện tại

**Cập nhật:** 16/07/2026 (UTC)  
**Phạm vi:** dữ liệu Part 1 hiện có trong workspace và toàn bộ artifact thí nghiệm đã lưu  
**Trạng thái:** báo cáo nội bộ; các điểm số bên dưới là local validation, không phải hidden-test/leaderboard score

## 1. Tóm tắt

TIGER SQ-AI là bài toán multi-task trên một frame RGB của phẫu thuật nội soi lồng ngực:

1. semantic segmentation chi tiết 31 lớp;
2. semantic segmentation gộp 16 lớp;
3. multi-label classification cho 14 lymph-node stations nhìn thấy trong frame.

Dữ liệu Part 1 gồm **140 frame từ 10 ca**, mỗi ca 14 frame. Bộ dữ liệu nhỏ và lệch lớp mạnh: 3/31 fine class không xuất hiện, `Left subclavian artery` chỉ xuất hiện trong 1 frame, trong khi background chiếm 37,90% tổng số pixel. Vì các frame cùng ca có ngữ cảnh rất giống nhau và có cả frame trùng pixel, việc chia train/validation bắt buộc phải theo **case**, không chia ngẫu nhiên theo ảnh.

Phương pháp có bằng chứng mạnh nhất hiện tại là multi-task **ResNet-34 + FPN, full fine-tuning, affine augmentation**, đạt **0,5530 ± 0,0627** trên 5-fold case-level CV. Tversky–Focal đạt điểm Fold 0 cao nhất là **0,6244**, nhưng chưa được chạy đủ 5 folds nên hiện mới là ứng viên cần xác nhận, chưa thay thế baseline chính.

## 2. Nguồn thông tin và mức độ xác minh

- Trang Data do challenge cung cấp: [Synapse wiki 639942](https://www.synapse.org/Synapse:syn74209386/wiki/639942).
- Trang mô tả challenge được repo tham chiếu: [Synapse wiki 639462](https://www.synapse.org/Synapse:syn74209386/wiki/639462).
- Gói dữ liệu đã tải từ Synapse: [`data/readme.md`](data/readme.md), [`data/labelmap.csv`](data/labelmap.csv), [`data/lymph_node_station_visibility.csv`](data/lymph_node_station_visibility.csv) và [`data/manifest.csv`](data/manifest.csv).
- Kết quả thí nghiệm mới nhất: [`SCORE_REPORT_ALL_VERSIONS.md`](SCORE_REPORT_ALL_VERSIONS.md) và các file `history.json` dưới `artifacts/`.
- Code pipeline: [`train.py`](train.py), [`tiger_baseline/data.py`](tiger_baseline/data.py), [`tiger_baseline/model.py`](tiger_baseline/model.py), [`tiger_baseline/losses.py`](tiger_baseline/losses.py), [`tiger_baseline/metrics.py`](tiger_baseline/metrics.py).

Trang Synapse cần JavaScript/phiên đăng nhập nên nội dung wiki không được trả đầy đủ qua crawler. Các thông tin dữ liệu trong báo cáo được đối chiếu trực tiếp với README, label map, visibility CSV và Synapse entity IDs trong manifest đã tải về. EDA được tính lại trên toàn bộ file local, không lấy lại con số từ báo cáo cũ.

## 3. Các task

| Task | Input | Ground truth / output cần dự đoán | Metric được mô tả trong repo |
|---|---|---|---|
| **Task 1 — Fine segmentation** | 1 frame RGB | RGB mask gồm 31 fine classes | clinically weighted macro Dice ↑ và normalized Hausdorff Distance (nHD) ↓ |
| **Task 2 — Coarse segmentation** | 1 frame RGB | RGB mask gồm 16 merged classes | weighted macro Dice ↑ và nHD ↓ |
| **Task 3 — Station visibility** | 1 frame RGB | vector multi-hot 14 phần tử | macro F1 ↑ và AUROC ↑ |

Các class segmentation có `weight` từ 1 đến 3 trong `labelmap.csv`. Weight 3 dành cho các cấu trúc có độ ưu tiên cao như lymph node, pulmonary ligament, subclavian artery, vagal/recurrent laryngeal nerve và pulmonary artery. Vì vậy pixel accuracy không phù hợp: một model có thể đúng phần lớn background/vùng lớn nhưng vẫn bỏ sót cấu trúc nhỏ quan trọng.

### 3.1 Task 1 — Fine segmentation

- Mỗi pixel trong `masks_fine/*.png` là một mã RGB.
- Tra `(R,G,B)` trong `labelmap.csv` để nhận `fine_id` và `fine_name`.
- Có 31 ID liên tục từ 0 đến 30, bao gồm background và 30 class còn lại.

Ví dụ: `(255, 0, 204)` tương ứng `fine_id=19`, `Lymph node`, `weight=3`.

### 3.2 Task 2 — Coarse segmentation

Task 2 gộp các fine class có liên quan về cùng nhóm giải phẫu. Ví dụ:

| Fine classes | Coarse class |
|---|---|
| Trachea, Right/Left main bronchus | Respiratory Tract |
| Esophagus, Fatty tissue esophagus, Gastric conduit, Omentum | Gastroesophageal |
| Right/Left inferior pulmonary ligament, Pleura | Pleura |
| Right/Left subclavian artery, Right bronchial artery | Vessels |
| Right vagal nerve, Right/Left recurrent laryngeal nerve | Nerves |

Kiểm tra lại trên **274.176.000 pixel** cho thấy coarse mask chính thức khớp tuyệt đối với coarse mask suy ra từ `fine_id → merged_id`: **0 pixel sai khác**.

### 3.3 Task 3 — Lymph-node station visibility

Task 3 là multi-label classification, không phải single-label classification. Tên station trong filename/column `annotated frame` là station mục tiêu khi frame được chọn, nhưng một frame có thể đồng thời nhìn thấy nhiều station.

Thứ tự 14 output trong code hiện tại:

```text
6L, 6R, 7L, 7R, 8, 9, 10L, 10R, 11L, 11R, 12L, 12R, 13L, 13R
```

Ví dụ CSV:

```csv
center_1_case_13,10L,"6L, 7L, 8, 10L"
```

Target đúng là vector multi-hot có bốn giá trị 1 tại `6L`, `7L`, `8`, `10L`; không được chỉ dùng `10L` làm label.

## 4. Tổng quan dữ liệu

| Thuộc tính | Kết quả kiểm tra |
|---|---:|
| Số case | 10 (`center_1_case_6` đến `center_1_case_15`) |
| Frame mỗi case | 14 |
| Tổng frame | 140 |
| Fine masks | 140 |
| Coarse masks | 140 |
| Dòng visibility label | 140 |
| 1920×1080 | 126 frame (90%) |
| 1280×720 | 14 frame (10%) |
| Tỷ lệ khung hình | 16:9 ở cả hai resolution |
| Màu fine mask không có trong label map | 0 pixel |
| Fine → coarse không khớp | 0 / 274.176.000 pixel |

Cấu trúc một sample:

```text
data/images/<case>_<annotated_station>.png
data/masks_fine/<case>_<annotated_station>.png
data/masks_coarse/<case>_<annotated_station>.png
```

Tên file trên local được chuẩn hóa thành chữ thường; tên entity trong Synapse manifest có thể dùng hậu tố station viết hoa như `10L`.

## 5. Sample để đọc label

### 5.1 Sample nhiều cấu trúc: `center_1_case_10_9`

| Ảnh RGB | Fine mask — 31-class palette | Coarse mask — 16-class palette |
|---|---|---|
| ![RGB sample case 10 station 9](data/images/center_1_case_10_9.png) | ![Fine mask sample case 10 station 9](data/masks_fine/center_1_case_10_9.png) | ![Coarse mask sample case 10 station 9](data/masks_coarse/center_1_case_10_9.png) |

- Annotated frame: `9`.
- Visibility labels: `9, 13R, 10R, 11R`.
- Fine mask có 20 class, phù hợp để xem cách palette biểu diễn đồng thời airway, esophagus, pleura, heart/vessels, lung, lymph node, fatty tissue và vùng resection.

### 5.2 Sample chứa class cực hiếm: `center_1_case_13_10l`

| Ảnh RGB | Fine mask | Coarse mask |
|---|---|---|
| ![RGB sample rare class](data/images/center_1_case_13_10l.png) | ![Fine mask rare class](data/masks_fine/center_1_case_13_10l.png) | ![Coarse mask rare class](data/masks_coarse/center_1_case_13_10l.png) |

- Annotated frame: `10L`.
- Visibility labels: `6L, 7L, 8, 10L`.
- Đây là **sample duy nhất** trong Part 1 có `Left subclavian artery` (`fine_id=21`, RGB `255,136,178`, weight 3).
- Sample còn có các class ưu tiên cao `Lymph node`, `Pulmonary artery`, `Left recurrent laryngeal nerve`.

### 5.3 Sample có số visibility label tối đa: `center_1_case_15_7l`

| Ảnh RGB | Fine mask | Coarse mask |
|---|---|---|
| ![RGB sample multilabel](data/images/center_1_case_15_7l.png) | ![Fine mask multilabel](data/masks_fine/center_1_case_15_7l.png) | ![Coarse mask multilabel](data/masks_coarse/center_1_case_15_7l.png) |

- Annotated frame: `7L`.
- Visibility labels: `7L, 8, 10L, 11L, 13L` — năm positive labels, là mức lớn nhất quan sát được.
- Fine mask có 18 class, trong đó có `Left inferior pulmonary ligament`, `Lymph node` và `Pulmonary artery`.

Lưu ý: màu mask là **mã class chính xác**, không phải heatmap và không biểu diễn xác suất. Khi train phải decode RGB bằng bảng màu; khi submission phải encode prediction trở lại đúng RGB theo label map.

## 6. EDA segmentation labels

Tỷ lệ pixel được tính trên toàn bộ 274.176.000 pixel. `Frames` là số frame có ít nhất một pixel của class.

| ID | Fine class | Weight | Frames | Frame % | Pixel % |
|---:|---|---:|---:|---:|---:|
| 0 | Background | 1 | 140 | 100,00 | 37,9032 |
| 1 | Instrument | 1 | 137 | 97,86 | 5,4384 |
| 2 | Other | 1 | 85 | 60,71 | 0,1684 |
| 3 | Trachea | 2 | 80 | 57,14 | 3,0570 |
| 4 | Right main bronchus | 2 | 43 | 30,71 | 1,0674 |
| 5 | Left main bronchus | 2 | 68 | 48,57 | 1,5660 |
| 6 | Esophagus | 2 | 135 | 96,43 | 7,0562 |
| 7 | Fatty tissue esophagus | 1 | 115 | 82,14 | 5,1364 |
| 8 | Right inferior pulmonary ligament | 3 | 36 | 25,71 | 0,9659 |
| 9 | Left inferior pulmonary ligament | 3 | 22 | 15,71 | 0,2884 |
| 10 | Pleura | 2 | 119 | 85,00 | 9,8928 |
| 11 | Pericardium | 2 | 82 | 58,57 | 3,2793 |
| 12 | Inferior pulmonary vein | 2 | 57 | 40,71 | 1,4572 |
| 13 | Right subclavian artery | 3 | 13 | 9,29 | 0,2293 |
| 14 | Right vagal nerve | 3 | 35 | 25,00 | 0,5295 |
| 15 | Aorta | 2 | 95 | 67,86 | 4,7878 |
| 16 | Azygos vein | 2 | 82 | 58,57 | 0,9427 |
| 17 | Superior caval vein | 2 | 26 | 18,57 | 0,7363 |
| 18 | Lung | 1 | 88 | 62,86 | 5,7972 |
| 19 | Lymph node | 3 | 101 | 72,14 | 1,7999 |
| 20 | Fatty tissue | 1 | 118 | 84,29 | 3,2890 |
| 21 | Left subclavian artery | 3 | **1** | **0,71** | **0,0163** |
| 22 | Right bronchial artery | 3 | **0** | **0,00** | **0,0000** |
| 23 | Pulmonary artery | 3 | 25 | 17,86 | 0,2235 |
| 24 | Pool of blood | 1 | 68 | 48,57 | 0,9579 |
| 25 | Resection area | 1 | 93 | 66,43 | 2,9963 |
| 26 | Gastric conduit | 1 | **0** | **0,00** | **0,0000** |
| 27 | Right recurrent laryngeal nerve | 3 | 15 | 10,71 | 0,0621 |
| 28 | Left recurrent laryngeal nerve | 3 | 24 | 17,14 | 0,2912 |
| 29 | Omentum | 1 | **0** | **0,00** | **0,0000** |
| 30 | Thoracic duct | 1 | 12 | 8,57 | 0,0644 |

### 6.1 Nhận xét chính từ fine labels

- Ba class vắng hoàn toàn: `Right bronchial artery`, `Gastric conduit`, `Omentum`. Model không thể học supervised signal cho các class này từ Part 1.
- `Left subclavian artery` chỉ có 1 frame và 0,0163% pixel. Một case split có thể khiến class chỉ nằm ở train hoặc validation.
- Các class weight 3 nhưng rất ít pixel gồm `Right recurrent laryngeal nerve` (0,0621%), `Pulmonary artery` (0,2235%), `Right subclavian artery` (0,2293%) và `Left recurrent laryngeal nerve` (0,2912%).
- `Lymph node` có mặt ở 101 frame nhưng chỉ chiếm 1,80% pixel; prevalence theo frame không đồng nghĩa vùng mask lớn.
- Background + các vùng lớn chi phối gradient nếu chỉ dùng CE thông thường. Cần Dice/Tversky, clinical weight hoặc sampling/loss được kiểm soát; tuy nhiên oversampling quá mạnh đã cho kết quả xấu trong ablation hiện tại.

### 6.2 Coarse-label prevalence

| ID | Coarse class | Frames | Frame % | Pixel % |
|---:|---|---:|---:|---:|
| 0 | Background | 140 | 100,00 | 37,9032 |
| 1 | Respiratory Tract | 103 | 73,57 | 5,6904 |
| 2 | Gastroesophageal | 136 | 97,14 | 12,1927 |
| 3 | Pleura | 126 | 90,00 | 11,1471 |
| 4 | Heart | 83 | 59,29 | 4,7365 |
| 5 | Vessels | 14 | 10,00 | 0,2456 |
| 6 | Nerves | 63 | 45,00 | 0,8828 |
| 7 | Aorta | 95 | 67,86 | 4,7878 |
| 8 | Azygos Vein | 82 | 58,57 | 0,9427 |
| 9 | Superior Caval Vein | 26 | 18,57 | 0,7363 |
| 10 | Lung | 88 | 62,86 | 5,7972 |
| 11 | Lymphatic Tissue | 101 | 72,14 | 1,7999 |
| 12 | Non-anatomical Other | 138 | 98,57 | 5,6068 |
| 13 | Fatty Tissue | 118 | 84,29 | 3,2890 |
| 14 | Pulmonary artery | 25 | 17,86 | 0,2235 |
| 15 | Anatomical Other | 123 | 87,86 | 4,0185 |

Coarse labels cân bằng hơn fine labels nhưng vẫn có long tail rõ rệt: `Pulmonary artery` và `Vessels` chỉ chiếm khoảng 0,22–0,25% pixel.

## 7. EDA Task 3 labels

Tổng cộng có 417 positive station labels trên 140 frame, trung bình **2,98 labels/frame**.

| Số positive labels trong frame | Số frame |
|---:|---:|
| 1 | 8 |
| 2 | 35 |
| 3 | 56 |
| 4 | 34 |
| 5 | 7 |

| Station | Positive frames | Prevalence |
|---|---:|---:|
| 6L | 30 | 21,43% |
| 6R | 26 | 18,57% |
| 7L | 40 | 28,57% |
| 7R | 28 | 20,00% |
| 8 | 33 | 23,57% |
| 9 | 23 | 16,43% |
| 10L | 37 | 26,43% |
| 10R | 31 | 22,14% |
| 11L | 25 | 17,86% |
| 11R | 28 | 20,00% |
| 12L | 25 | 17,86% |
| 12R | 25 | 17,86% |
| 13L | 31 | 22,14% |
| 13R | 35 | 25,00% |

Task 3 cân bằng hơn Task 1: prevalence nằm trong khoảng 16,43–28,57%. Dù vậy threshold 0,5 chung cho tất cả station chưa chắc tối ưu; threshold phải được calibrate từ out-of-fold predictions, không fit trên hidden test hoặc trên chính fold dùng để báo cáo cuối.

## 8. Kiểm tra chất lượng và rủi ro dữ liệu

### 8.1 Tính toàn vẹn

- Tập tên file của images, fine masks và coarse masks khớp đầy đủ.
- Mỗi image có đúng một visibility row.
- Annotated station luôn nằm trong danh sách visible stations.
- Không có pixel RGB ngoài `labelmap.csv`.
- Fine-to-coarse mapping khớp 100% với coarse masks phát hành.

### 8.2 Frame trùng pixel

Có bốn cặp ảnh RGB trùng pixel hoàn toàn:

```text
center_1_case_10_6r.png = center_1_case_10_7r.png
center_1_case_11_6l.png = center_1_case_11_7l.png
center_1_case_12_6r.png = center_1_case_12_7r.png
center_1_case_15_6r.png = center_1_case_15_7r.png
```

Các cặp đều nằm trong cùng case, nên case-level split giữ chúng cùng phía train hoặc validation. Random frame split có thể đưa hai bản của cùng frame sang hai phía và gây leakage trực tiếp.

### 8.3 Các rủi ro chính

| Rủi ro | Tác động | Kiểm soát hiện tại / đề xuất |
|---|---|---|
| Chỉ 10 case | Variance giữa fold lớn, dễ overfit | 5-fold case-level CV; báo cáo mean ± SD |
| 3 class không có ground truth | Không học được supervised signal; metric class có thể undefined | Xử lý absent class rõ ràng trong metric; cân nhắc public external data hợp lệ |
| Class nhỏ/hiếm | Dice bằng 0, boundary khó học | Tversky/Focal/clinical weights; per-class audit |
| Duplicate frames | Tăng trọng số ngầm và gây leakage nếu split sai | Group theo case; cân nhắc deduplicate/weight trong ablation riêng |
| Hai resolution | Batch native khó và batch size nhỏ | Resized 512×896 là baseline; native resolution đã ablate riêng |
| Laterality L/R | Horizontal flip có thể làm sai class/station | H-flip mặc định tắt; chỉ dùng khi swap đầy đủ fine labels và station labels |
| Visibility là image-level | Crop có thể xóa station nhưng label vẫn positive | Augmentation hiện tại không crop |
| Credential trong downloader | Rò rỉ quyền Synapse | Thu hồi token, xóa khỏi source/history, đọc từ secret/env; không ghi token trong report |

## 9. Phương pháp tiếp cận hiện tại

### 9.1 Kiến trúc multi-task

```text
RGB frame
  └── shared encoder (primary: ImageNet ResNet-34)
       └── FPN-style decoder/features
            ├── fine head:       31-class segmentation
            ├── coarse head:     16-class segmentation
            ├── boundary head:   auxiliary boundary prediction
            └── visibility head: 14 independent logits
```

Thiết kế dùng encoder chung để tận dụng tín hiệu giữa nhận diện giải phẫu và visibility. Coarse head được học trực tiếp thay vì chỉ suy ra hậu xử lý từ fine prediction; điều này tạo thêm supervision ở mức nhóm class. Boundary head là surrogate hỗ trợ metric khoảng cách nHD.

### 9.2 Data split và preprocessing

- 5 folds theo case, mỗi fold validation gồm 2 case/28 frame; không có case overlap.
- Các validation pair theo implementation hiện tại:

| Fold | Validation cases |
|---:|---|
| 0 | case 10, case 15 |
| 1 | case 11, case 6 |
| 2 | case 12, case 7 |
| 3 | case 13, case 8 |
| 4 | case 14, case 9 |

- Baseline resize về 512×896, batch size 4.
- Photometric augmentation: brightness, contrast và color mức nhẹ.
- Spatial augmentation đang dùng: rotation ±10°, translation ±5%, zoom 0,90–1,10; không crop.
- Horizontal flip mặc định tắt vì laterality; code có mapping swap nhưng cần đánh giá như một ablation riêng.

### 9.3 Loss và optimization

Baseline dùng:

- clinically weighted cross-entropy + soft Dice cho fine/coarse segmentation;
- BCE cho boundary head;
- weighted BCEWithLogits cho 14 visibility labels, với `pos_weight` tính từ training fold;
- AdamW, cosine learning-rate schedule, encoder learning rate thấp hơn heads;
- full fine-tuning là strategy chính sau initial screening.

Các fine loss đã thử gồm baseline, Generalized Dice–Focal, Dice–Focal, Tversky–Focal và Dice–Hausdorff. Tversky–Focal hiện là ứng viên screening tốt nhất nhưng cần complete 5-fold CV.

### 9.4 Metric local dùng để chọn checkpoint

Repo đang dùng balanced surrogate:

```text
Task 1 = (Fine Dice + 1 - Fine nHD) / 2
Task 2 = (Coarse Dice + 1 - Coarse nHD) / 2
Task 3 = (Macro F1 + AUROC) / 2
Overall = (Task 1 + Task 2 + Task 3) / 3
```

Đây là cách nhóm sáu metric để chọn checkpoint và so sánh nội bộ, **không được gọi là official challenge score** nếu chưa xác nhận công thức aggregate chính thức.

## 10. Kết quả hiện tại

### 10.1 Bằng chứng chính — complete 5-fold CV, 100 epochs

| Rank | Setup | Overall mean ± SD | Task 1 | Task 2 | Task 3 |
|---:|---|---:|---:|---:|---:|
| 1 | **ResNet-34 `affine_full`** | **0,5530 ± 0,0627** | **0,3744** | 0,4897 | **0,7948** |
| 2 | `gdf_affine_frozen_bn` | 0,5478 ± 0,0605 | 0,3496 | **0,5106** | 0,7832 |
| 3 | `monai_gdf_affine_no_boundary` | 0,5355 ± 0,0591 | 0,3269 | 0,4888 | 0,7909 |

| Setup | Fine Dice ↑ | Fine nHD ↓ | Coarse Dice ↑ | Coarse nHD ↓ | F1 ↑ | AUROC ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **`affine_full`** | **0,3418** | **0,5930** | 0,4664 | 0,4870 | **0,6833** | 0,9062 |
| `gdf_affine_frozen_bn` | 0,3094 | 0,6102 | **0,4795** | **0,4582** | 0,6535 | **0,9130** |
| `monai_gdf_affine_no_boundary` | 0,2855 | 0,6317 | 0,4544 | 0,4768 | 0,6738 | 0,9080 |

Kết luận hiện tại: giữ `affine_full` làm primary candidate vì là setup đứng đầu có đủ 5 folds. Khoảng cách với hạng hai chỉ 0,0052, nhỏ hơn nhiều so với biến thiên giữa folds, nên chưa có bằng chứng về khác biệt có ý nghĩa thống kê.

### 10.2 Các screening result chỉ trên Fold 0

| Experiment | Fold-0 score | Diễn giải |
|---|---:|---|
| **Tversky–Focal + ResNet-34 + affine** | **0,6244** | Best completed Fold-0 run; cần 5-fold CV |
| SurgeNet-Public CaFormer-S18 full + affine | 0,6230 | Gần ngang ResNet-34; chưa đủ folds |
| ResNet-34 `affine_full` | 0,6212 | Fold-0 reference |
| Dice–Hausdorff | 0,6211 | Partial 90/100 epochs, không rank như complete run |
| Dice–Focal | 0,6135 | Complete Fold 0 |

Fold 0 là fold thuận lợi: trong complete CV, `affine_full` đạt 0,6166 ở Fold 0 nhưng chỉ 0,4606 ở Fold 1. Vì vậy không dùng 0,6244 làm expected generalization score.

### 10.3 Các hướng chưa hiệu quả

- Targeted class-aware sampler giảm score từ 0,6244 xuống 0,5988 và không cứu được các target class đang Dice 0.
- Native-resolution run bị dừng sau epoch 63; best score 0,5517, thấp hơn resized control 0,5821.
- nnU-Net Fold-0 pilot có segmentation-task mean 0,2632 và không có Task 3; chưa cạnh tranh với multi-task model.
- ResNet-50 + GDFL + affine đạt 0,5301 trên Fold 0, nhưng vì nhiều yếu tố thay đổi cùng lúc nên không kết luận ResNet-50 tự thân kém hơn.

## 11. Đề xuất bước tiếp theo

1. Chạy **Tversky–Focal ResNet-34 + affine** trên đủ 5 case folds cùng protocol 100 epochs.
2. Giữ `affine_full` làm primary model cho tới khi candidate mới thắng trên complete CV.
3. Tổng hợp out-of-fold visibility predictions và calibrate 14 thresholds; báo cáo trước/sau calibration.
4. Audit per-class Dice/nHD trên đủ folds, ưu tiên class weight 3 và phân tích riêng Fold 1.
5. Thử ensemble có kiểm soát giữa `affine_full` và `gdf_affine_frozen_bn`, vì model thứ hai tốt hơn Task 2/AUROC.
6. Nếu thử lại rare-class handling, giảm sampling strength hoặc dùng loss/patch strategy mới; chỉ thay đổi một yếu tố mỗi ablation.
7. Hoàn thiện inference → RGB encoding → Docker và external-data disclosure; không download model/data lúc runtime.
8. Thu hồi ngay Synapse token đang hard-code trong `dl_data.py`, xóa khỏi Git history và chuyển sang secret/environment variable trước khi chia sẻ repo hoặc build image.

## 12. Kết luận

Bottleneck chính là fine segmentation cho các cấu trúc nhỏ và hiếm, không phải Task 3. Hướng multi-task hiện tại hợp lý và đã có baseline 5-fold tái lập được. Quyết định model nên dựa trên complete case-level CV và per-class clinical metrics; các kết quả Fold 0 cao hơn chỉ nên dùng để chọn candidate cho vòng validation tiếp theo.

