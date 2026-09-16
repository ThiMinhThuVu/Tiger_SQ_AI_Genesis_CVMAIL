# Protocol tái lập và so sánh công bằng — TIGER SQ-AI

**Cập nhật:** 2026-07-17 (UTC)  
**Mục đích:** cố định dữ liệu, split, tiền xử lý, cách chọn checkpoint và metric để một phương pháp mới có thể được chạy lại và so sánh công bằng với phương pháp hiện tại.

## 1. Kết luận thực hành

Baseline đối chứng chính hiện tại là **`affine_full`**: multi-task ResNet-34 + FPN, encoder ImageNet-1K, full fine-tuning, affine augmentation và baseline loss. Đây là phương pháp duy nhất đã được so sánh hoàn chỉnh trên cùng **5 folds theo ca** trong 100 epochs. Kết quả local CV: **0.5530 ± 0.0627** (mean ± sample SD của `selection_score`).

Không có test set có nhãn riêng trong workspace. Vì vậy, trong báo cáo nội bộ:

- `validation` = 2 ca được giữ lại trong mỗi fold;
- `test` = hidden test/leaderboard của challenge (không có local label, không được dùng để chọn model, threshold hay hyperparameter);
- ước lượng performance công bằng = trung bình kết quả **out-of-fold (OOF) của 5 validation folds**.

Các kết quả Fold-0-only (ví dụ Tversky–Focal 0.6244) chỉ là screening, **không thay thế** kết quả 5-fold ở trên.

## 2. Môi trường và dấu vết dữ liệu

Tạo environment theo đúng phiên bản đã dùng:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Các package then chốt: PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`, MONAI `1.5.2`, NumPy `2.2.6`, SciPy `1.15.3`, scikit-learn `1.6.1`.

Dataset local cần có cấu trúc:

```text
data/
├── images/*.png                         # 140 RGB frames
├── masks_fine/*.png                     # 140 RGB fine masks
├── masks_coarse/*.png                   # chỉ dùng để kiểm tra/đối chiếu
├── labelmap.csv
└── lymph_node_station_visibility.csv
```

| Thành phần | Quy ước cố định |
|---|---|
| Cases | 10: `center_1_case_6` … `center_1_case_15` |
| Frames | 140, đúng 14 frame/case |
| Fine label | 31 ID liên tiếp `0..30`, mask PNG RGB được decode qua `labelmap.csv` |
| Coarse label | 16 lớp, **suy ra từ fine ID** qua cột `merged_id` của `labelmap.csv` |
| Visibility label | vector multi-hot 14 chiều từ CSV; thứ tự: `6L, 6R, 7L, 7R, 8, 9, 10L, 10R, 11L, 11R, 12L, 12R, 13L, 13R` |
| Hash label map | SHA-256 `3cd640283a0e7157f096d85e72961a6deb117380dfe413f906b98d09c2cf4671` |
| Hash visibility CSV | SHA-256 `be72045c2ea4d6d5a6ff44721482b107e5581d77560bba334c046c755eeb8434` |

Mỗi tên file chứa case ID, ví dụ `center_1_case_10_6L.png`. Không được chia ngẫu nhiên theo frame: các frame cùng một ca có ngữ cảnh rất tương quan và sẽ gây leakage.

## 3. Train / validation / test split chuẩn

Hàm split là `case_folds()` trong `tiger_baseline/data.py`: sắp xếp case theo thứ tự chữ và lấy `ordered[fold::5]` làm validation. Mọi frame của một case luôn ở cùng một partition.

| Fold | Validation cases (28 frames) | Train cases (112 frames) |
|---:|---|---|
| 0 | `case_10`, `case_15` | `case_6, 7, 8, 9, 11, 12, 13, 14` |
| 1 | `case_11`, `case_6` | `case_7, 8, 9, 10, 12, 13, 14, 15` |
| 2 | `case_12`, `case_7` | `case_6, 8, 9, 10, 11, 13, 14, 15` |
| 3 | `case_13`, `case_8` | `case_6, 7, 9, 10, 11, 12, 14, 15` |
| 4 | `case_14`, `case_9` | `case_6, 7, 8, 10, 11, 12, 13, 15` |

`case_` được lược bớt trong bảng cho gọn; ID đầy đủ có prefix `center_1_`.

**Quy tắc so sánh:** phương pháp mới phải chạy đủ 5 fold này, dùng đúng 112/28 frames ở mỗi fold. Không dùng validation folds của baseline để train thêm, và không chọn setup theo hidden test.

## 4. Input và tiền xử lý

### 4.1 Input của model

Mỗi mẫu là một RGB PNG gốc (1280×720 hoặc 1920×1080, tỷ lệ 16:9). Standard protocol resize ảnh và fine mask về **H×W = 512×896**:

- image: bilinear resize;
- segmentation mask: nearest-neighbor resize, sau khi decode RGB thành fine ID;
- coarse mask: map từ fine ID sau resize;
- tensor image: `float32`, chia `255`, sau đó ImageNet normalization: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.

Không crop. Điều này đặc biệt quan trọng vì Task 3 là nhãn ở mức toàn ảnh; crop có thể làm thay đổi việc một station có còn nhìn thấy hay không.

### 4.2 Augmentation khi train của đối chứng

`affine_full` dùng đúng các augmentation sau, chỉ trên train set:

| Biến đổi | Xác suất | Tham số |
|---|---:|---|
| Affine đồng thời ảnh + fine mask | 0.75 | rotation ±10°, translation mỗi chiều ±5%, scale 0.90–1.10; fill 0; ảnh bilinear, mask nearest |
| Brightness | 0.75 | factor 0.82–1.18 |
| Contrast | 0.75 | factor 0.85–1.15 |
| Color | 0.50 | factor 0.88–1.12 |
| Horizontal flip | 0.0 | tắt trong protocol chính |

Validation/test không augmentation. Nếu phương pháp mới cần augmentation khác, phải báo cáo đây là thay đổi phương pháp; không được thay split, resize hay metric đồng thời mà không có ablation.

## 5. Baseline hiện tại cần đối chứng

Kiến trúc là encoder chung ResNet-34 pretrained ImageNet-1K + FPN (160 channels), decoder với dropout 0.10 và bốn đầu ra:

1. fine logits: 31 channels;
2. coarse logits: 16 channels;
3. boundary logits: 31 channels, auxiliary only;
4. visibility logits: 14 channels, MLP `(... → 384 → 14)` với dropout 0.20.

Full fine-tuning: toàn bộ encoder và heads trainable. Các setting cố định:

| Nhóm | Giá trị |
|---|---|
| Epochs | 100 |
| Batch size / workers | 4 / 8 |
| Seed | 2026 |
| Optimizer | AdamW |
| LR heads / encoder | `3e-4` / `4.5e-5` (encoder scale 0.15) |
| Weight decay | `1e-4` |
| Scheduler | cosine annealing, `eta_min = 6e-6`, `T_max = 100` |
| Precision | CUDA bfloat16 autocast + GradScaler; TF32 enabled |
| Gradient clip | global norm 1.0 |
| Sampling baseline | uniform shuffled frames; mỗi epoch 112 draws |
| Eval | mỗi epoch; checkpoint chọn theo validation `selection_score` cao nhất |

Loss baseline:

```text
L = [0.50 × weighted CE_fine + 1.00 × weighted soft-Dice_fine]
  + 0.30 × [weighted CE_coarse + weighted soft-Dice_coarse]
  + 0.40 × BCEWithLogits_visibility(pos_weight từ train fold, clip 1..8)
  + 0.08 × weighted boundary BCE_fine
```

Trọng số segmentation lấy từ cột `weight` trong label map (clinical tier), không phải tần suất pixel. `pos_weight` Task 3 chỉ tính trên train partition của fold đó: `clip(n_negative / max(n_positive, 1), 1, 8)`.

## 6. Lệnh chạy chuẩn

Lệnh dưới đây tái tạo một fold của đối chứng (ví dụ fold 0):

```bash
.venv/bin/python train.py \
  --fold 0 --n-folds 5 --epochs 100 \
  --output-dir artifacts/reproduction \
  --run-name fold0_affine_full \
  --encoder resnet34 --finetune full \
  --spatial-augmentation \
  --batch-size 4 --num-workers 8 --height 512 --width 896 \
  --seed 2026
```

Chạy toàn bộ 5-fold đối chứng/đối thủ với đúng resource protocol:

```bash
.venv/bin/python crossval_setups.py \
  --setups affine_full \
  --n-folds 5 --epochs 100 \
  --output-dir artifacts/reproduction/cv_affine_full \
  -- --batch-size 4 --num-workers 8 --height 512 --width 896 --seed 2026
```

Với method mới, giữ nguyên `--fold`, `--n-folds`, epochs, seed, resolution, batch size, train frame budget và cách checkpoint selection. Chỉ thay entrypoint/model/loss/sampler thuộc method mới. Ghi đầy đủ command và `config.json` của từng fold.

## 7. Predict / inference như thế nào

Với checkpoint `best.pt`, lấy `args` đã lưu trong checkpoint để khôi phục encoder, kích thước input, normalization và các lớp. Luồng prediction cho **một frame**:

```text
RGB PNG gốc
  → RGB + resize 512×896 + ImageNet normalization
  → model
  ├─ fine logits [31,H,W]       → argmax(channel) → fine-ID mask
  ├─ coarse logits [16,H,W]     → argmax(channel) → coarse-ID mask
  └─ visibility logits [14]     → sigmoid → 14 probabilities
                                      → threshold từng station → multi-hot output
```

- Fine/coarse segmentation dùng `argmax`, không threshold foreground.
- Khi cần nộp fine mask RGB, encode từng fine ID ngược lại màu `(fine_r, fine_g, fine_b)` đúng trong `labelmap.csv`; không được xuất palette/index PNG thay cho RGB code.
- Visibility dùng sigmoid. Khi chưa có calibration hợp lệ, ngưỡng mặc định là `0.5` cho mọi station.
- Sau khi có đủ 5 fold, threshold Task 3 được fit **một lần** từ toàn bộ OOF predictions bằng grid `0.05, 0.075, …, 0.95`, tối đa F1 riêng cho từng station. Lệnh:

```bash
.venv/bin/python calibrate_visibility.py \
  artifacts/reproduction/cv_affine_full --method affine_full
```

Không fit threshold trên hidden test, hoặc trên validation fold đang được dùng để công bố score cuối. `train.py` đã lưu `best_validation_visibility.npz` để làm việc này.

Lưu ý phạm vi code hiện tại: `train.py` thực hiện validation inference và `generate_qualitative_results.py` có ví dụ nạp checkpoint/infer; chưa có một CLI inference/submission hoàn chỉnh cho dữ liệu test không nhãn. Khi tạo script submission, script đó phải thực hiện đúng luồng trên, đọc threshold JSON đã fit OOF và ghi mask RGB theo label map.

## 8. Đánh giá và tiêu chí chọn checkpoint

Mỗi epoch validation tạo các metric sau:

| Task | Metric |
|---|---|
| Fine (31 lớp) | clinically weighted macro Dice ↑, weighted macro normalized Hausdorff distance (nHD) ↓ |
| Coarse (16 lớp) | weighted macro Dice ↑, weighted macro nHD ↓ |
| Visibility (14 labels) | macro F1 ↑, macro AUROC ↑ |

Selection score nội bộ dùng để chọn `best.pt` trong mỗi fold:

```text
mean(Fine Dice, 1 − Fine nHD,
     Coarse Dice, 1 − Coarse nHD,
     Visibility macro F1, Visibility macro AUROC)
```

Đây là local surrogate của repository, không phải official hidden-test score. Cần báo cáo cả sáu metric thành phần, score mean ± sample SD qua 5 folds, metric per-class/per-station và checkpoint epoch của từng fold. Không dùng `final_metrics.json` để lấy best score: file này lưu metric ở epoch cuối; dùng record có `selection_score` cao nhất trong `history.json`/`best.pt`.

## 9. Checklist công bằng cho phương pháp mới

- [ ] Đúng cùng 140 frames, label map và visibility CSV (kiểm tra hash ở mục 2).
- [ ] Đúng 5 case-level folds; không frame/case leakage.
- [ ] Cùng preprocessing 512×896 và validation không augmentation.
- [ ] Cùng ngân sách 100 epochs, cùng optimizer-step budget (baseline là 28 steps/epoch với batch 4), cùng seed 2026; nếu khác phải nêu rõ và chạy control tương ứng.
- [ ] Không dùng extra/private data. Nếu dùng ImageNet/public pretraining hoặc external checkpoint, công khai nguồn, license và dữ liệu pretraining.
- [ ] Hyperparameter chỉ tune bằng training/validation của từng fold; không tune trên hidden test.
- [ ] Chọn checkpoint bằng cùng validation `selection_score`; báo cáo OOF 5-fold, không chỉ Fold 0.
- [ ] Threshold Task 3 chỉ calibrate từ OOF predictions; tách rõ F1 threshold 0.5 và F1 calibrated nếu cả hai được báo cáo.
- [ ] Giữ Task 1, 2 và 3 trong so sánh. Nếu method chỉ giải được segmentation (như nnU-Net pilot), báo cáo Task 1/2 riêng và **không** so overall three-task score.
- [ ] Lưu `config.json`, `history.json`, `best.pt`, log package/CUDA/GPU, code commit hash và prediction/OOF file để audit.

## 10. Mốc tham chiếu hiện có

| Evidence level | Method | Protocol | Local score |
|---|---|---|---:|
| Primary | `affine_full` | 5-fold × 100 epochs | **0.5530 ± 0.0627** |
| Secondary | `gdf_affine_frozen_bn` | 5-fold × 100 epochs | 0.5478 ± 0.0605 |
| Screening only | ResNet-34 Tversky–Focal + affine | Fold 0 × 100 epochs | 0.6244 |
| Screening only | SurgeNet-Public full + affine | Fold 0 × 100 epochs | 0.6230 |

Chênh lệch giữa hai phương pháp đã hoàn thành CV là 0.0052, nhỏ hơn biến thiên giữa folds. Vì vậy một method mới chỉ nên được coi là tốt hơn khi có kết quả 5-fold theo protocol này, không chỉ vượt vài phần nghìn trên Fold 0.

## 11. Tệp nguồn trong repository

- Data/split/preprocessing: `tiger_baseline/data.py`
- Model: `tiger_baseline/model.py`
- Loss: `tiger_baseline/losses.py`
- Metrics và selection score: `tiger_baseline/metrics.py`
- Trainer và artifact format: `train.py`
- Runner CV chuẩn: `crossval_setups.py`, `sweep_setups.py`, `scripts/slurm/cv_top_setups_100_mig.sbatch`
- OOF threshold calibration: `calibrate_visibility.py`
- Audit kết quả hiện có: `SCORE_REPORT_ALL_VERSIONS.md`
