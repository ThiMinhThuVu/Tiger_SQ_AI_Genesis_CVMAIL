# Báo cáo dữ liệu và thực nghiệm CholecSeg8k

**Ngày tổng hợp:** 2026-08-04  
**Phạm vi:** các split, cấu hình, log, checkpoint và kết quả CholecSeg8k hiện có trong repository `TIGER_SQ_AI`.

> [!CAUTION]
> Các kết quả trong báo cáo này chỉ có giá trị **diagnostic** và không được dùng làm kết quả semantic segmentation chính thức. Bước chuẩn bị dữ liệu hiện tại đọc `*_endo_mask.png` (annotation-tool mask) thay vì dense `*_endo_watershed_mask.png`. Quan sát trực tiếp cho thấy annotation mask có thể chỉ chứa nét/seed hoặc vùng tô không đồng nhất giữa các frame. Vì train, validation và test cùng dùng target này, tất cả metric bên dưới đo khả năng khớp annotation-tool mask, không đo dense anatomy segmentation.

## 1. Tổng quan dataset

CholecSeg8k gồm **8.080 frame**, kích thước gốc **854 × 480**, lấy từ 17 video phẫu thuật nội soi cắt túi mật. Mỗi frame có bốn ảnh PNG:

| Hậu tố | Vai trò | Cách dùng phù hợp |
|---|---|---|
| `*_endo.png` | Ảnh nội soi RGB | Input của model |
| `*_endo_mask.png` | Annotation-tool mask, chứa nhãn/nét/seed trung gian | Audit annotation; không nên dùng trực tiếp làm dense GT |
| `*_endo_watershed_mask.png` | Phân vùng watershed đầy đủ | Decode thành class ID để làm dense GT |
| `*_endo_color_mask.png` | Bản màu trực quan của phân vùng | Hiển thị và kiểm tra mapping |

Dataset định nghĩa 13 lớp:

1. background
2. abdominal wall
3. liver
4. gastrointestinal tract
5. fat
6. grasper
7. connective tissue
8. blood
9. cystic duct
10. L-hook electrocautery
11. gallbladder
12. hepatic vein
13. liver ligament

Tham khảo: [CholecSeg8k dataset card](https://huggingface.co/datasets/minwoosun/CholecSeg8k) và [dataset loader](https://huggingface.co/datasets/minwoosun/CholecSeg8k/blob/main/CholecSeg8k.py).

## 2. Protocol chia dữ liệu hiện tại

Split được cố định theo **video**, seed `2026`, nhằm ngăn frame leakage giữa train, validation và test.

- Tổng cộng: 8.080 frame, 17 video.
- Fixed test: 1.600 frame từ 3 video (`video09`, `video28`, `video52`), tương đương 19,8% số frame.
- Development: 6.480 frame từ 14 video.
- Development được chia 5 fold; mỗi lần dùng một nhóm video làm validation và các video development còn lại làm train.
- Fixed test giống nhau cho cả năm checkpoint/fold.

| Fold | Train frames | Validation frames | Validation videos |
|---:|---:|---:|---|
| 0 | 5.200 | 1.280 | `video01` |
| 1 | 5.120 | 1.360 | `video20`, `video24`, `video35` |
| 2 | 5.200 | 1.280 | `video26`, `video43`, `video48` |
| 3 | 5.120 | 1.360 | `video12`, `video17`, `video18`, `video55` |
| 4 | 5.280 | 1.200 | `video25`, `video27`, `video37` |

Protocol có kiểm tra giao nhau ở cả mức đường dẫn ảnh và video. Tuy nhiên, target trong manifest hiện trỏ đến `grasp_splits_seed2026/semantic_masks/`, được tạo từ kênh đầu của `*_endo_mask.png` bằng LUT ID, không phải từ watershed mask.

## 3. Các phương pháp đã thử nghiệm

### 3.1. GraSP / Mask2Former-R50 baseline

- Framework: Detectron2/Mask2Former trong cây mã nguồn GraSP/TAPIS.
- Backbone: ResNet-50, khởi tạo từ checkpoint COCO panoptic Mask2Former.
- Head: MaskFormer semantic segmentation, 13 lớp, 100 object query.
- Pixel decoder: multi-scale deformable-attention pixel decoder.
- Transformer decoder: 10 layer, hidden dimension 256, 8 attention head.
- Loss: classification weight 2, mask weight 5, Dice weight 5; deep supervision bật.
- Optimizer: AdamW; learning rate `2e-4`; weight decay `0.05`.
- Schedule: 9.000 iteration; LR step tại 6.000 và 8.000.
- Batch size: 12; AMP bật; native resolution 480 × 854.
- Thực nghiệm: đủ 5 fold train/validation và đủ 5 lần đánh giá trên fixed test.
- Metric: mIoU, frequency-weighted IoU (fwIoU), mean accuracy (mACC), pixel accuracy (pACC), đơn vị `%`.

### 3.2. SurgiGraph-Q A3-r2-GF / Mask2Former-Swin-S

- Backbone/checkpoint: `facebook/mask2former-swin-small-coco-panoptic`.
- Output: 13 lớp semantic segmentation.
- Matching: asymmetric partial Sinkhorn optimal transport.
- OT: epsilon `0.5`, tối đa 500 iteration, tolerance `1e-7`.
- Adaptive target mass: alpha `0.5`, maximum `2.0`.
- Dustbin: false-positive weight `0.1`, cost `0.0`.
- Optimizer: AdamW; learning rate `1e-4`; weight decay `1e-4`.
- Batch logic: effective batch 16, micro-batch 4, gradient accumulation 1.
- Resolution: 480 × 854.
- Early stopping: patience 15, minimum delta `1e-4`.
- Metric: fine Dice, normalized Hausdorff distance (nHD), và `Task1 score = mean(Dice, 1 − nHD)`.
- Trạng thái train: fold 0 và 1 bị dừng sau khoảng 22 epoch; fold 2 bị dừng sau 12 epoch; fold 3 bị dừng sau 14 epoch; fold 4 hoàn tất 20 epoch. Mọi fold đều có checkpoint `best.pt` và đã được test trên fixed test.

### 3.3. Smoke tests

Bốn cấu hình vận hành ngắn đã được thử để kiểm tra bộ nhớ và batch/micro-batch: batch 4, batch 8, batch 12/micro 4, và batch 16/micro 8 với AMP. Đây là engineering smoke tests một epoch, không phải kết quả nghiên cứu và không được đưa vào bảng so sánh chính.

## 4. Kết quả

### 4.1. GraSP / Mask2Former-R50 — validation

| Fold | mIoU ↑ | fwIoU ↑ | mACC ↑ | pACC ↑ |
|---:|---:|---:|---:|---:|
| 0 | 19.8769 | 54.3619 | 26.8947 | 71.2374 |
| 1 | 25.3399 | 69.8120 | 39.8978 | 77.7367 |
| 2 | 21.0976 | 72.5898 | 32.2912 | 80.4905 |
| 3 | 16.9005 | 51.2736 | 22.5293 | 68.7882 |
| 4 | 26.0966 | 61.9619 | 37.2465 | 76.4952 |
| **Mean ± SD** | **21.8623 ± 3.4400** | **61.9998 ± 8.3255** | **31.7719 ± 6.4107** | **74.9496 ± 4.3035** |

### 4.2. GraSP / Mask2Former-R50 — fixed test

| Training fold | mIoU ↑ | fwIoU ↑ | mACC ↑ | pACC ↑ |
|---:|---:|---:|---:|---:|
| 0 | 17.5291 | 54.7173 | 27.3312 | 71.1492 |
| 1 | 23.0381 | 56.2533 | 35.4426 | 71.2840 |
| 2 | 19.0761 | 54.2983 | 30.8242 | 69.7244 |
| 3 | 17.6713 | 56.2432 | 25.5460 | 72.7258 |
| 4 | 15.9253 | 54.8566 | 23.9849 | 71.7045 |
| **Mean ± SD** | **18.6480 ± 2.4114** | **55.2737 ± 0.8166** | **28.6258 ± 4.0988** | **71.3176 ± 0.9696** |

### 4.3. SurgiGraph-Q A3-r2-GF — validation checkpoint tốt nhất

Epoch trong bảng dùng chỉ số zero-based như log/checkpoint.

| Fold | Best epoch | Task1 score ↑ | Fine Dice ↑ | Fine nHD ↓ |
|---:|---:|---:|---:|---:|
| 0 | 8 | 0.6846 | 0.5817 | 0.2124 |
| 1 | 13 | 0.8136 | 0.7052 | 0.0780 |
| 2 | 0 | 0.8307 | 0.7179 | 0.0564 |
| 3 | 6 | 0.7160 | 0.5914 | 0.1593 |
| 4 | 12 | 0.8220 | 0.7083 | 0.0644 |
| **Mean ± SD** | — | **0.7734 ± 0.0607** | **0.6609 ± 0.0609** | **0.1141 ± 0.0613** |

### 4.4. SurgiGraph-Q A3-r2-GF — fixed test

| Training fold | Task1 score ↑ | Fine Dice ↑ | Fine nHD ↓ |
|---:|---:|---:|---:|
| 0 | 0.7941 | 0.6940 | 0.1057 |
| 1 | 0.7872 | 0.6837 | 0.1093 |
| 2 | 0.7733 | 0.6659 | 0.1193 |
| 3 | 0.7831 | 0.6745 | 0.1083 |
| 4 | 0.7892 | 0.6840 | 0.1057 |
| **Mean ± SD** | **0.7854 ± 0.0070** | **0.6804 ± 0.0095** | **0.1096 ± 0.0050** |

### 4.5. Qualitative diagnostic của SurgiGraph fold 4

Ba frame test được chọn theo per-image Dice:

| Case | Frame | Dice ↑ | nHD ↓ |
|---|---|---:|---:|
| Best | `video28/video28_00080/frame_146_endo.png` | 0.8186 | 0.0423 |
| Median | `video52/video52_00400/frame_401_endo.png` | 0.7112 | 0.0863 |
| Worst | `video28/video28_00400/frame_436_endo.png` | 0.4946 | 0.2557 |

Quan sát qualitative chính là bằng chứng phát hiện lỗi target: best và worst có GT chủ yếu là nét/seed, trong khi median có vùng anatomy tô kín. Tỷ lệ pixel khác background lần lượt khoảng 3,1%, 56,5% và 6,5%.

## 5. Diễn giải và giới hạn

1. **Không thể công bố các bảng trên như dense semantic segmentation.** Cả hai method dùng chung manifest và target được sinh từ annotation-tool mask; do đó so sánh nội bộ chỉ phản ánh pipeline hiện tại.
2. **Không so sánh trực tiếp trị số giữa GraSP và SurgiGraph.** GraSP báo mIoU/mACC theo Detectron2, còn SurgiGraph báo Dice/nHD/Task1 score theo evaluator riêng.
3. **Fixed test bị đánh giá lặp lại bằng năm checkpoint.** Đây là cách báo cáo độ biến thiên theo training fold, nhưng nếu dùng test để chọn checkpoint/method sẽ gây test-set selection bias.
4. **Một số fold SurgiGraph bị dừng thủ công.** Checkpoint tốt nhất vẫn tồn tại, nhưng ngân sách train không đồng nhất giữa fold.
5. **Validation variance lớn.** Mỗi fold chỉ có 1–4 video validation; khác biệt anatomy và video làm metric dao động đáng kể.
6. **Hai lớp hiếm có thể không xuất hiện ở một số split.** Log GraSP có metric `NaN` cho hepatic vein/liver ligament ở một số fold, khiến macro metric phụ thuộc quy ước bỏ qua lớp vắng mặt.

## 6. Hành động cần làm trước thực nghiệm chính thức

1. Sửa `prepare_cholecseg8k.py` để đọc `*_endo_watershed_mask.png`.
2. Xây LUT source value/RGB → contiguous ID 0–12 từ watershed mask và kiểm tra chéo với `*_endo_color_mask.png`.
3. Tạo thư mục mask/split mới, không ghi đè artifact hiện tại; lưu checksum manifest.
4. Audit tối thiểu: ID hợp lệ, tỷ lệ lớp, tỷ lệ foreground, equivalence watershed–color, và hiển thị ngẫu nhiên ≥30 frame.
5. Giữ nguyên video-level split hiện tại nếu audit xác nhận không leakage, để so sánh protocol nhất quán.
6. Train lại baseline GraSP và SurgiGraph với cùng target dense; thống nhất thêm metric Dice, mIoU và nHD cho cả hai.
7. Chỉ chạy fixed test một lần sau khi khóa config/checkpoint selection bằng validation.

## 7. Nguồn artifact và khả năng tái lập

- Split metadata: `cross_data/data/CholecSeg8k/grasp_splits_seed2026/splits.json`
- Data preparation: `test_sota/GraSP/TAPIS/tools/prepare_cholecseg8k.py`
- GraSP config: `test_sota/GraSP/TAPIS/region_proposals/configs/cholecseg8k/Base-CholecSeg8k-SemanticSegmentation.yaml`
- GraSP outputs: `outputs/grasp_cholecseg8k/`
- SurgiGraph config: `configs/surgigraph_q_a3_r2_gf_cholecseg8k.yaml`
- SurgiGraph train/eval code: `scripts/train_surgigraph_cholecseg8k.py`
- SurgiGraph artifacts: `artifacts/cholecseg8k_surgigraph/surgigraph_q_a3_r2_gf_cholecseg8k/`
- Qualitative manifest: `artifacts/cholecseg8k_surgigraph/surgigraph_q_a3_r2_gf_cholecseg8k/qualitative_fold4/selections.json`

