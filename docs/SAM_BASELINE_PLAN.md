# SAM 2.1 — kế hoạch chạy lại 5-fold 7/1/2

**Ngày cập nhật:** 2026-07-17 UTC  
**Trạng thái:** ba smoke đã pass; ba full jobs 100 epoch đang chạy  
**Phiên bản:** `sam_7_1_2_boundary_100`

## Mục tiêu cố định

- Chạy đủ ba encoder SAM 2.1: Hiera Tiny, Hiera Small và Hiera Base+.
- Giữ 5 fold theo case; mỗi fold có 7 train / 1 validation / 2 test case.
- Cặp validation cũ trở thành test. Validation mới lấy một case từ training pool.
- Tối đa 100 epoch, early stopping theo `val_loss`, patience 15, min delta `1e-4`.
- Loss: `0.50 CE + 1.00 soft Dice + 0.40 visibility BCE + 0.20 (boundary BCE + boundary Dice)`.
- Predict và metric cuối cùng chỉ chạy trên test split bằng checkpoint `best_val.pt`.
- Không dùng test để chọn epoch hoặc early stopping.

## Split đã khóa

| Fold | Validation (1 case) | Test (2 case) | Train |
|---:|---|---|---:|
| 0 | `center_1_case_11` | `center_1_case_10`, `center_1_case_15` | 7 case / 98 ảnh |
| 1 | `center_1_case_12` | `center_1_case_11`, `center_1_case_6` | 7 case / 98 ảnh |
| 2 | `center_1_case_13` | `center_1_case_12`, `center_1_case_7` | 7 case / 98 ảnh |
| 3 | `center_1_case_14` | `center_1_case_13`, `center_1_case_8` | 7 case / 98 ảnh |
| 4 | `center_1_case_10` | `center_1_case_14`, `center_1_case_9` | 7 case / 98 ảnh |

Mỗi case xuất hiện đúng một lần trong test qua 5 fold. Ba split trong từng fold không giao nhau ở mức case hoặc frame.

## Artifact bắt buộc cho mỗi fold

- Toàn bộ loss/accuracy theo batch: `train_batch_metrics.csv`.
- Toàn bộ train loss, validation loss, accuracy, metric và learning rate theo epoch: `epoch_metrics.csv`, `history.json`.
- Checkpoint: `best_train.pt`, `best_val.pt`, `best_selection.pt`, `best.pt` (alias selection), `last.pt`.
- Tiêu chí checkpoint: `checkpoint_manifest.json`, `best_*_metrics.json`.
- Early stopping: `early_stopping.json`, `training_completion.json`.
- Predict test: `predictions/task1`, `task2`, `task3.csv`, entropy từng ảnh.
- Định lượng test: `quantitative_metrics.json`, `prediction_validation.json`, official fixture và native-fast report.
- Định tính test: `qualitative/*.png` gồm original, GT, prediction, semantic error, boundary overlay, coarse masks, entropy và visibility probabilities.
- Log Slurm tổng và log riêng từng fold trong `logs/`.

## Bố trí kết quả

- Kết quả mới: `artifacts/sam_7_1_2_boundary_100/<model>/fold_<k>`.
- Smoke test: `artifacts/sam_7_1_2_boundary_100/smoke/<model>/fold_0`.
- Kết quả cũ được chuyển nguyên trạng vào `artifacts/archive/2026-07-17_pre_7-1-2_boundary_100/`.
- Log cũ được chuyển vào `logs/archive/2026-07-17_pre_7-1-2_boundary_100/`.
- Report cũ được chuyển vào `reports/archive/2026-07-17_pre_7-1-2_boundary_100/`.

## Gate thực thi

1. Data audit và unit tests phải pass.
2. Submit ba smoke job 1 epoch.
3. Mỗi full job 100 epoch dùng Slurm `afterok:<smoke_job_id>` tương ứng; không cần Codex theo dõi để submit tiếp.
4. Smoke fail thì full job tương ứng không chạy.
5. Full runner chạy fold 0→4 tuần tự, hỗ trợ resume từ `last.pt`, và chỉ đánh dấu fold hoàn tất khi đủ checkpoint/log/metric/qualitative.

## Job tracking

| Model | Smoke job | Full job (`afterok`) | Trạng thái |
|---|---:|---:|---|
| Hiera Tiny | 87 | 90 (`afterok:87`) | smoke COMPLETED; full RUNNING |
| Hiera Small | 86 | 91 (`afterok:86`) | smoke COMPLETED; full RUNNING |
| Hiera Base+ | 88 | 89 (`afterok:88`) | smoke COMPLETED; full RUNNING |

Trạng thái trên được xác nhận bằng `squeue` ngày 2026-07-17 UTC. Cả ba dependency đã được giải phóng sau khi smoke exit 0.

## Lệnh kiểm tra

```bash
.venv/bin/python scripts/audit_sam_data.py --data-root data --full
.venv/bin/pytest -q
bash -n scripts/run_sam2_smoke.sh scripts/run_sam2_variant_100fold.sh
```
