# Patch: 5-fold 32-train / 8-val split (no held-out local test)

**Vì sao có patch này:** `full_version/` thuộc owner/group `thu.vtm`, mode
`0775` — user `hieu.ph` không có quyền ghi vào bất kỳ đâu trong đó (kể cả
tạo file mới), nên toàn bộ thay đổi được chuẩn bị ở đây, tại thư mục gốc
repo mà cả hai user đều ghi được (`0777`). Không có gì trong
`full_version/` bị sửa trực tiếp.

**Ý tưởng:** dataset hiện dùng split 3 tier 28 train / 4 val / 8 test mỗi
fold (`case_folds_v1.json`). Vì hidden test của challenge mới là test thật,
tier test nội bộ 8 case/fold không phục vụ gì cho training — chỉ dùng cho
`evaluate_test.py`/OOF report. Bỏ tier đó đi, gộp 8 case ấy làm validation
trực tiếp: **32 train / 8 val**, không còn test nội bộ. Cách gom nhóm case
tái dùng đúng `TEST_CASES_SEED_2026` đã được review/khóa trước đó (mỗi ca
vẫn validate đúng 1 lần/5 fold, đủ 6 center mỗi fold) — chỉ bỏ bớt tier,
không đổi cách gán ca.

Đã verify: `records_for_manifest` gốc (không sửa) chạy đúng trên manifest
mới, không leak case giữa train/val, đủ 40 case × 524 frame qua 5 fold.

## Nội dung thư mục

| File | Đích copy vào |
|---|---|
| `case_folds_v2_trainval.json` | `full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json` |
| `data_full_ADDITION.py` | nội dung cần **append** vào cuối `full_version/task2_fine_31cls/scripts/data_full.py`, ngay trước `def load_manifest(...)` |
| `data_full.py.patch` | unified diff tương đương, apply bằng `patch`/`git apply` |
| `train_full.py.patch` | diff cho `full_version/task2_fine_31cls/scripts/train_full.py` (chấp nhận cả 2 manifest version) |
| `train_coarse_only.py.patch` | diff cho `full_version/task1_merged_16cls/scripts/train_coarse_only.py` (tương tự) |
| `modified/*.py` | bản đầy đủ đã sửa sẵn, có thể copy đè trực tiếp thay vì apply patch |
| `configs/full40_fine31_trainval32_8_seed2026.yaml` | `full_version/task2_fine_31cls/configs/` — hyperparameter giống hệt `full40_fine31_c4_a0_seed2026.yaml`, chỉ đổi `experiment_id` |
| `configs/coarse_only_swin_small_trainval32_8_seed2026.yaml` | `full_version/task1_merged_16cls/configs/` — tương tự cho Task 1 |
| `jobs/train_task2_fine31_trainval32_8.sbatch` | `full_version/task2_fine_31cls/jobs/` |
| `jobs/train_task1_coarse_only_trainval32_8.sbatch` | `full_version/task1_merged_16cls/jobs/` |

## Cách merge (chạy bởi user có quyền ghi `full_version/`, vd. thu.vtm)

```bash
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI

# 1. Áp code patch (chọn 1 trong 2 cách; đã dry-run pass với patch -p0)
patch -p0 < patch_hieu_trainval_split/data_full.py.patch
patch -p0 < patch_hieu_trainval_split/train_full.py.patch
patch -p0 < patch_hieu_trainval_split/train_coarse_only.py.patch
# --- hoặc đơn giản hơn, copy đè cả file ---
cp patch_hieu_trainval_split/modified/data_full.py         full_version/task2_fine_31cls/scripts/data_full.py
cp patch_hieu_trainval_split/modified/train_full.py         full_version/task2_fine_31cls/scripts/train_full.py
cp patch_hieu_trainval_split/modified/train_coarse_only.py  full_version/task1_merged_16cls/scripts/train_coarse_only.py

# 2. Copy split manifest + config + job
cp patch_hieu_trainval_split/case_folds_v2_trainval.json \
   full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json
cp patch_hieu_trainval_split/configs/full40_fine31_trainval32_8_seed2026.yaml \
   full_version/task2_fine_31cls/configs/
cp patch_hieu_trainval_split/configs/coarse_only_swin_small_trainval32_8_seed2026.yaml \
   full_version/task1_merged_16cls/configs/
cp patch_hieu_trainval_split/jobs/train_task2_fine31_trainval32_8.sbatch \
   full_version/task2_fine_31cls/jobs/
cp patch_hieu_trainval_split/jobs/train_task1_coarse_only_trainval32_8.sbatch \
   full_version/task1_merged_16cls/jobs/

# 3. (khuyến nghị) regenerate manifest tại chỗ bằng script chính thức thay vì
#    dùng file JSON copy từ ngoài, để provenance sạch:
python full_version/task2_fine_31cls/scripts/prepare_trainval_splits.py \
   --output full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json
#    (script này CHƯA tồn tại — xem mẫu prepare_splits.py, chỉ cần đổi
#    build_manifest -> build_trainval_manifest, mặc định output đổi tên)

# 4. Smoke test 1 fold trước khi submit cả mảng
python full_version/task2_fine_31cls/scripts/train_full.py \
  --config full_version/task2_fine_31cls/configs/full40_fine31_trainval32_8_seed2026.yaml \
  --split-manifest full_version/task2_fine_31cls/splits/case_folds_v2_trainval.json \
  --fold 0 --smoke \
  --output-root full_version/task2_fine_31cls/artifacts/smoke_trainval32_8

# 5. Submit thật (SAU khi smoke pass)
sbatch full_version/task2_fine_31cls/jobs/train_task2_fine31_trainval32_8.sbatch
sbatch full_version/task1_merged_16cls/jobs/train_task1_coarse_only_trainval32_8.sbatch
```

## Việc còn lại / lưu ý

- **Task 3 (visibility)**: `full_version/task3_visibility` hiện dùng
  `source_segmentation` là OOF **test** predictions của Task 2
  (`full_version/task2_fine_31cls/artifacts/seed_2026/test_oof/...`). Với
  scheme 32/8 mới không còn test tier, Task 3 cần đổi sang dùng OOF
  **validation** predictions (32/8) làm nguồn — chưa làm trong patch này,
  cần một bước nữa sau khi Task 2 train xong.
- **So sánh công bằng**: baseline C4-A0 hiện tại (0.774027 Task-2 OOF,
  0.742988 Task-1 OOF) được đo trên **test** tier (8 case/fold, chưa từng
  train). Kết quả từ scheme 32/8 mới sẽ được đo trên **validation** tier
  (cũng 8 case/fold, cũng chưa từng train trong đúng fold đó) — về học
  thuật đây vẫn là so sánh hợp lệ (cùng là held-out case-level), nhưng nên
  nêu rõ trong report là "validation-selected, no separate test tier" để
  không nhầm là cùng một loại điểm với OOF test cũ.
- Sau khi review lại, nên xoá thư mục `patch_hieu_trainval_split/` khỏi
  repo root (đây chỉ là staging, không phải vị trí lâu dài).
- Không tự chmod/chgrp `full_version/` để né việc này — nếu muốn tôi sửa
  trực tiếp về sau, cần thu.vtm hoặc admin cấp quyền ghi cho `hieu.ph`.
