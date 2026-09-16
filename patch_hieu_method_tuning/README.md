
---

## RUNBOOK (chạy bởi thu.vtm — hieu.ph không exec được .venv)

> `.venv` là venv trên `/home/thu.vtm/miniconda3` → hieu.ph không traverse được;
> system python không có numpy. Mọi bước dưới đây cần thu.vtm.

### 0. Deploy (1 lần)
```bash
cd /mnt/disk_1/backup_user/thu.vtm/TIGER_SQ_AI
cp patch_hieu_method_tuning/configs/full40_fine31_c4a1_improved_v2_seed2026.yaml \
   full_version/task2_fine_31cls/configs/
cp patch_hieu_method_tuning/jobs/*.sbatch full_version/task2_fine_31cls/jobs/
```

### 1. Smoke fold 0 (bắt buộc trước 5-fold)
```bash
sbatch full_version/task2_fine_31cls/jobs/smoke_c4a1_iv2_fold0.sbatch
# đọc log: full_version/task2_fine_31cls/logs/smoke_c4a1_iv2_<jobid>.log
# PASS = có epoch_metrics.json + best.pt trong artifacts/smoke_c4a1_iv2/.../fold_0,
#        val_task1_score in ra hợp lý (~0.74-0.78), loss các term (fine/coarse/
#        hierarchy/presence) đều > 0 và giảm. Rồi xóa artifacts/smoke_c4a1_iv2.
```

### 2. Train 5-fold seed 2026
```bash
sbatch full_version/task2_fine_31cls/jobs/train_task2_c4a1_improved_v2.sbatch
# → artifacts/seed_2026/full40_fine31_c4a1_improved_v2_seed2026/fold_{0..4}/best.pt
```

### 3. Held-out OOF test (Task 2, so với C4-A0 0.774027)
```bash
sbatch full_version/task2_fine_31cls/jobs/evaluate_c4a1_iv2_test_5fold.sbatch
sbatch --dependency=afterok:<jobid_eval> \
  full_version/task2_fine_31cls/jobs/aggregate_c4a1_iv2_test.sbatch
# → artifacts/seed_2026/test_oof_c4a1_iv2/oof_test_metrics.json
```

### 4. Task 1 (coarse) — cập nhật sau khi Task 2 xong
`finalize_from_fine_oof.py` cần native-ID OOF PNG của model mới. Export bằng
p0_baseline_lock-style OOF native trên checkpoint mới, rồi:
```bash
python full_version/task1_merged_16cls/scripts/finalize_from_fine_oof.py \
  --fine-prediction-dir <new native-id oof dir> \
  --fine-metrics full_version/task2_fine_31cls/artifacts/seed_2026/test_oof_c4a1_iv2/oof_test_metrics.json \
  --output-dir full_version/task1_merged_16cls/artifacts/full40_c4a1_iv2_seed2026
```

### Gate promote
Chỉ coi là win nếu Task-2 OOF ≥ C4-A0 (0.774027) trên **≥3/5 fold paired**
và mean tăng ≥ +0.005. Report paired fold delta, không chỉ mean.
