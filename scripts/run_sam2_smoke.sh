#!/usr/bin/env bash
set -euo pipefail

model="${1:?model required}"
experiment="${2:?experiment required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tiger_mpl_${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
output="artifacts/sam_7_1_2_boundary_100/smoke/${experiment}/fold_0"
if [[ -e "${output}" ]]; then
  retry_archive="artifacts/sam_7_1_2_boundary_100/smoke_retry_archive/${experiment}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$(dirname "${retry_archive}")"
  mv "${output}" "${retry_archive}"
fi
mkdir -p "${output}"

"${python_bin}" scripts/train_sam.py --model "${model}" --fold 0 --n-folds 5 \
  --epochs 1 --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
  --gradient-checkpointing --finetune partial --mixed-precision bf16 \
  --num-workers 4 --seed 2026 --run-kind smoke --boundary-loss-weight 0.20 \
  --early-stopping-patience 15 --output-dir "${output}"
"${python_bin}" scripts/predict_sam.py --checkpoint "${output}/best_val.pt" \
  --fold 0 --split test --output-dir "${output}/predictions"
"${python_bin}" scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
  --fold 0 --split test --run-official --official-fixture \
  --output "${output}/prediction_validation.json"
"${python_bin}" scripts/generate_sam_qualitative.py \
  --prediction-dir "${output}/predictions" --fold 0 --split test \
  --output-dir "${output}/qualitative"
test -f "${output}/best_train.pt"
test -f "${output}/best_val.pt"
test -f "${output}/best_selection.pt"
test -f "${output}/quantitative_metrics.json"
test -f "${output}/qualitative/SAM_ERROR_ANALYSIS.md"
