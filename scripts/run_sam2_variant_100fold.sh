#!/usr/bin/env bash
set -euo pipefail

model="${1:?model required}"
experiment="${2:?experiment required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tiger_mpl_${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"
base="artifacts/sam_7_1_2_boundary_100/${experiment}"
mkdir -p "${base}" logs
exec 9>"${base}/.training.lock"
flock -n 9 || exit 73

for fold in 0 1 2 3 4; do
  output="${base}/fold_${fold}"
  completion="${output}/completion.json"
  if [[ -f "${completion}" ]]; then
    echo "Fold ${fold} already complete; preserving artifacts"
    continue
  fi
  mkdir -p "${output}"
  log="logs/${experiment}_7-1-2_boundary_fold_${fold}_100.log"
  resume=()
  if [[ -f "${output}/last.pt" ]]; then resume+=(--resume); fi
  if [[ ! -f "${output}/training_completion.json" ]]; then
    "${python_bin}" scripts/train_sam.py \
      --model "${model}" --fold "${fold}" --n-folds 5 --epochs 100 \
      --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
      --gradient-checkpointing --finetune partial --mixed-precision bf16 \
      --num-workers 4 --seed 2026 --run-kind primary \
      --boundary-loss-weight 0.20 --early-stopping-patience 15 \
      --early-stopping-min-delta 0.0001 --output-dir "${output}" \
      "${resume[@]}" >"${log}" 2>&1
  fi
  if [[ ! -f "${output}/predictions/prediction_summary.json" ]]; then
    if [[ -d "${output}/predictions" ]]; then
      mv "${output}/predictions" "${output}/predictions_incomplete_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    "${python_bin}" scripts/predict_sam.py --checkpoint "${output}/best_val.pt" \
      --fold "${fold}" --split test --output-dir "${output}/predictions" >>"${log}" 2>&1
  fi
  "${python_bin}" scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
    --fold "${fold}" --split test --output "${output}/prediction_validation.json" >>"${log}" 2>&1
  "${python_bin}" scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
    --fold "${fold}" --split test --run-official --official-fixture \
    --output "${output}/official_fixture_validation.json" >>"${log}" 2>&1
  "${python_bin}" scripts/prepare_official_native_input.py \
    --prediction-dir "${output}/predictions" --fold "${fold}" --split test \
    --destination "${output}/official_native_input" >>"${log}" 2>&1
  "${python_bin}" scripts/run_native_official_fast.py \
    --gt "${output}/official_native_input/gt" \
    --pred "${output}/official_native_input/predictions" \
    --out "${output}/official_native_fast_report.md" --figures-dir none >>"${log}" 2>&1
  "${python_bin}" scripts/generate_sam_qualitative.py \
    --prediction-dir "${output}/predictions" --fold "${fold}" --split test \
    --output-dir "${output}/qualitative" >>"${log}" 2>&1
  for required in best_train.pt best_val.pt best_selection.pt last.pt \
      train_batch_metrics.csv epoch_metrics.csv quantitative_metrics.json \
      prediction_validation.json early_stopping.json qualitative/SAM_ERROR_ANALYSIS.md; do
    test -f "${output}/${required}"
  done
  "${python_bin}" - "${fold}" "${completion}" <<'PY'
import json, sys
from datetime import datetime, timezone
fold, path = int(sys.argv[1]), sys.argv[2]
with open(path, "w") as handle:
    json.dump({"status": "completed", "fold": fold, "split": "test",
               "completed_utc": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
    handle.write("\n")
PY
done

"${python_bin}" scripts/summarize_sam_test_cv.py --experiment-dir "${base}" \
  --report "reports/${experiment}_7-1-2_boundary_100.md"
