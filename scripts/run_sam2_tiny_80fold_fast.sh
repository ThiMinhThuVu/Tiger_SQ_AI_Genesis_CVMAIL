#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
experiment="artifacts/new_baselines/sam2_hiera_tiny_partial"
lock_file="${experiment}/.training.lock"
mkdir -p "${experiment}" logs
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "Another SAM training pipeline holds ${lock_file}" >&2
  exit 73
fi

"${python_bin}" scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
  --status running --last-completed-phase primary_80_epoch_cross_validation_started

completed=()
for fold in 0 1 2 3 4; do
  output="${experiment}/fold_${fold}"
  completion="${output}/completion.json"
  if [[ "${fold}" == 0 && -f "${output}/official_native_fast_validation.json" ]]; then
    mkdir -p "${output}"
    "${python_bin}" - "${completion}" <<'PY'
import json, sys
from datetime import datetime, timezone
path = sys.argv[1]
json.dump({"status":"completed", "fold":0, "completed_utc":datetime.now(timezone.utc).isoformat(), "oracle_prompt":False, "native_evaluator":"exact_edt_compat"}, open(path, "w"), indent=2)
open(path, "a").write("\n")
PY
  fi
  if [[ -f "${completion}" ]]; then
    echo "Fold ${fold} already complete; preserving artifacts"
    completed+=("${fold}")
    continue
  fi

  mkdir -p "${output}"
  log="logs/sam2_hiera_tiny_partial_fold_${fold}_80_fast.log"
  resume=()
  if [[ -f "${output}/last.pt" ]]; then resume+=(--resume); fi
  set +e
  "${python_bin}" scripts/train_sam.py \
    --model sam2_hiera_tiny --fold "${fold}" --n-folds 5 --epochs 80 \
    --height 512 --width 896 --batch-size 1 --gradient-accumulation 4 \
    --gradient-checkpointing --finetune partial --mixed-precision bf16 \
    --num-workers 4 --seed 2026 --run-kind primary --output-dir "${output}" \
    "${resume[@]}" >"${log}" 2>&1
  code=$?
  set -e
  echo "${code}" >"${output}/train_exit_code.txt"
  if [[ ${code} -ne 0 ]]; then
    "${python_bin}" scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
      --status failed --last-completed-phase "fold_${fold}_training_failed" \
      --failure-reason "80-epoch training exit ${code}; see ${log}"
    exit "${code}"
  fi

  "${python_bin}" scripts/predict_sam.py --checkpoint "${output}/best.pt" \
    --fold "${fold}" --output-dir "${output}/predictions" >>"${log}" 2>&1
  "${python_bin}" scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
    --fold "${fold}" --output "${output}/prediction_validation.json" >>"${log}" 2>&1
  "${python_bin}" scripts/validate_predictions.py --prediction-dir "${output}/predictions" \
    --fold "${fold}" --run-official --official-fixture \
    --output "${output}/official_fixture_validation.json" >>"${log}" 2>&1
  "${python_bin}" scripts/prepare_official_native_input.py \
    --prediction-dir "${output}/predictions" --fold "${fold}" \
    --destination "${output}/official_native_input" >>"${log}" 2>&1
  "${python_bin}" scripts/run_native_official_fast.py \
    --gt "${output}/official_native_input/gt" \
    --pred "${output}/official_native_input/predictions" \
    --out "${output}/official_native_fast_report.md" \
    --figures-dir none >>"${log}" 2>&1
  "${python_bin}" scripts/generate_sam_qualitative.py --prediction-dir "${output}/predictions" \
    --fold "${fold}" --output-dir "${output}/qualitative" >>"${log}" 2>&1
  "${python_bin}" - "${fold}" "${completion}" <<'PY'
import json, sys
from datetime import datetime, timezone
fold, path = int(sys.argv[1]), sys.argv[2]
json.dump({"status":"completed", "fold":fold, "completed_utc":datetime.now(timezone.utc).isoformat(), "oracle_prompt":False, "native_evaluator":"exact_edt_compat"}, open(path, "w"), indent=2)
open(path, "a").write("\n")
PY
  completed+=("${fold}")
  completed_json="[$(IFS=,; echo "${completed[*]}")]"
  "${python_bin}" scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
    --status running --last-completed-phase "fold_${fold}_80_epoch_completed" \
    --completed-folds "${completed_json}"
done

"${python_bin}" scripts/summarize_sam_cv.py
"${python_bin}" scripts/update_experiment_registry.py sam2_hiera_tiny_partial \
  --status completed --last-completed-phase five_fold_80_epoch_primary_complete \
  --completed-folds '[0,1,2,3,4]'
