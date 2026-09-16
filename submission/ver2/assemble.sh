#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
task="${1:?usage: assemble.sh task2|task3 [BASE_IMAGE_TAR] [OUTPUT_TAR]}"
base="${2:-${submission_dir}/dist/tigersqai_submission_task123_v1.tar}"
artifacts="${repository_dir}/full_version/center_stratified_32_8/artifacts"
output="${3:-${submission_dir}/dist/tigersqai_Genesis_CVMAIL_ver2_${task}.tar}"

case "${task}" in
  task2)
    args=(--task2 "${artifacts}/center_strat32x8_task2_fine_p2_hr_improved_v2")
    ;;
  task3)
    args=(
      --task2 "${artifacts}/center_strat32x8_task2_fine_improved_v2"
      --task3 "${artifacts}/task3_visibility_csv_20260907"
    )
    ;;
  *) echo "task must be task2 or task3" >&2; exit 2 ;;
esac

python "${submission_dir}/assemble_image.py" \
  --base "${base}" --context "${submission_dir}" \
  --version-context "${submission_dir}/ver2" --standalone-task "${task}" \
  "${args[@]}" --output "${output}" \
  --tag "tigersqai26_genesis_cvmail:ver2-${task}"
sha256sum "${output}" > "${output}.sha256"
echo "Assembled ${output}"
