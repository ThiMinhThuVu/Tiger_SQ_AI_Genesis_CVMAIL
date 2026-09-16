#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
base="${1:-${submission_dir}/dist/tigersqai_submission_task123_v1.tar}"
output="${2:-${submission_dir}/dist/tigersqai_Genesis_CVMAIL_ver1_task123.tar}"
artifacts="${repository_dir}/full_version/center_stratified_32_8/artifacts"
python "${submission_dir}/assemble_image.py" \
  --base "${base}" --context "${submission_dir}" \
  --version-context "${submission_dir}/ver1" \
  --task1 "${artifacts}/center_strat32x8_task1_coarse_best" \
  --task2 "${artifacts}/center_strat32x8_task2_fine_improved_v2" \
  --task3 "${artifacts}/task3_visibility" \
  --output "${output}" --tag tigersqai26_genesis_cvmail:ver1
sha256sum "${output}" > "${output}.sha256"
echo "Assembled ${output}"
