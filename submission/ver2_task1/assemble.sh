#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
base="${1:-${submission_dir}/dist/tigersqai_submission_task123_v1.tar}"
output="${2:-${submission_dir}/dist/tigersqai_Genesis_CVMAIL_ver2_task1_d517.tar}"
task1="${repository_dir}/full_version/center_stratified_32_8/artifacts/center_strat32x8_d517_task1_coarse_best"

for fold in 0 1 2 3 4; do test -f "${task1}/fold_${fold}/best.pt"; done
python "${submission_dir}/assemble_image.py" \
  --base "${base}" --context "${submission_dir}" \
  --version-context "${submission_dir}/ver2_task1" --standalone-task task1 \
  --task1 "${task1}" --output "${output}" \
  --tag "tigersqai26_genesis_cvmail:ver2-task1-d517"
sha256sum "${output}" > "${output}.sha256"
echo "Assembled ${output}"
