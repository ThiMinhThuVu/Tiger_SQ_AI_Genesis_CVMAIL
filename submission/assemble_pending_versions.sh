#!/usr/bin/env bash
set -euo pipefail

submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
base="${submission_dir}/dist/tigersqai_submission_task123_v1.tar"
context="${submission_dir}"
version_context="${submission_dir}/ver2"
dist="${submission_dir}/dist"
center_artifacts="${repository_dir}/full_version/center_stratified_32_8/artifacts"
cetl_artifacts="${repository_dir}/full_version/task2_fine_31cls/artifacts/p3_cetl_w200_pilot/p3_hr640x1120_cetl_w200_seed2026"

assemble_task2() {
  local version="$1"
  local suffix="$2"
  local checkpoints="$3"
  local height="$4"
  local width="$5"
  local output="${dist}/tigersqai_Genesis_CVMAIL_${version}_task2_${suffix}.tar"
  python "${submission_dir}/assemble_image.py" \
    --base "${base}" --context "${context}" --version-context "${version_context}" \
    --standalone-task task2 --task2 "${checkpoints}" \
    --task2-height "${height}" --task2-width "${width}" \
    --output "${output}" --tag "tigersqai26_genesis_cvmail:${version}-task2-${suffix}"
  sha256sum "${output}" >"${output}.sha256"
  gzip -1 -f -k "${output}"
  sha256sum "${output}.gz" >"${output}.gz.sha256"
}

assemble_task3() {
  local version="$1"
  local suffix="$2"
  local encoder="$3"
  local head="$4"
  local height="$5"
  local width="$6"
  local output="${dist}/tigersqai_Genesis_CVMAIL_${version}_task3_${suffix}.tar"
  python "${submission_dir}/assemble_image.py" \
    --base "${base}" --context "${context}" --version-context "${version_context}" \
    --standalone-task task3 --task2 "${encoder}" --task3 "${head}" \
    --task3-height "${height}" --task3-width "${width}" \
    --output "${output}" --tag "tigersqai26_genesis_cvmail:${version}-task3-${suffix}"
  sha256sum "${output}" >"${output}.sha256"
  gzip -1 -f -k "${output}"
  sha256sum "${output}.gz" >"${output}.gz.sha256"
}

assemble_task2 ver3 cetl "${cetl_artifacts}" 640 1120
assemble_task2 ver4 p0 "${center_artifacts}/center_strat32x8_d517_task2_fine_improved_v2" 512 896
assemble_task2 ver5 p2 "${center_artifacts}/center_strat32x8_d517_task2_fine_p2_hr_improved_v2" 640 1120

assemble_task3 ver3 p2 \
  "${center_artifacts}/center_strat32x8_task2_fine_p2_hr_improved_v2" \
  "${center_artifacts}/task3_visibility_p2" 640 1120
assemble_task3 ver4 p0 \
  "${center_artifacts}/center_strat32x8_d517_task2_fine_improved_v2" \
  "${center_artifacts}/d517_task3_visibility" 512 896
assemble_task3 ver5 p2 \
  "${center_artifacts}/center_strat32x8_d517_task2_fine_p2_hr_improved_v2" \
  "${center_artifacts}/d517_task3_visibility_p2" 640 1120

echo "All pending task archives assembled."
