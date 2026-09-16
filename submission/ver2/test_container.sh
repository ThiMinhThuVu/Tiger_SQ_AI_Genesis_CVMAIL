#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task="${1:?usage: test_container.sh task2|task3 IMAGE_TAG INPUT_DIR [OUTPUT_DIR]}"
image_tag="${2:?usage: test_container.sh task2|task3 IMAGE_TAG INPUT_DIR [OUTPUT_DIR]}"
input_dir="$(cd "${3:?usage: test_container.sh task2|task3 IMAGE_TAG INPUT_DIR [OUTPUT_DIR]}" && pwd)"
output_dir="${4:-${submission_dir}/test_output_${task}}"
case "${task}" in task2|task3) ;; *) echo "task must be task2 or task3" >&2; exit 2 ;; esac
mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
if find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Output directory must be empty: ${output_dir}" >&2; exit 2
fi
docker run --rm --platform linux/amd64 --gpus all --network none --shm-size 8g \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --volume "${input_dir}:/input:ro" --volume "${output_dir}:/output:rw" "${image_tag}"
docker run --rm --platform linux/amd64 --network none --entrypoint python \
  --volume "${input_dir}:/input:ro" --volume "${output_dir}:/output:ro" "${image_tag}" \
  /opt/algorithm/validate.py /input /output
