#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
image_tag="${1:-tigersqai26_genesis_cvmail:ver1}"
experiment="${repository_dir}/full_version/center_stratified_32_8/artifacts"
task1="${experiment}/center_strat32x8_task1_coarse_best"
task2="${experiment}/center_strat32x8_task2_fine_improved_v2"
task3="${experiment}/task3_visibility"
vendor_dir="$(mktemp -d /tmp/tigersqai-ver1-vendor.XXXXXX)"
trap 'rm -rf "${vendor_dir}"' EXIT
for fold in 0 1 2 3 4; do
  test -f "${task1}/fold_${fold}/best.pt"
  test -f "${task2}/fold_${fold}/best.pt"
  test -f "${task3}/fold_${fold}/best_visibility.pt"
done
python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements.txt"
python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements-nodeps.txt"
docker build --platform linux/amd64 \
  --build-context "task1=${task1}" --build-context "task2=${task2}" \
  --build-context "task3=${task3}" --build-context "vendor=${vendor_dir}" \
  --file "${submission_dir}/ver1/Dockerfile" --tag "${image_tag}" "${submission_dir}"
echo "Built ${image_tag}"

