#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
image_tag="${1:-tigersqai26_genesis_cvmail:ver2-task1-d517}"
checkpoint="${repository_dir}/full_version/center_stratified_32_8/artifacts/center_strat32x8_d517_task1_coarse_best"
vendor_dir="$(mktemp -d /tmp/tigersqai-ver2-task1-vendor.XXXXXX)"
trap 'rm -rf "${vendor_dir}"' EXIT

for fold in 0 1 2 3 4; do test -f "${checkpoint}/fold_${fold}/best.pt"; done
python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements.txt"
python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements-nodeps.txt"
docker build --platform linux/amd64 --build-context "task1=${checkpoint}" \
  --build-context "vendor=${vendor_dir}" \
  --file "${submission_dir}/ver2_task1/task1/Dockerfile" \
  --tag "${image_tag}" "${submission_dir}"
echo "Built ${image_tag}"
