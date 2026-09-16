#!/usr/bin/env bash
set -euo pipefail

submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
image_tag="${1:-tigersqai26_submission:v1}"

segmentation_dir="${repository_dir}/artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2"
visibility_dir="${repository_dir}/artifacts/mask2former_improved/mask2former_swin_small_clinical_hier_presence_v2_task3"
vendor_dir="$(mktemp -d /tmp/tigersqai-cp311-vendor.XXXXXX)"

cleanup() {
    rm -rf "${vendor_dir}"
}
trap cleanup EXIT

for fold in 0 1 2 3 4; do
    test -f "${segmentation_dir}/fold_${fold}/best.pt"
    test -f "${visibility_dir}/fold_${fold}/best_visibility.pt"
done

python -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-compile \
    --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --abi cp311 \
    --target "${vendor_dir}" \
    --requirement "${submission_dir}/requirements.txt"

python -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-compile \
    --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --abi cp311 \
    --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    --target "${vendor_dir}" \
    --requirement "${submission_dir}/requirements-nodeps.txt"

docker build \
    --platform linux/amd64 \
    --build-context "segmentation=${segmentation_dir}" \
    --build-context "visibility=${visibility_dir}" \
    --build-context "vendor=${vendor_dir}" \
    --file "${submission_dir}/Dockerfile" \
    --tag "${image_tag}" \
    "${submission_dir}"

echo "Built ${image_tag}"
