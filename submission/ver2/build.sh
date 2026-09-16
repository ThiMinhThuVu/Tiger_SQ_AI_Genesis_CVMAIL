#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_dir="$(cd "${submission_dir}/.." && pwd)"
task="${1:?usage: build.sh task2|task3 [IMAGE_TAG]}"
image_tag="${2:-tigersqai26_genesis_cvmail:ver2-${task}}"
artifacts="${repository_dir}/full_version/center_stratified_32_8/artifacts"
vendor_dir="$(mktemp -d /tmp/tigersqai-ver2-vendor.XXXXXX)"
trap 'rm -rf "${vendor_dir}"' EXIT

case "${task}" in
  task2)
    checkpoint="${artifacts}/center_strat32x8_task2_fine_p2_hr_improved_v2"
    for fold in 0 1 2 3 4; do test -f "${checkpoint}/fold_${fold}/best.pt"; done
    contexts=(--build-context "task2=${checkpoint}")
    ;;
  task3)
    encoder="${artifacts}/center_strat32x8_task2_fine_improved_v2"
    head="${artifacts}/task3_visibility_csv_20260907"
    for fold in 0 1 2 3 4; do
      test -f "${encoder}/fold_${fold}/best.pt"
      test -f "${head}/fold_${fold}/best_visibility.pt"
    done
    contexts=(--build-context "encoder=${encoder}" --build-context "head=${head}")
    ;;
  *) echo "task must be task2 or task3" >&2; exit 2 ;;
esac

python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements.txt"
python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
  --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  --python-version 3.11 --abi cp311 --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 --target "${vendor_dir}" \
  --requirement "${submission_dir}/requirements-nodeps.txt"
docker build --platform linux/amd64 "${contexts[@]}" \
  --build-context "vendor=${vendor_dir}" \
  --file "${submission_dir}/ver2/${task}/Dockerfile" \
  --tag "${image_tag}" "${submission_dir}"
echo "Built ${image_tag}"
