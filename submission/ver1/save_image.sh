#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${1:-tigersqai26_genesis_cvmail:ver1}"
team_name="${2:-Genesis_CVMAIL}"
destination_dir="${3:-${submission_dir}/dist}"
mkdir -p "${destination_dir}"
archive="${destination_dir}/tigersqai_${team_name}_ver1_task123.tar.gz"
docker image inspect "${image_tag}" >/dev/null
docker save "${image_tag}" | gzip -c > "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
echo "Saved ${archive}"

