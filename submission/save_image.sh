#!/usr/bin/env bash
set -euo pipefail

submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_tag="${1:?usage: save_image.sh IMAGE_TAG TEAM_NAME [DESTINATION_DIR]}"
team_name="${2:?usage: save_image.sh IMAGE_TAG TEAM_NAME [DESTINATION_DIR]}"
destination_dir="${3:-${submission_dir}/dist}"

if [[ ! "${team_name}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "TEAM_NAME may contain only letters, digits, '_' and '-'" >&2
    exit 2
fi

mkdir -p "${destination_dir}"
archive="${destination_dir}/tigersqai_${team_name}_task123.tar.gz"
docker image inspect "${image_tag}" >/dev/null
docker save "${image_tag}" | gzip -c > "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
echo "Saved ${archive}"
