#!/usr/bin/env bash
set -euo pipefail
submission_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task="${1:?usage: save_image.sh task2|task3 [IMAGE_TAG] [TEAM_NAME]}"
image_tag="${2:-tigersqai26_genesis_cvmail:ver2-${task}}"
team_name="${3:-Genesis_CVMAIL}"
case "${task}" in task2|task3) ;; *) echo "task must be task2 or task3" >&2; exit 2 ;; esac
archive="${submission_dir}/dist/tigersqai_${team_name}_ver2_${task}.tar.gz"
docker image inspect "${image_tag}" >/dev/null
docker save "${image_tag}" | gzip -c > "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
echo "Saved ${archive}"
