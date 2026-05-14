#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
IMAGE_PATH="${1:-}"

curl -fsS "${BASE_URL}/health"
echo

if [[ -n "${IMAGE_PATH}" ]]; then
  curl -fsS -X POST "${BASE_URL}/segment" \
    -F "image=@${IMAGE_PATH}" \
    -F "mpp=1.0" \
    -F "image_id=smoke" | python -m json.tool
else
  echo "Health OK. Pass a PNG path to also test /segment."
fi
