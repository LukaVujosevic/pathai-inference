#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
INFERENCE_DIR="${INFERENCE_DIR:-${WORKSPACE}/Inference}"
ENV_FILE="${ENV_FILE:-${INFERENCE_DIR}/.env.runpod.512}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run scripts/runpod_setup_512.sh first." >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
  echo "Missing MODEL_CHECKPOINT=${MODEL_CHECKPOINT}" >&2
  echo "Run strip_checkpoint.py after uploading the original .pth." >&2
  exit 1
fi

export PYTHONPATH="${MODEL_CODE_DIR}:${INFERENCE_DIR}:${PYTHONPATH:-}"

cd "${INFERENCE_DIR}"
python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
