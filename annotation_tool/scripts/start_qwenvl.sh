#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the local Qwen3-VL checkpoint directory}"

QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-22003}"
QWEN_TP="${QWEN_TP:-4}"

if [[ ! -f "${QWEN_MODEL_PATH}/config.json" ]]; then
  echo "Qwen config not found: ${QWEN_MODEL_PATH}/config.json" >&2
  exit 2
fi

exec python -m sglang.launch_server \
  --model-path "${QWEN_MODEL_PATH}" \
  --host "${QWEN_HOST}" \
  --port "${QWEN_PORT}" \
  --tp "${QWEN_TP}" \
  "$@"
