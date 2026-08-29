#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROI_HOST="${ROI_HOST:-127.0.0.1}"
ROI_PORT="${ROI_PORT:-8002}"
INIT_SV4D="${INIT_SV4D:-1}"

cd "${TOOL_ROOT}"

if [[ "${SKIP_MODEL_CHECKS:-0}" != "1" ]]; then
  L4GM_WEIGHT="${L4GM_CHECKPOINT:-${TOOL_ROOT}/model4dgen/L4GM_MoGA/pretrained/recon.safetensors}"
  if [[ ! -f "${L4GM_WEIGHT}" ]]; then
    echo "L4GM checkpoint not found: ${L4GM_WEIGHT}" >&2
    exit 2
  fi
  if [[ "${INIT_SV4D}" == "1" ]]; then
    SV4D_WEIGHT="${SV4D_CHECKPOINT:-${TOOL_ROOT}/model4dgen/generative-models/checkpoints/sv4d2.0/sv4d2_with_first_stage_encoder_v2.safetensors}"
    if [[ ! -f "${SV4D_WEIGHT}" ]]; then
      echo "SV4D checkpoint not found: ${SV4D_WEIGHT}" >&2
      exit 2
    fi
  fi
fi

args=(--host "${ROI_HOST}" --port "${ROI_PORT}")
if [[ "${INIT_SV4D}" == "1" ]]; then
  args+=(--l4gm-init-sv4d)
fi

exec python roi_backend.py "${args[@]}" "$@"
