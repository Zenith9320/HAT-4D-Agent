#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3D_HOST="${SAM3D_HOST:-127.0.0.1}"
SAM3D_PORT="${SAM3D_PORT:-8015}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${TOOL_ROOT}/checkpoints/sam/sam_vit_h_4b8939.pth}"
SAM3D_PIPELINE_CONFIG="${SAM3D_PIPELINE_CONFIG:-${TOOL_ROOT}/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/checkpoints/hf/pipeline.yaml}"

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${SAM3D_PIPELINE_CONFIG}" ]]; then
  echo "SAM3D pipeline config not found: ${SAM3D_PIPELINE_CONFIG}" >&2
  exit 2
fi

cd "${TOOL_ROOT}"
exec python viewer_plugins/sam3d_gen/backend.py \
  --host "${SAM3D_HOST}" \
  --port "${SAM3D_PORT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --sam3d-pipeline-config "${SAM3D_PIPELINE_CONFIG}" \
  --no-flash-attn \
  --one-dino \
  "$@"
