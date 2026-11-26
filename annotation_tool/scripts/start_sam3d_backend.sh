#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3D_ENV="${SAM3D_ENV:-/data/hat4d/conda/envs/sam3d-objects}"
SAM3D_PYTHON="${SAM3D_PYTHON:-${SAM3D_ENV}/bin/python}"
SAM3D_HOST="${SAM3D_HOST:-127.0.0.1}"
SAM3D_PORT="${SAM3D_PORT:-8015}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-${TOOL_ROOT}/checkpoints/sam/sam_vit_h_4b8939.pth}"
SAM3D_PIPELINE_CONFIG="${SAM3D_PIPELINE_CONFIG:-${TOOL_ROOT}/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/checkpoints/hf/pipeline.yaml}"

export TMPDIR="${TMPDIR:-/data/hat4d/tmp}"
export TORCH_HOME="${TORCH_HOME:-/data/hat4d/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/hat4d/cache/torch-extensions/hat4d-sam3d}"
export HF_HOME="${HF_HOME:-/data/hat4d/cache/huggingface}"
export LIDRA_SKIP_INIT="${LIDRA_SKIP_INIT:-1}"
export LD_LIBRARY_PATH="${SAM3D_ENV}/lib:${SAM3D_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${HF_HOME}"

if [[ ! -x "${SAM3D_PYTHON}" ]]; then
  echo "SAM3D Python not found: ${SAM3D_PYTHON}" >&2
  exit 2
fi

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${SAM3D_PIPELINE_CONFIG}" ]]; then
  echo "SAM3D pipeline config not found: ${SAM3D_PIPELINE_CONFIG}" >&2
  exit 2
fi

cd "${TOOL_ROOT}"
exec "${SAM3D_PYTHON}" viewer_plugins/sam3d_gen/backend.py \
  --host "${SAM3D_HOST}" \
  --port "${SAM3D_PORT}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --sam3d-pipeline-config "${SAM3D_PIPELINE_CONFIG}" \
  --no-flash-attn \
  --one-dino \
  "$@"
