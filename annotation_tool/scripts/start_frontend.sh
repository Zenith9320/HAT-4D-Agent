#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_IMAGE_DIR:?Set INPUT_IMAGE_DIR to the directory containing video frames}"
: "${EVAL_GT_DIR:?Set EVAL_GT_DIR to the evaluation-view root}"
: "${WORKSPACE_ROOT:?Set WORKSPACE_ROOT to the output directory for this case}"

TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_HOST_PORT="${FRONTEND_PORT:-8897}"
ROI_URL="${ROI_URL:-http://127.0.0.1:8002/select_roi}"
SAM3D_URL="${SAM3D_URL:-http://127.0.0.1:8015}"

export VLM_LOCAL_BASE_URL="${VLM_LOCAL_BASE_URL:-http://127.0.0.1:22003/v1}"
export VLM_LOCAL_MODEL="${VLM_LOCAL_MODEL:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
export TMPDIR="${TMPDIR:-/data/hat4d/tmp}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/hat4d/cache/torch-extensions/hat4d-frontend}"
export CUDA_HOME="${CUDA_HOME:-/root/miniforge/envs/hat4d-roi}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-${CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}"

args=(
  --http-get-retries 2
  --http-post-retries 0
  --http-retry-backoff-ms 400
  --backend-url "${ROI_URL}"
  --sam3d-backend-url "${SAM3D_URL}"
  --input-image-dir "${INPUT_IMAGE_DIR}"
  --eval-gt-dir "${EVAL_GT_DIR}"
  --workspace-root "${WORKSPACE_ROOT}"
  --camera-target-origin
  --one-dino
  --no-raster-preview
  --profile-frame-load
  --profile-frame-threshold-ms 50
  --gs-display-fixed-points "${GS_DISPLAY_FIXED_POINTS:-3000000}"
  --use-pose-adjust-v2
  --port "${FRONTEND_HOST_PORT}"
)

if [[ -n "${RESUME_WORKSPACE:-}" ]]; then
  args+=(--resume-workspace "${RESUME_WORKSPACE}")
else
  args+=(--start-empty)
fi
if [[ -n "${GS_DIR:-}" ]]; then
  args+=(--gs-dir "${GS_DIR}")
fi
if [[ -n "${VLM_SHARED_DIR:-}" ]]; then
  args+=(--vlm-shared-dir "${VLM_SHARED_DIR}")
fi
if [[ "${ONLINE_METRIC:-0}" == "1" ]]; then
  args+=(--online-metric)
fi

cd "${TOOL_ROOT}"
exec python viser_frontend.py "${args[@]}" "$@"
