# HAT-4D Annotation Tool: English Guide

[中文](README_ch.md) | [UI workflow](../docs/ui_README.md) | [Back to HAT-4D](../README.md)

The tool turns a monocular RGB frame sequence into an editable 4D Gaussian sequence. A browser frontend coordinates two GPU backends:

| Process | Default address | Responsibility |
| --- | --- | --- |
| QwenVL via SGLang | `http://127.0.0.1:22003/v1` | Optional video understanding and Human-Agent decisions |
| ROI/L4GM backend | `http://127.0.0.1:8002` | 4D generation, propagation, ROI repair, and SV4D refinement |
| SAM3D backend | `http://127.0.0.1:8015` | Point-prompt masks, 3D object generation, and pose optimization |
| Viser frontend | `http://127.0.0.1:8897` | Interactive annotation, inspection, and workspace persistence |

QwenVL is optional for fully manual annotation. The two model backends and frontend are required for the complete workflow. Run all processes on the same machine unless you replace every `127.0.0.1` URL with a reachable host and secure the exposed ports.

## 1. What Is Included

- Frontend, editor plugins, VLM agents, backend adapters, and service tests.
- The HAT-4D L4GM/MoGA runtime source under `model4dgen/L4GM_MoGA`.
- The modified SV4D2 runtime source under `model4dgen/generative-models`.
- Pinned installers for SAM 3D Objects, Segment Anything, and Qwen3-VL source.

The full MVOIK-4D dataset and model weights are intentionally excluded. Two reviewed demo subsets are included under [`asset/`](../asset/README.md) to make the annotation workflow directly testable; their CC BY-NC 4.0 data terms are separate from the source-code license. L4GM model weights are subject to their model license, including non-commercial restrictions described by the model provider. SAM3D and Qwen checkpoints have their own terms. Review all model licenses before downloading or redistributing weights.

## 2. Prerequisites

- Linux, NVIDIA CUDA GPUs, Conda/Mamba, Git, and Python 3.10 or 3.11.
- Enough local storage for Qwen3-VL, SAM3D, L4GM, and SV4D checkpoints.
- Separate environments for Qwen/SGLang, ROI/L4GM, and SAM3D. Their CUDA and PyTorch constraints are different.

An example 8-GPU allocation is QwenVL on GPUs `0,1,2,3`, ROI/L4GM on GPU `4`, and SAM3D on GPU `5`. Adjust this to the memory capacity of your hardware. Do not place the two generation backends on the same nearly-full GPU.

## 3. Install Source Dependencies

From the HAT-4D repository root:

```bash
python annotation_tool/scripts/install_external_dependencies.py \
  --only sam-3d-objects \
  --only segment-anything \
  --only Qwen3-VL
```

The installer uses immutable revisions from `external_dependencies.json`. Qwen3-VL source is optional when SGLang is installed from its package and the checkpoint is stored elsewhere.

### Frontend environment

```bash
conda create -n hat4d-frontend python=3.11 -y
conda activate hat4d-frontend
python -m pip install --upgrade pip
python -m pip install -r annotation_tool/requirements.txt
```

The public runtime uses `viser==1.0.30`. A locally modified Viser checkout under
`annotation_tool/thirdparty/viser` is optional and intentionally excluded from
the release. The tool checks for its non-standard keyboard callback at runtime,
so the official package remains fully usable without the custom shortcuts.

### ROI/L4GM environment

```bash
conda create -n hat4d-roi python=3.10 -y
conda activate hat4d-roi
python -m pip install --upgrade pip
python -m pip install -r annotation_tool/requirements.txt
python -m pip install -r annotation_tool/requirements_roi.txt
```

Install the PyTorch, xFormers, gsplat, and CUDA builds appropriate for your GPU before resolving the remaining packages if the default wheels do not match your system.

### SAM3D environment

Follow `annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/doc/setup.md`. The upstream project currently requires a Linux CUDA environment and gated checkpoints. Also install Segment Anything in that environment:

```bash
conda activate sam3d-objects
python -m pip install -e \
  annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/thirdparty/segment-anything
python -m pip install viser==1.0.30 imageio scipy
```

### QwenVL environment

Create a dedicated environment and install a SGLang release compatible with your CUDA driver and Qwen3-VL checkpoint. The launcher follows SGLang's OpenAI-compatible server interface:

```bash
conda create -n hat4d-qwen python=3.11 -y
conda activate hat4d-qwen
python -m pip install --upgrade pip
python -m pip install sglang
```

## 4. Place Checkpoints

The runtime expects these files or explicit environment-variable overrides:

| Model | Expected path |
| --- | --- |
| L4GM reconstruction | `annotation_tool/model4dgen/L4GM_MoGA/pretrained/recon.safetensors`, or set `L4GM_CHECKPOINT` |
| HAT-4D SV4D2 encoder variant | `annotation_tool/model4dgen/generative-models/checkpoints/sv4d2.0/sv4d2_with_first_stage_encoder_v2.safetensors`, or set `SV4D_CHECKPOINT` |
| SAM | `annotation_tool/checkpoints/sam/sam_vit_h_4b8939.pth`, or set `SAM_CHECKPOINT` |
| SAM3D Objects | `.../sam-3d-objects/checkpoints/hf/pipeline.yaml`, or set `SAM3D_PIPELINE_CONFIG` |
| Qwen3-VL | Any local checkpoint directory, passed as `QWEN_MODEL_PATH` |

The SV4D weight is the HAT-4D encoder variant, not the standard upstream `sv4d2.safetensors`. Renaming a different checkpoint will not make it compatible.

Download the official L4GM and Segment Anything weights from the repository
root:

```bash
python -m pip install 'huggingface-hub[cli]<1.0'

mkdir -p annotation_tool/model4dgen/L4GM_MoGA/pretrained
hf download jiawei011/L4GM recon.safetensors \
  --local-dir annotation_tool/model4dgen/L4GM_MoGA/pretrained

mkdir -p annotation_tool/checkpoints/sam
curl --fail --location \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  --output annotation_tool/checkpoints/sam/sam_vit_h_4b8939.pth
```

SAM3D Objects is gated. After access is approved and `hf auth login` has been
run locally:

```bash
SAM3D_ROOT=annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects
hf download facebook/sam-3d-objects \
  --repo-type model \
  --local-dir "$SAM3D_ROOT/checkpoints/hf-download" \
  --max-workers 1
mv "$SAM3D_ROOT/checkpoints/hf-download/checkpoints" \
  "$SAM3D_ROOT/checkpoints/hf"
rm -rf "$SAM3D_ROOT/checkpoints/hf-download"
```

Obtain `sv4d2_with_first_stage_encoder_v2.safetensors` from the HAT-4D model
release. The public Stability AI `sv4d2.safetensors` is an upstream base model,
not a substitute for this encoder-variant checkpoint. Keep authentication tokens
outside the repository and review the L4GM, SAM, SAM3D, SV4D, and Qwen model
licenses before use.

## 5. Prepare a Case

Frames must be directly inside the input directory. Numeric, zero-padded names are recommended:

```text
data/example_case/
├── input/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── eval_gt/
    ├── eval_0/000000.png ...
    ├── eval_1/000000.png ...
    ├── eval_2/000000.png ...
    └── eval_3/000000.png ...
```

`eval_gt` is required only for online PSNR/LPIPS evaluation, but the launcher accepts a directory unconditionally. Use an empty directory when no multiview ground truth is available and leave `ONLINE_METRIC` unset.

## 6. Start the Services

Use four terminals or tmux windows. Commands below run from the repository root.

For a machine that already has the original source-tested environments, use
`4dgsEditor` for the ROI backend and `4dgsEditor_sam3d_v2` for both SAM3D and
the frontend. Skip Terminal A entirely for manual annotation without VLA/VLM
guidance.

### Terminal A: QwenVL (optional)

```bash
conda activate hat4d-qwen
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-235B-A22B-Instruct
export QWEN_TP=4
annotation_tool/scripts/start_qwenvl.sh
```

Wait until the model endpoint responds:

```bash
curl --fail http://127.0.0.1:22003/v1/models
```

Use the returned model ID as `VLM_LOCAL_MODEL` if it differs from `Qwen/Qwen3-VL-235B-A22B-Instruct`.

### Terminal B: ROI/L4GM backend

```bash
conda activate hat4d-roi
export CUDA_VISIBLE_DEVICES=4
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-roi"
export L4GM_CHECKPOINT=/path/to/recon.safetensors
export SV4D_CHECKPOINT=/path/to/sv4d2_with_first_stage_encoder_v2.safetensors
annotation_tool/scripts/start_roi_backend.sh
```

### Terminal C: SAM3D backend

```bash
conda activate sam3d-objects
export CUDA_VISIBLE_DEVICES=5
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-sam3d"
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export SAM3D_PIPELINE_CONFIG=/path/to/sam-3d-objects/checkpoints/hf/pipeline.yaml
annotation_tool/scripts/start_sam3d_backend.sh
```

### Terminal D: frontend

```bash
conda activate hat4d-frontend
export INPUT_IMAGE_DIR=/path/to/data/example_case/input
export EVAL_GT_DIR=/path/to/data/example_case/eval_gt
export WORKSPACE_ROOT=/path/to/results/example_case
export VLM_LOCAL_BASE_URL=http://127.0.0.1:22003/v1
export VLM_LOCAL_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
export VLM_LOCAL_API_KEY=EMPTY
annotation_tool/scripts/start_frontend.sh
```

Open `http://127.0.0.1:8897`. To enable online metrics, add `export ONLINE_METRIC=1` before starting the frontend.

For a remote OpenAI-compatible endpoint, set its root URL only in the current
shell and derive the API URL without recording the private address in a tracked
file:

```bash
export BASE_URL="https://your-qwen-endpoint.example/proxy/8000"
export VLM_LOCAL_BASE_URL="${BASE_URL%/}/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
```

## 7. Annotation Workflow

### Initialize objects with SAM3D

1. Go to the frame that shows all target objects clearly.
2. In `SAM 3D Gen`, click `Begin SAM3D Gen`.
3. Enable `Pixel Pick Mode`. Click positive points on the object; use `Negative Point` to remove other regions and `Eraser Point` to delete prompts.
4. Inspect the mask, then click `Generate 3D (Current Mask)`.
5. Keep `Pose optimize after GEN` enabled for automatic alignment. If depth is still wrong, adjust the object transform, configure the axis locks in `Pose Opt`, and click `Optimize Pose (Mask Align)`.
6. Click `New Object` and repeat for every independently moving object.
7. Click `Add All Objects & Exit` only after every object is acceptable.

### Propagate through time

1. In `L4GM_4DGEN`, click `Generate next segment`. One segment normally covers the next 16 frames.
2. Click the button again when it reads `Pause Generate` if you want to inspect each segment before continuing.
3. Use the top-level play/pause control and inspect silhouettes, geometry, relative depth, contacts, and motion continuity.
4. When an intermediate frame is broken, go to that frame, enable `SAM 3D Gen > Total cover` before `Begin SAM3D Gen`, rebuild all visible objects, and resume propagation. Omitting `Total cover` mixes the replacement with the old frame PLY.

### Human-Agent mode

In `VLM Assistor`, verify the Qwen endpoint and model name, run Agent 0 video understanding, inspect its temporal scene graph, then run the object generation/validation pipeline. Treat agent output as a proposal: correct masks, pose, object contact, and failure frames before accepting the workspace.

## 8. Save and Resume

Every run creates a timestamped directory under `WORKSPACE_ROOT`. To resume exactly one workspace:

```bash
export RESUME_WORKSPACE=/path/to/results/example_case/workspace_YYYYMMDD_HHMMSS
annotation_tool/scripts/start_frontend.sh
```

Do not point `RESUME_WORKSPACE` at the parent results directory. The selected directory should contain `meta.json`, `gs_sequence/`, `images/`, `selections/`, and `results/`.

## 9. Verification and Troubleshooting

Model-free package checks:

```bash
python -m compileall -q \
  -x '(^|/)(thirdparty|tmp|checkpoints|\.cache)(/|$)' \
  annotation_tool
pytest -q annotation_tool/tests
python annotation_tool/scripts/check_publication.py annotation_tool
```

Before publishing, also inspect `git status --short`. Do not commit input videos,
evaluation images, workspaces, checkpoints, Hugging Face caches, tokens, private
service URLs, or machine-specific absolute paths. The scanner cannot recognize
faces, screens, badges, locations, or other private content inside images, so
media requires manual review.

Health-only checks do not load checkpoints:

```bash
python annotation_tool/roi_backend.py --health-only --port 8002
python annotation_tool/viewer_plugins/sam3d_gen/backend.py --health-only --port 8015
```

Common failures:

- `L4GM checkpoint not found`: check the exact `pretrained/recon.safetensors` path.
- `SV4D2 checkpoint not found`: use the encoder-variant filename and directory above.
- SAM3D reports `sam_ready: false`: verify `SAM_CHECKPOINT`, model type, and CUDA device.
- SAM3D reports `sam3d_ready: false`: verify gated weights and `SAM3D_PIPELINE_CONFIG`.
- Qwen requests fail while `/v1/models` works: set `VLM_LOCAL_MODEL` to the exact returned ID.
- The browser opens but backend actions fail: confirm port alignment and that all processes run on the same host.
- A resumed session appears empty: point `RESUME_WORKSPACE` to the timestamped workspace, not its parent.
