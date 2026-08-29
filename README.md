<div align="center">

# HAT-4D

### Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration

**ECCV 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2606.28215-b31b1b.svg)](https://arxiv.org/pdf/2606.28215) [![Project Page](https://img.shields.io/badge/Project-HAT--4D-2b6cb0.svg)](https://lijiaxin0111.github.io/HAT4D/) [![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-MVOIK--4D-yellow.svg)](https://huggingface.co/datasets/Lijiaxin0111/MVOIK-4D) [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

<p align="center">
  <img src="asset/teaser_new.jpg" width="95%" alt="HAT-4D teaser">
</p>

## News 🔥

- **2026-06-30**: The **MVOIK-4D** dataset has been released on [Hugging Face](https://huggingface.co/datasets/Lijiaxin0111/MVOIK-4D).
- **2026-06-26**: HAT-4D is accepted to **ECCV 2026**.

## Overview ✨

HAT-4D is an agentic framework for reconstructing 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single monocular video. The project targets dynamic, in-the-wild object interactions where severe occlusion, depth ambiguity, and complex motion make conventional monocular 4D reconstruction difficult.

By integrating vision-language models with a multi-level human-in-the-loop feedback mechanism, HAT-4D resolves interaction-induced ambiguities during 3D generation and 4D propagation. The framework further serves as a scalable data engine for **MVOIK-4D**, an open-world benchmark for monocular 4D interaction reconstruction with evaluation focused on physical plausibility and temporal consistency.

## Method 🧠

<p align="center">
  <img src="asset/method_pipeline.jpg" width="95%" alt="HAT-4D method pipeline">
</p>

HAT-4D decomposes monocular interaction lifting into agent-assisted perception, human feedback, 3D asset generation, and temporally consistent 4D propagation. This design enables the system to recover physically plausible multi-object interactions without relying on expensive multi-camera capture setups.

## Annotation Tool

The HAT-4D annotation tool, its two model backends, QwenVL integration, and
reproducible launch scripts are included under [`annotation_tool`](annotation_tool/README.md).

- [English installation and annotation guide](annotation_tool/README_eng.md)
- [Chinese installation and annotation guide](annotation_tool/README_ch.md)
- [English UI guide: SAM3D initialization, 4D propagation, and frame repair](docs/ui_README.md)
- [Chinese UI guide](docs/ui_README_ch.md)

Model checkpoints and the full MVOIK-4D dataset are not bundled. Two reviewed
demo subsets are included under [`asset/`](asset/README.md) for testing the
annotation workflow; they are distributed under separate data terms. The
instructions below provide the shortest tested setup path; the language-specific
guides cover annotation, repair, and workspace recovery in detail.

## Installation

### 1. Clone HAT-4D and its pinned external source dependencies

```bash
git clone https://github.com/Lijiaxin0111/HAT-4D-Lifting-Monocular-Video-for-4D-Multi-Object-Interactions-via-Human-Agent-Collaboration.git
cd HAT-4D-Lifting-Monocular-Video-for-4D-Multi-Object-Interactions-via-Human-Agent-Collaboration

python annotation_tool/scripts/install_external_dependencies.py \
  --only sam-3d-objects \
  --only segment-anything \
  --only Qwen3-VL

# Verify every checkout against annotation_tool/external_dependencies.json.
python annotation_tool/scripts/install_external_dependencies.py --check
```

The installer clones SAM 3D Objects, Segment Anything, and Qwen3-VL at pinned
commits. Do not replace these revisions with arbitrary branch heads. Qwen3-VL
source is optional when an existing OpenAI-compatible Qwen endpoint is used.
The modified L4GM/MoGA and SV4D2 inference sources are already included in this
repository.

### 2. Create isolated environments

The ROI/L4GM and SAM3D stacks have different CUDA constraints and should not be
installed into one environment.

```bash
# Frontend.
conda create -n hat4d-frontend python=3.11 -y
conda activate hat4d-frontend
python -m pip install --upgrade pip
python -m pip install -r annotation_tool/requirements.txt

# ROI/L4GM backend.
conda create -n hat4d-roi python=3.10 -y
conda activate hat4d-roi
python -m pip install --upgrade pip
python -m pip install -r annotation_tool/requirements.txt
python -m pip install -r annotation_tool/requirements_roi.txt
```

Install SAM3D from its pinned checkout using its upstream CUDA setup:

```bash
cd annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects
mamba env create -f environments/default.yml
mamba activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
python -m pip install -e '.[dev]'
python -m pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
python -m pip install -e '.[inference]'
./patching/hydra
python -m pip install -e thirdparty/segment-anything
python -m pip install viser==1.0.30 imageio scipy
cd -
```

Use the PyTorch, xFormers, gsplat, and CUDA wheels that match the local driver.
The commands above follow the pinned SAM3D checkout; consult its
`doc/setup.md` if its binary packages do not match the machine. A locally
modified Viser checkout is optional. The public installation uses
`viser==1.0.30`; custom keyboard shortcuts are disabled gracefully when its
non-standard `ClientHandle.on_key()` API is unavailable.

QwenVL is optional for manual annotation. To host it locally, create a separate
environment and install an SGLang build compatible with the GPU and checkpoint:

```bash
conda create -n hat4d-qwen python=3.11 -y
conda activate hat4d-qwen
python -m pip install --upgrade pip
python -m pip install sglang
```

### 3. Download and place checkpoints

Install the Hugging Face CLI, authenticate only in the local user profile, and
never commit a token:

```bash
python -m pip install 'huggingface-hub[cli]<1.0'
hf auth login
```

| Checkpoint | Destination | Source |
| --- | --- | --- |
| L4GM reconstruction | `annotation_tool/model4dgen/L4GM_MoGA/pretrained/recon.safetensors` | `jiawei011/L4GM` |
| HAT-4D SV4D2 encoder variant | `annotation_tool/model4dgen/generative-models/checkpoints/sv4d2.0/sv4d2_with_first_stage_encoder_v2.safetensors` | HAT-4D model release |
| Segment Anything ViT-H | `annotation_tool/checkpoints/sam/sam_vit_h_4b8939.pth` | Meta Segment Anything |
| SAM 3D Objects | `annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/checkpoints/hf/pipeline.yaml` plus sibling files | gated `facebook/sam-3d-objects` repository |
| Qwen3-VL | any directory containing `config.json` | Qwen model repository, or an external compatible service |

Download the publicly hosted weights:

```bash
mkdir -p annotation_tool/model4dgen/L4GM_MoGA/pretrained
hf download jiawei011/L4GM recon.safetensors \
  --local-dir annotation_tool/model4dgen/L4GM_MoGA/pretrained

mkdir -p annotation_tool/checkpoints/sam
curl --fail --location \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  --output annotation_tool/checkpoints/sam/sam_vit_h_4b8939.pth

SAM3D_ROOT=annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects
hf download facebook/sam-3d-objects \
  --repo-type model \
  --local-dir "$SAM3D_ROOT/checkpoints/hf-download" \
  --max-workers 1
mv "$SAM3D_ROOT/checkpoints/hf-download/checkpoints" \
  "$SAM3D_ROOT/checkpoints/hf"
rm -rf "$SAM3D_ROOT/checkpoints/hf-download"
```

The HAT-4D SV4D2 encoder checkpoint is not the upstream
`stabilityai/sv4d2.0` `sv4d2.safetensors` file. Obtain the encoder-variant
checkpoint from the HAT-4D model release and keep its exact filename; renaming
the upstream checkpoint does not make it compatible. Review every model license
before downloading or redistributing weights.

## Test The Annotation Tool

The complete workflow uses one GPU for ROI/L4GM and one GPU for SAM3D. QwenVL
may be hosted elsewhere. Run each block from the repository root in a separate
terminal.

```bash
# Terminal 1: ROI/L4GM on GPU 0.
conda activate hat4d-roi
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-roi"
annotation_tool/scripts/start_roi_backend.sh
```

```bash
# Terminal 2: SAM3D on GPU 1.
conda activate sam3d-objects
export CUDA_VISIBLE_DEVICES=1
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-sam3d"
annotation_tool/scripts/start_sam3d_backend.sh
```

Confirm both backends before starting the UI:

```bash
curl --fail http://127.0.0.1:8002/health
curl --fail http://127.0.0.1:8015/health
```

For a remote OpenAI-compatible Qwen service, set `BASE_URL` only in the shell.
It must be the server root before `/v1`; do not put a private proxy URL in this
README or in a tracked `.env` file.

```bash
# Terminal 3: frontend using the included demo case.
conda activate hat4d-frontend

export BASE_URL="https://your-qwen-endpoint.example/proxy/8000"
export INPUT_IMAGE_DIR="$PWD/asset/demo_case_1/input"
export EVAL_GT_DIR="$PWD/asset/demo_case_1/eval_gt"
export WORKSPACE_ROOT="$PWD/results/demo_case_1"
export VLM_LOCAL_BASE_URL="${BASE_URL%/}/v1"
export VLM_LOCAL_MODEL="Qwen3vl"
export VLM_LOCAL_API_KEY="EMPTY"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

mkdir -p "$WORKSPACE_ROOT"
curl --fail "$BASE_URL/health"
curl --fail "$VLM_LOCAL_BASE_URL/models"
annotation_tool/scripts/start_frontend.sh
```

Open `http://127.0.0.1:8897`. When using the source-tested environments from
the original development machine, `4dgsEditor` can replace `hat4d-roi`, and
`4dgsEditor_sam3d_v2` can replace both `sam3d-objects` and `hat4d-frontend`.

### Publication and privacy checks

All services bind to `127.0.0.1` by default. Do not expose model or annotation
ports publicly without authentication and transport security. Private input
frames and evaluation views, workspaces, model weights, tokens, private endpoint
URLs, and machine-specific absolute paths must remain untracked. The only tracked
input/evaluation frames are the reviewed demo subsets documented in
[`asset/README.md`](asset/README.md).

Before publishing a change, run:

```bash
python -m compileall -q \
  -x '(^|/)(thirdparty|tmp|checkpoints|\.cache)(/|$)' \
  annotation_tool
pytest -q annotation_tool/tests
python annotation_tool/scripts/install_external_dependencies.py --check
python annotation_tool/scripts/check_publication.py \
  README.md docs/ui_README.md docs/ui_README_ch.md annotation_tool asset
git status --short
```

Review `git status` manually before every commit. The automated scan catches
common paths, credentials, generated artifacts, symlinks, and oversized files,
but cannot determine whether an image contains a face, screen, badge, location,
or other personally identifying content.


## ToDo List ✅

- [x] Open-source MVOIK-4D .
- [x] Open-source HAT-4D annotation toolkit.

## Citation 📝

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{Li2026hat4d,
  title     = {HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration},
  author    = {Li, Jiaxin and Author, Second and Author, Third},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  note      = {Accepted, to appear}
}
```

## References 🙏

We thank the following projects and systems for making this work possible:

- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [SV4D](https://github.com/Stability-AI/generative-models)
- [viser](https://github.com/nerfstudio-project/viser)

## License 📄

The project source code is released under the [Apache License 2.0](LICENSE).
The demo data under `asset/demo_case_1` and `asset/demo_case_2` is distributed
under [CC BY-NC 4.0](asset/README.md). Bundled third-party components retain
their respective licenses; see
[`annotation_tool/THIRD_PARTY_NOTICES.md`](annotation_tool/THIRD_PARTY_NOTICES.md).

Unless required by applicable law or agreed to in writing, software distributed under this license is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE) file for the full license text.
