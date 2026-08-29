# HAT-4D 标注工具：中文指南

[English](README_eng.md) | [UI 操作指南](../docs/ui_README_ch.md) | [返回 HAT-4D](../README.md)

该工具用于把单目 RGB 视频帧序列转换为可编辑的 4D Gaussian 序列。浏览器前端协调两个 GPU 后端，并可选接入 QwenVL：

| 进程 | 默认地址 | 功能 |
| --- | --- | --- |
| QwenVL / SGLang | `http://127.0.0.1:22003/v1` | 可选的视频理解与 Human-Agent 决策 |
| ROI/L4GM 后端 | `http://127.0.0.1:8002` | 4D 初始化、传播、ROI 修复与 SV4D 优化 |
| SAM3D 后端 | `http://127.0.0.1:8015` | 点提示分割、3D 物体生成与位姿优化 |
| Viser 前端 | `http://127.0.0.1:8897` | 交互标注、检查与工作区保存 |

纯人工标注不要求启动 QwenVL；完整流程需要两个模型后端和前端。默认 URL 都是 `127.0.0.1`，因此建议所有进程运行在同一台机器上。若跨机器部署，必须把每个 URL 改成可访问地址，并做好端口访问控制。

## 1. 开源包内容

- 前端、编辑插件、VLM Agent、后端适配器和服务测试。
- `model4dgen/L4GM_MoGA` 中的 HAT-4D L4GM/MoGA 运行时代码。
- `model4dgen/generative-models` 中的定制 SV4D2 运行时代码。
- SAM 3D Objects、Segment Anything、Qwen3-VL 和 OpenAI CLIP 的固定版本安装器。

仓库不包含完整的 MVOIK-4D 数据集、模型权重、缓存和生成结果。根目录 [`asset/`](../asset/README.md) 中保留了两个经过检查的 demo 子集，以便直接测试标注流程；它们采用 CC BY-NC 4.0 数据许可，与源代码许可相互独立。L4GM 权重包含模型提供方声明的非商业限制；SAM3D 和 Qwen 权重也有各自许可。下载或分发权重前请自行确认并接受对应条款。

## 2. 基础要求

- Linux、NVIDIA CUDA GPU、Conda/Mamba、Git、Python 3.10 或 3.11。
- 足够空间存放 Qwen3-VL、SAM3D、L4GM 和 SV4D 权重。
- Qwen/SGLang、ROI/L4GM、SAM3D 分别使用独立环境，它们的 PyTorch/CUDA 依赖并不完全兼容。

8 卡机器可参考：QwenVL 使用 `0,1,2,3`，ROI/L4GM 使用 `4`，SAM3D 使用 `5`。请根据显存调整，不要让 ROI 与 SAM3D 在同一张接近满载的 GPU 上同时运行。

## 3. 安装源码依赖

在 HAT-4D 仓库根目录运行：

```bash
python annotation_tool/scripts/install_external_dependencies.py \
  --only sam-3d-objects \
  --only segment-anything \
  --only Qwen3-VL \
  --only openai-clip
```

安装器按照 `external_dependencies.json` 中的固定 commit 拉取源码。如果通过 Python 包安装 SGLang 且 Qwen 权重放在仓库外，可以不下载 Qwen3-VL 源码；ROI/L4GM 环境仍需要 OpenAI CLIP。

### 前端环境

```bash
conda create -n hat4d-frontend python=3.11 -y
conda activate hat4d-frontend
python -m pip install --upgrade pip
python -m pip install -r annotation_tool/requirements.txt
```

公开版本默认使用 `viser==1.0.30`。`annotation_tool/thirdparty/viser` 下的本地
定制版本是可选增强，不随发布包分发。代码会在运行时检测其非标准键盘回调；
使用官方 Viser 时只有定制快捷键不可用，其他标注功能不受影响。

### ROI/L4GM 环境

```bash
conda create -n hat4d-roi python=3.11 -y
conda activate hat4d-roi
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r annotation_tool/requirements.txt
python -m pip install 'setuptools<81'
python -m pip install --no-build-isolation -e annotation_tool/thirdparty/CLIP
python -m pip install -r annotation_tool/requirements_roi.txt
```

锁定的 ROI 环境已在 PyTorch 2.7.0、Torchvision 0.22.0 和 CUDA 12.8
上测试。如果默认 wheel 与机器 CUDA 不匹配，请先安装适合本机驱动的 wheel。

### 后端安全配置

在所有后端和前端终端中设置相同 token。仓库之外的输入、输出目录必须加入
以冒号分隔的允许列表：

```bash
export HAT4D_TOKEN_FILE="${XDG_RUNTIME_DIR:-/tmp}/hat4d-${UID}.token"
test -s "$HAT4D_TOKEN_FILE" || (umask 077 && openssl rand -hex 32 > "$HAT4D_TOKEN_FILE")
export HAT4D_API_TOKEN="$(cat "$HAT4D_TOKEN_FILE")"
export HAT4D_ALLOWED_ROOTS="$PWD:/path/to/input-data:/path/to/results"
```

健康检查不需要 token，其他请求需要。没有 token 时，后端拒绝绑定非回环地址。
Viser 前端没有内置鉴权，会拒绝所有非回环绑定；远程访问必须通过带鉴权的隧道或
Notebook 代理。

### SAM3D 环境

按照 `annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/doc/setup.md` 创建环境并申请 gated 权重，然后安装 Segment Anything：

```bash
conda activate sam3d-objects
python -m pip install -e \
  annotation_tool/viewer_plugins/sam3d_gen/thirdparty/sam-3d-objects/thirdparty/segment-anything
python -m pip install viser==1.0.30 imageio scipy
```

### QwenVL 环境

创建独立环境，安装与本机 CUDA 和 Qwen3-VL 权重兼容的 SGLang：

```bash
conda create -n hat4d-qwen python=3.11 -y
conda activate hat4d-qwen
python -m pip install --upgrade pip
python -m pip install sglang
```

## 4. 放置模型权重

| 模型 | 代码默认读取位置 |
| --- | --- |
| L4GM | `annotation_tool/model4dgen/L4GM_MoGA/pretrained/recon.safetensors`，或设置 `L4GM_CHECKPOINT` |
| HAT-4D SV4D2 encoder 版本 | `annotation_tool/model4dgen/generative-models/checkpoints/sv4d2.0/sv4d2_with_first_stage_encoder_v2.safetensors`，或设置 `SV4D_CHECKPOINT` |
| SAM | `annotation_tool/checkpoints/sam/sam_vit_h_4b8939.pth`，或设置 `SAM_CHECKPOINT` |
| SAM3D Objects | `.../sam-3d-objects/checkpoints/hf/pipeline.yaml`，或设置 `SAM3D_PIPELINE_CONFIG` |
| Qwen3-VL | 任意本地权重目录，通过 `QWEN_MODEL_PATH` 指定 |

这里的 SV4D 权重是 HAT-4D 使用的 encoder 定制版本，不是官方标准
`sv4d2.safetensors`。仅重命名其他 checkpoint 不能兼容。该权重的公开下载页和
模型许可证仍待发布；发布之前，外部用户无法复现 ROI 初始化流程。
正确文件大小为 12,117,981,868 字节，SHA256 为
`fa46c1d2101ad8cc32e6d3cc4513e1c083f4e2729935467e3d42a57a88a6403a`。

在仓库根目录下载 L4GM 和 Segment Anything 官方权重：

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

SAM3D Objects 权重需要先在 Hugging Face 申请访问权限，并仅在本机执行
`hf auth login`：

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

`sv4d2_with_first_stage_encoder_v2.safetensors` 应从 HAT-4D 模型发布页获取。
Stability AI 公开的 `sv4d2.safetensors` 是上游基础模型，不能替代该 encoder
定制权重。认证 token 不得写入仓库；使用前还需检查 L4GM、SAM、SAM3D、
SV4D 和 Qwen 各自的模型许可。

## 5. 准备标注 case

输入帧直接放在 `input` 目录下，推荐使用六位补零文件名：

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

只有在线 PSNR/LPIPS 评测需要 `eval_gt` 内容。没有多视角 GT 时创建空目录即可，同时不要设置 `ONLINE_METRIC=1`。

## 6. 启动服务

建议开启四个终端或 tmux 窗口，以下命令均从仓库根目录执行。

如果机器上保留了原项目配好的环境，ROI 后端可直接使用 `4dgsEditor`，
SAM3D 后端和前端使用 `4dgsEditor_sam3d_v2`。测试无 VLA/VLM 指导的纯人工流程时，
完全跳过终端 A 即可。

### 终端 A：QwenVL（可选）

```bash
conda activate hat4d-qwen
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QWEN_MODEL_PATH=/path/to/Qwen3-VL-235B-A22B-Instruct
export QWEN_TP=4
annotation_tool/scripts/start_qwenvl.sh
```

等待接口可用：

```bash
curl --fail http://127.0.0.1:22003/v1/models
```

如果返回的模型 ID 不是 `Qwen/Qwen3-VL-235B-A22B-Instruct`，启动前端时应把 `VLM_LOCAL_MODEL` 设置成实际返回值。

### 终端 B：ROI/L4GM 后端

```bash
conda activate hat4d-roi
export CUDA_VISIBLE_DEVICES=4
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-roi"
export L4GM_CHECKPOINT=/path/to/recon.safetensors
export SV4D_CHECKPOINT=/path/to/sv4d2_with_first_stage_encoder_v2.safetensors
annotation_tool/scripts/start_roi_backend.sh
```

### 终端 C：SAM3D 后端

```bash
conda activate sam3d-objects
export CUDA_VISIBLE_DEVICES=5
export TORCH_EXTENSIONS_DIR="$HOME/.cache/hat4d-sam3d"
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export SAM3D_PIPELINE_CONFIG=/path/to/sam-3d-objects/checkpoints/hf/pipeline.yaml
annotation_tool/scripts/start_sam3d_backend.sh
```

### 终端 D：前端

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

浏览器打开 `http://127.0.0.1:8897`。需要在线指标时，在启动前额外执行 `export ONLINE_METRIC=1`。

使用远程 OpenAI 兼容接口时，只在当前 shell 中设置服务根地址，并派生 API
地址，不要把真实的内部代理地址写进被 Git 跟踪的文件：

```bash
export BASE_URL="https://your-qwen-endpoint.example/proxy/8000"
export VLM_LOCAL_BASE_URL="${BASE_URL%/}/v1"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
```

## 7. 标注流程

### 使用 SAM3D 初始化物体

1. 切换到能清楚看到全部目标物体的帧。
2. 在 `SAM 3D Gen` 面板点击 `Begin SAM3D Gen`。
3. 勾选 `Pixel Pick Mode`，在目标上添加正点；用 `Negative Point` 排除多余区域，用 `Eraser Point` 删除提示点。
4. 确认 mask 后点击 `Generate 3D (Current Mask)`。
5. 建议保持 `Pose optimize after GEN` 开启。若深度仍有偏差，先调整物体 transform，再在 `Pose Opt` 中配置轴锁定，并点击 `Optimize Pose (Mask Align)`。
6. 点击 `New Object`，对每个独立运动物体重复以上步骤。
7. 所有物体都确认后，点击 `Add All Objects & Exit` 写入当前帧并退出。

### 进行 4D 传播

1. 在 `L4GM_4DGEN` 面板点击 `Generate next segment`，一个 segment 通常生成后续 16 帧。
2. 若希望每段完成后人工检查，可在按钮变成 `Pause Generate` 后再次点击暂停。
3. 使用顶部播放/暂停控制检查轮廓、几何形状、相对深度、接触关系和运动连续性。
4. 修复中间帧时，先切到问题帧，在点击 `Begin SAM3D Gen` 之前务必勾选 `SAM 3D Gen > Total cover`，然后重建该帧全部可见物体并继续传播。遗漏 `Total cover` 会把新物体与旧 PLY 混合。

### Human-Agent 模式

在 `VLM Assistor` 中先确认 Qwen 地址和模型名，运行 Agent 0 视频理解并检查 Temporal Scene Graph，再运行物体生成/验证 pipeline。Agent 输出只应作为建议；接受结果前仍需人工检查 mask、位姿、物体接触和失败帧。

## 8. 保存与续标

每次运行都会在 `WORKSPACE_ROOT` 下生成带时间戳的工作区。继续某次结果时：

```bash
export RESUME_WORKSPACE=/path/to/results/example_case/workspace_YYYYMMDD_HHMMSS
annotation_tool/scripts/start_frontend.sh
```

`RESUME_WORKSPACE` 必须指向具体的时间戳目录，而不是其父目录。正确工作区通常包含 `meta.json`、`gs_sequence/`、`images/`、`selections/` 和 `results/`。

## 9. 验证与排障

不加载模型的发布包检查：

```bash
python -m compileall -q \
  -x '(^|/)(thirdparty|tmp|checkpoints|\.cache)(/|$)' \
  annotation_tool
pytest -q annotation_tool/tests
python annotation_tool/scripts/check_publication.py annotation_tool
```

发布前还要人工检查 `git status --short`。不要提交输入视频、评测图片、工作区、
模型权重、Hugging Face 缓存、token、私有服务 URL 或机器绝对路径。自动扫描
无法识别图片中的人脸、屏幕、工牌、地点等隐私信息，因此媒体文件必须人工审核。

只启动健康检查接口：

```bash
python annotation_tool/roi_backend.py --health-only --port 8002
python annotation_tool/viewer_plugins/sam3d_gen/backend.py --health-only --port 8015
```

常见问题：

- `L4GM checkpoint not found`：检查 `pretrained/recon.safetensors` 的精确路径。
- `SV4D2 checkpoint not found`：确认使用上表中的 encoder 版本文件名。
- SAM3D 的 `sam_ready: false`：检查 `SAM_CHECKPOINT`、模型类型和 CUDA 设备。
- SAM3D 的 `sam3d_ready: false`：检查 gated 权重和 `SAM3D_PIPELINE_CONFIG`。
- `/v1/models` 正常但 Qwen 请求失败：把 `VLM_LOCAL_MODEL` 改为接口返回的准确 ID。
- 浏览器能打开但后端操作失败：确认端口一致，且进程运行在同一主机。
- 续标后场景为空：确认 `RESUME_WORKSPACE` 指向具体时间戳工作区。
