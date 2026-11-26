# Agent- and Physics-Guided HAT-4D

本仓库是在 HAT-4D 公开代码基础上独立开发的研究原型，用于从单目视频生成多物体 4D Gaussian 场景。它不是 HAT-4D 官方仓库，也不代表原作者维护的版本。

项目保留原项目的版权、引用和许可证要求，在原流程上增加了自动视频理解、交互知识图谱、对象分类、物理规划、自动分割与验证、时序记忆以及失败修复能力。

## 主要修改

### 1. 视频理解与 IKG

新增 Agent0，对视频帧进行分段理解并输出 Interaction Knowledge Graph（IKG）：

- 对象清单、对象类别和属性；
- 阶段、事件、关系与约束；
- 深度顺序和相对位置时间线；
- 分段摘要与 segment memory。

IKG 会继续传给位姿调整、物理规划和 4D 验证模块，而不是只作为日志保存。

### 2. 自动分割与对象生成

新增 Agent1 自动生成 SAM 正负提示点，并结合以下规则筛选 mask：

- 正点覆盖与负点排除；
- VLM 对目标完整性、额外对象和缺失部件的检查；
- 颜色语义约束；
- 多候选 mask 选择和失败重试。

通过 mask 后调用 SAM3D 生成每个对象的初始 3D 表示。

### 3. 相对位姿与碰撞约束

新增 Agent2 候选位姿搜索，综合检查：

- 输入图像素对齐；
- IKG 中的前后、上下和左右关系；
- AABB 碰撞、穿透和边界；
- 多物体深度顺序与视图居中程度。

### 4. 对象分类与物理规划

新增 `PhysicsPlanningAgent`，根据对象外观、IKG 和运动证据估计连续物理属性，包括刚性、弹性、关节性、薄片性、拓扑稳定性和已观察形变，并选择：

- `rigid`
- `elastic`
- `articulated`
- `cloth`
- `free_deform`

分类结果用于生成允许运动、禁止运动和物理约束。它不是独立展示用标签，而会参与后续 Gaussian 运动投影。

### 5. Gaussian 物理约束

新增 `gaussian_physics.py`、`rigid_gaussian.py` 和 SV4D 初始化适配，将物理计划投影到时序 Gaussian：

- 刚体形状与尺度保持；
- 弹性形变幅度限制；
- 速度和加速度平滑；
- 拓扑、接触、残差和相对位置约束。

### 6. 4D 验证、记忆与修复

新增 Agent4 多视角验证器，对输入对齐、交互语义、物理合理性、形变、时序一致性和 3D 清晰度评分。失败时可诊断为当前 3D 初始化问题或时序记忆问题，并触发：

- 重新运行完整 3D 生成；
- 使用 memory frame 重建后续片段；
- 更新容量受限的 memory bank。

### 7. 工程兼容性

- 修复 SAM3D/前端图像方向处理，避免无条件上下翻转；
- 增加公开 SV4D 权重兼容模式；
- 增加 PyTorch/CUDA attention 和显存兼容处理；
- 增加 headless 3D/4D 入口、统一启动脚本和离线 PLY 渲染工具；
- 增加物理规划、mask 选择、颜色定位和 SV4D 兼容测试。

## 流程

```text
视频帧
  -> Agent0 视频理解 / IKG
  -> Agent1 自动提示点 / SAM mask / SAM3D
  -> Agent2 相对位姿与碰撞检查
  -> PhysicsPlanningAgent 分类与物理计划
  -> L4GM / SV4D 时序生成
  -> Gaussian 物理投影
  -> Agent4 多视角验证
  -> 记忆更新或自动修复
```

## 目录

```text
annotation_tool/
  headless_3d_gen.py                 # 无界面 3D 初始化
  headless_4dgen.py                  # 无界面完整 4D 流程
  roi_backend.py                     # ROI、L4GM、SV4D 后端
  viser_frontend.py                  # 交互式前端
  scripts/                           # 服务启动脚本
  model4dgen/                        # 4D 生成及兼容修改
  viewer_plugins/sam3d_gen/          # SAM/SAM3D 与 mask 选择
  viewer_plugins/vlm_assistor/       # Agents、IKG、物理规划、验证器
```

## 环境与模型

本项目依赖多个隔离环境和本地模型服务。默认端口如下：

| 服务 | 默认地址 |
| --- | --- |
| Qwen3-VL OpenAI-compatible API | `http://127.0.0.1:22003/v1` |
| ROI/L4GM/SV4D backend | `http://127.0.0.1:8002/select_roi` |
| SAM3D backend | `http://127.0.0.1:8015` |
| Viser frontend | `http://127.0.0.1:8897` |

模型权重、第三方模型仓库、数据集和运行结果不提交到 Git。请按本机路径配置环境变量，具体安装细节可参考 `annotation_tool/README_ch.md` 和原项目文档历史。

## 启动

以下命令均在 `annotation_tool/` 目录执行。

```bash
# Terminal 1: Qwen3-VL
QWEN_MODEL_PATH=/path/to/Qwen3-VL ./scripts/start_qwenvl.sh

# Terminal 2: ROI/L4GM/SV4D
./scripts/start_roi_backend.sh

# Terminal 3: SAM3D
./scripts/start_sam3d_backend.sh

# Terminal 4: frontend
INPUT_IMAGE_DIR=/path/to/frames \
EVAL_GT_DIR=/path/to/frames \
WORKSPACE_ROOT=/path/to/results/case_name \
./scripts/start_frontend.sh
```

公开 `sv4d2.safetensors` 需要显式启用近似兼容模式：

```bash
HAT4D_SV4D_PUBLIC_COMPAT=1 ./scripts/start_roi_backend.sh
```

该模式用于工程兼容，不能等同于未公开的原始 HAT-4D SV4D 变体。

## Headless 运行

```bash
cd annotation_tool
python headless_4dgen.py \
  --input-image-dir /path/to/frames \
  --workspace-root /path/to/results/case_name \
  --backend-url http://127.0.0.1:8002/select_roi \
  --sam3d-backend-url http://127.0.0.1:8015 \
  --vlm-local-base-url http://127.0.0.1:22003/v1 \
  --vlm-local-model /path/to/Qwen3-VL \
  --start-empty \
  --camera-target-origin
```

已有工作区可用 `--resume-workspace` 恢复；已有 Gaussian 序列可用 `--gs-dir` 指定。

## 输出与检查

每次运行会创建独立 workspace。重点查看：

```text
workspace_xxx/
  gs_sequence/                        # 每帧 Gaussian PLY
  images/input/                       # 本次运行使用的输入帧
  preview/                            # 物理投影及预览中间结果
  vlm_assistor/
    *_headless_4dgen_result.json      # 完整流程结果
    agent4_validator/                 # 验证器多视角渲染
    checkpoints/                      # Agent 状态与恢复点
```

在 headless JSON 中，IKG 和分类/物理信息主要位于：

- `result.init_3d.agent0.scene_graph`
- `result.init_3d.agent0.segment_memories`
- 各对象的 `agent2.object_relation_context`、`final_rule_diagnostics` 和 `final_vlm_eval`
- 物理规划与投影相关的 `physics_plan`、`physical_attributes` 或 `physics_projection` 字段和对应中间 JSON
- 各帧的 `validator.evaluation`、`issue_diagnosis` 和 `memory_bank_update`

`ok: false` 不一定表示没有生成结果。例如所有帧已覆盖但某一帧验证失败时，顶层会因 `validator_incomplete` 返回失败；此时应同时检查 `completed_until_frame`、`fully_covered`、`validator_summary` 和失败帧详情。

## 测试

```bash
cd annotation_tool
python -m pytest tests
```

涉及完整 SAM3D、L4GM、SV4D 或 CUDA 渲染的测试仍要求对应模型、GPU 和服务已经就绪。

## 提交范围

建议提交：

- 所有新增和修改的 Python、Shell、测试与文档；
- 根目录的 `.gitignore`、`README.md` 和原项目 `LICENSE`；
- 为复现所必需且许可证允许提交的小型配置文件。

不要提交：

- `asset/demo_case_3/` 等本地或受限数据；
- checkpoint、模型权重、模型缓存和指向本机 `/data/...` 的软链接；
- `workspace_*`、`results/`、PLY、验证截图和临时日志；
- API key、token、私有配置和 Conda/venv 环境。

提交前建议检查：

```bash
git status --short
git diff --check
git ls-files | rg 'checkpoint|safetensors|workspace_|/data/'
```

## 已知限制

- 小型 VLM 可能漏检细微运动或给出互相矛盾的关系；
- 细杆、齿状结构和严重遮挡对象的 mask 仍可能不稳定；
- 物理约束只能限制已生成运动，不能恢复模型完全没有生成的动作；
- 当前碰撞主要基于 AABB 和 VLM 视觉判断，不是完整物理仿真；
- 多视角清晰度和透明边缘仍受 SAM3D/L4GM 初始几何质量影响。

## 上游与许可证

本项目来源于 HAT-4D：*Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration*。请同时遵守本仓库 `LICENSE`、上游 HAT-4D 许可证，以及 SAM、SAM3D、L4GM、SV4D、Qwen 等第三方组件各自的许可证。

发表或分发衍生结果时，应保留上游归属并引用原 HAT-4D 工作；本仓库新增部分应作为独立衍生实现说明，不应表述为原作者合作或官方版本。
