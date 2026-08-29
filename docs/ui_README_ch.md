# HAT-4D 标注工具 UI 使用指南

[English](ui_README.md) | [返回项目 README](../README.md) | [环境安装与启动](../annotation_tool/README_ch.md)

本文介绍完成服务启动后，如何使用 Viser 前端初始化 3D 物体、传播 4D
Gaussian 序列，以及修复中间失败帧。默认前端地址为
`http://127.0.0.1:8897`。

## SAM3D 初始化

### 适用情况

- 在第一帧初始化场景中的全部目标物体。
- 后续帧出现明显形状变形、位置漂移、深度错误或其他未知大偏差时，重建该帧。
- 进入 SAM3D 前，应切换到能在 `input` 视角清楚看到全部待重建物体的帧。
  被遮挡或完全不在画面中的物体无法通过当前帧的 SAM mask 正确生成。中间帧使用
  `Total cover` 时尤其需要保证所有物体可见，因为该模式会替换当前帧的全部 3D。

### 进入生成模式

1. 展开 `SAM 3D Gen` 面板。
2. 点击 `Begin SAM3D Gen`。
3. 勾选 `Pixel Pick Mode`，进入 SAM 点提示分割模式。

> **中间帧整体修复：** 必须在点击 `Begin SAM3D Gen` **之前**勾选
> `SAM 3D Gen > Total cover (clear session PLY on begin)`。该操作会清空当前帧
> 已有的 session PLY，因此随后必须重新生成该帧的全部可见物体。第一帧初始化时
> 或仅希望保留当前帧已有物体时，不要误用该选项。

### 分割并生成第一个物体

1. 保持 `Negative Point` 和 `Eraser Point` 未勾选，在图像中的目标物体上点击，
   添加正向点。
2. 需要排除错误区域时勾选 `Negative Point`，再点击不应属于目标的区域。
3. 需要删除提示点时勾选 `Eraser Point`，再点击对应提示点。也可以使用
   `Undo Point` 撤销上一步，或用 `Clear Points` 清空当前物体的全部提示点。
4. 检查 mask 已完整覆盖目标且没有包含其他物体后，点击
   `Generate 3D (Current Mask)`。

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-point-prompts.png" width="92%" alt="SAM3D point prompts and current mask">
</p>

生成和默认位姿优化通常需要约 10 秒，具体时间取决于 GPU、模型缓存和输入内容。
建议保持 `Pose optimize after GEN` 勾选，并等待界面出现生成完成和位姿优化完成提示
后再继续操作。

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-generated-object.png" width="92%" alt="Generated SAM3D object after pose optimization">
</p>

### 修正位姿和深度

生成完成后，回到 `input` 视角检查轮廓、尺度、深度和物体间接触关系。

1. 如果物体在世界坐标 X 方向明显偏离 `x=0`，先拖拽物体坐标系做粗略对齐。
2. 展开 `SAM 3D Gen > Pose Opt`。
3. 勾选 `Pose opt hard lock world X`，锁定刚刚人工调整的 X 位置。
4. 点击 `Optimize Pose (Mask Align)`，等待 mask 对齐优化完成。
5. 再次在 `input` 视角检查物体轮廓是否贴合、深度是否合理。若仍不正确，继续
   调整 transform 或其他轴锁定选项后重新优化。

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-pose-opt-and-propagate.png" width="92%" alt="SAM3D pose optimization and L4GM propagation controls">
</p>

### 生成其他物体并保存

1. 点击 `New Object` 创建下一个物体槽位。
2. 对每个独立运动物体重复“点提示分割、生成 3D、检查并优化位姿”的流程。
3. 确认全部物体都已生成且相对位置正确后，点击 `Add All Objects & Exit`。

`Add All Objects & Exit` 会把当前 SAM3D session 中的物体写入当前帧并退出分割
生成模式。不要在仍有物体未生成时提前点击。

## 4D Propagation

### 生成下一段

1. 完成当前帧的全部物体并退出 SAM3D 模式。
2. 展开 `L4GM_4DGEN`。
3. 点击 `Generate next segment`，从当前帧向后生成下一段 Gaussian 序列。

默认一段包含后续 16 帧。参考机器上通常约 10 秒完成一段，但实际耗时取决于 GPU
和缓存状态。生成后使用页面顶部的 `Play / Pause` 检查轮廓、几何、相对深度、
物体接触和运动连续性。

建议每生成 16 帧就暂停检查：点击 `Generate next segment` 开始生成；按钮文字变为
`Pause Generate` 后再次点击，使任务在当前 segment 完成后暂停。确认这一段没有问题
后，再继续生成下一段。

### 修复中间失败帧

当中间帧出现严重变形、漂移或深度错误时：

1. 暂停 4D 传播并切换到问题帧。
2. 确认 `input` 视角能看到该帧需要重建的全部物体。
3. 在点击 `Begin SAM3D Gen` **之前**勾选 `Total cover`。
4. 点击 `Begin SAM3D Gen`，对该帧的全部可见物体重新执行 SAM3D 初始化。
5. 点击 `Add All Objects & Exit` 保存完整替换结果。
6. 检查修复帧正确后，从该帧继续执行 `Generate next segment`。

<p align="center">
  <img src="annotation_tool/docs/assets/ui/sam3d-total-cover.png" width="92%" alt="SAM3D total cover workflow for replacing a failed frame">
</p>

> **不要遗漏 `Total cover`：** 如果修复中间帧时未勾选该选项，新生成物体可能与
> 当前帧旧 PLY 混合，产生重复物体、残留几何或错误遮挡。

## 每段检查清单

- 所有物体轮廓都与 `input` 图像一致，没有明显缺失或多余区域。
- 物体尺度、深度顺序和接触关系合理。
- 播放时没有突然拉伸、塌缩、旋转跳变或位置漂移。
- 修复帧没有残留旧 PLY，也没有遗漏需要重建的物体。
- 确认当前段正确后再继续传播，避免错误累积到后续 segment。
