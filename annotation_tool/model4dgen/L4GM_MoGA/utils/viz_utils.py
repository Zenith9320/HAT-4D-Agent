"""
可视化的辅助函数
"""


import os
import time
import json
import random
import contextlib
import psutil
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import amp
from safetensors.torch import load_file
from accelerate import Accelerator
from PIL import Image
import imageio
import tyro
import kiui
from core.options_conf import AllConfigs
import copy
import shutil
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import matplotlib.colors as mcolors

def save_images_clip_as_video(images_clip, CN, V, save_path="images_clip.mp4", fps=5):
    import torch, numpy as np, cv2, os
    from PIL import Image

    B, _, C, H, W = images_clip.shape
    images_clip = images_clip[0].detach().cpu()

    images_clip = images_clip.reshape(CN, V, C, H, W)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    images_clip = images_clip.permute(0,3,1,4,2).reshape(CN,H, W*V,C)
    imageio.mimwrite(save_path, images_clip, fps=2)


    print(f"✅ 保存完成: {save_path} ({CN} 帧, {V} 视角, 尺寸 {H}×{W})")
    
    
# ====== Group Viz ========
def logits_to_groups_with_capacity(
    router_logits: torch.Tensor, 
    capacity_factor: float = 1.0, 
    T: float = 1.0
) -> torch.Tensor:
    """
    根据 MoGA 路由 Logits 计算最终的分组 ID (final_group)。
    此函数实现了 Top-1 路由选择和基于容量因子的硬容量负载均衡。

    Args:
        router_logits (torch.Tensor): 路由 Logits，形状为 [B', L, M]，M 是组数。
        capacity_factor (float): 容量因子 (α)，用于计算每组最大容量。
        T (float): Softmax 温度参数。

    Returns:
        torch.Tensor: 最终的组 ID 分配结果 (final_group)，形状为 [N]，
                      其中 N 是总 token 数 (B' * L)。
    """
    
    # 获取形状参数
    B_prime, L, M = router_logits.shape
    N = B_prime * L # 总 token 数
    
    # -----------------------------
    # 1. softmax 得到路由权重和 Top-1 组
    # -----------------------------
    weights = (router_logits / T).softmax(dim=-1)        # [B', L, M]
    # Top-1 选择
    _, groups = torch.topk(weights, k=1, dim=-1)
    
    # 展平
    groups_flat  = groups.reshape(-1)                    # [N]

    # -----------------------------
    # 2. Hard Capacity
    # -----------------------------
    # 容量 = α * (平均每组 token 数)
    # 容量必须大于等于 1
    capacity = max(1, int(capacity_factor * (N / M)))

    # 每组计数器，用于追踪当前分配的 token 数量
    counts = torch.zeros(M, dtype=torch.int32, device=groups_flat.device)

    # 最终 group 分配结果
    final_group = torch.empty_like(groups_flat)

    # 顺序遍历 token，执行容量控制
    # 注意：此处的 for 循环是 PyTorch 性能瓶颈，但在没有自定义 CUDA/C++ Kernel 的情况下，
    # 它是忠实模拟 MoGA Hard Capacity 逻辑的方式。
    for idx in range(N):
        g = groups_flat[idx].item()
        
        if counts[g] < capacity:
            # 该组有空位，分配 Top-1 推荐的组
            final_group[idx] = g
            counts[g] += 1
        else:
            # 超出容量，必须找一个最空闲的组
            # argmin 找到第一个计数最小的组 ID
            alt = torch.argmin(counts).item() 
            final_group[idx] = alt
            counts[alt] += 1
            
    # final_group 形状为 [N]，即 [B' * L]
    return final_group


# 辅助函数 (可添加到 MoGAttention 类中，或作为独立函数)
# 作用是把 [B, L] 的分组 ID 恢复到 [BTV, H, W] 的空间结构
def get_group_map_from_flat(num_clip_frames,num_views, final_group, B, H, W):
    """从展平的 final_group 恢复到 BTV, H, W 的空间结构"""
    # final_group 形状: [B * L]
    
    # [B, L] -> [B, T, V, H, W]
    group_map_reshaped = final_group.reshape(B,num_clip_frames,num_views, H, W)
    
    # [B, T, V, H, W] -> [BTV, H, W]
    BTV = B * num_clip_frames * num_views
    print("group_map:", group_map_reshaped.shape)
    print("BTV, H, W:", (BTV, H, W))
    group_map = group_map_reshaped.reshape(BTV, H, W) 
    return group_map


def visualize_moga_groups_multiview(
    group_map: torch.Tensor,
    original_input_shape: Tuple[int, int, int, int], # [BTV, C, H_orig, W_orig]
    num_frames: int,                                  # T: 时间帧数
    num_views: int,                                   # V: 视角数
    sample_index: int = 0,
    save_path: Optional[str] = None
):
    """
    可视化 MoGA 分组 ID 映射图，按时间和视角 (T x V) 平铺展示。

    Args:
        group_map (torch.Tensor): 形状为 [BTV, H_sub, W_sub] 的分组 ID 图。
        original_input_shape (Tuple[int, int, int, int]): 原始输入的形状 (BTV, C, H_orig, W_orig)。
        num_frames (int): T，视频帧数或时间步长。
        num_views (int): V，相机视角数。
        sample_index (int): 要可视化的批次中的样本索引（对应原始 Batch B）。
        save_path (Optional[str]): 图像保存路径。
    """
    if group_map is None:
        print("Error: group_map is None.")
        return

    # 1. 数据准备与形状检查
    BTV, C_orig, H_orig, W_orig = original_input_shape
    
    # 计算原始批次大小 B
    B_orig = BTV // (num_frames * num_views)
    if sample_index >= B_orig:
        raise IndexError(f"Sample index {sample_index} out of bounds for original batch size {B_orig}")

    # 计算目标样本在 BTV 维度中的起始索引和结束索引
    start_idx = sample_index * (num_frames * num_views)
    end_idx = start_idx + (num_frames * num_views)
    
    # 提取并重塑目标样本的 Group Map: [T*V, H_sub, W_sub] -> [T, V, H_sub, W_sub]
    target_groups_flat = group_map[start_idx:end_idx]
    H_sub, W_sub = target_groups_flat.shape[1:]
    
    # 恢复 T 和 V 维度
    groups_tv = target_groups_flat.reshape(num_frames, num_views, H_sub, W_sub).cpu().numpy()
    
    # 2. 设置颜色映射
    num_groups = int(group_map.max().item()) + 1
    cmap = plt.cm.get_cmap('Spectral', num_groups) 
    
    # 3. 绘制 T x V 子图
    fig, axes = plt.subplots(
        nrows=num_frames, 
        ncols=num_views, 
        figsize=(4 * num_views, 4 * num_frames), 
        squeeze=False # 确保即使 T=1 或 V=1 也是二维数组
    )

    sample_rate_h = H_orig / H_sub
    sample_rate_w = W_orig / W_sub
    
    # 遍历 T 和 V 维度，绘制每个子图
    for t in range(num_frames):
        for v in range(num_views):
            ax = axes[t, v]
            group_ids = groups_tv[t, v]
            
            # 使用 imshow 绘制分组ID图
            im = ax.imshow(group_ids, cmap=cmap, interpolation='nearest')
            
            ax.set_title(f"Frame {t}, View {v}", fontsize=10)
            ax.axis('off')

    # 4. 添加总标题和全局颜色条
    fig.suptitle(
        f"MoGA Token Grouping Map (Sample {sample_index}) | Feature Size: {H_sub}x{W_sub}, Groups: {num_groups}", 
        fontsize=14
    )
    
    # 添加一个全局颜色条
    bounds = np.linspace(-0.5, num_groups - 0.5, num_groups + 1)
    ticks = np.arange(num_groups)
    
    # 创建一个 ScalarMappable 来统一颜色条
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(group_map.cpu().numpy().flatten()) # 绑定所有数据，确保颜色条完整
    
    # 将颜色条放置在右侧
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels([str(int(t)) for t in ticks], fontsize=8)
    cbar.set_label('Token Group ID (M)', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # 调整布局以容纳 suptitle 和颜色条

    # 5. 显示或保存
    if save_path:
        plt.savefig(save_path)
        print(f"Group map saved to {save_path}")
    else:
        plt.show()

#
