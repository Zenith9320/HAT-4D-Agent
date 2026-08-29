# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

from torch import Tensor
import torch
from torch import nn
from typing import Tuple, Literal, Dict, Optional
import torch.nn.functional as F
from xformers.ops  import  memory_efficient_attention, unbind
from xformers.ops import fmha
import xformers.ops as xops

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops  import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn: bool = False, num_frames:int = None,zero_out_cross_frame:bool=False) -> Tensor | tuple[Tensor, Tensor]:
        B, N, C = x.shape
        # print("Embedding.shape:",  x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_dropped = self.attn_drop(attn)
        
        # === 实现 Zero-out Cross-frame ===
        if zero_out_cross_frame and num_frames is not None:
            tokens_per_frame = N // num_frames  # 假设每帧有相同数量的 token
            device = attn_dropped.device

            # 构造 attention mask，shape: [N, N]
            mask = torch.zeros((N, N), device=device)
            for f in range(num_frames):
                start = f * tokens_per_frame
                end = (f + 1) * tokens_per_frame
                mask[start:end, start:end] = 1.0  # 保留 intra-frame 的 attention

            # 扩展到 shape [1, 1, N, N] 以便广播到 attn_dropped
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn_dropped = attn_dropped * mask  # zero out cross-frame attention
            print("Zero Out, attn_dropped:", attn_dropped)
        # === 结束 Zero-out 操作 ===
        else:
            print("Without Zero Out,attn_dropped:", attn_dropped)
        
        
        
        if return_attn and num_frames is not None:
            # 使用 interpolate 降维注意力矩阵
            # attn shape: [B, num_heads, N, N]
            # print("[SHAPE] attn:", attn.shape)
            # print("num frames:", num_frames)

            B, num_heads, N, _ = attn.shape
            T = num_frames  # 你知道每个序列有几个帧
            HW = N // T     # 空间位置数

            # reshape to [B, num_heads, T, HW, T, HW]
            attn = attn.reshape(B, num_heads, T, HW, T, HW)

            # 对 HW 维度平均：先对 key 的空间维度求平均，再对 query 的空间维度求平均
            attn_down = attn.mean(dim=3).mean(dim=4)  # -> [B, num_heads, T, T]

            # print("[SHAPE] after attn:", attn_down.shape)
            
            attn_return = attn_down.detach()
            # attn_return = torch.zeros(1,1,1,1)
        else:
            attn_return = attn.detach()

        x = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn_return # 返回未dropout的注意力权重
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False, num_frames:int = None,zero_out_cross_frame:bool=False) -> Tensor:
        if return_attn :
            print("MemEffAttention Return Will Atten")
            pass
        if (not XFORMERS_AVAILABLE) or return_attn or zero_out_cross_frame:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, return_attn=return_attn, num_frames=num_frames, zero_out_cross_frame=zero_out_cross_frame)
    
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# For MoGA
    @staticmethod
    def forward_varlen(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        attn_bias: Optional[Tensor] = None,  # 目前不额外支持外部 bias，仅做占位
        p: float = 0.0,
    ) -> Tensor:
        """
        利用 xFormers 的 memory_efficient_attention + BlockDiagonalMask
        实现高效的变长组内自注意力（Intra-Group Attention）。

        Args:
            q, k, v (Tensor):
                排列后的 QKV 张量，形状 [L_total, H, D]，其中
                L_total = 所有组 token 数之和。
                注意：xFormers 推荐使用 float16 或 bfloat16 以获得最高性能。

            cu_seqlens (Tensor):
                累积序列长度（组大小 prefix-sum），形状 [num_groups + 1]。
                例如：tensor([0, L1, L1+L2, L1+L2+L3, ...])

            max_seqlen (int):
                组的最大长度（一般是 seqlens.max().item()），
                这里对 xFormers 来说只是一个辅助信息（你也可以不用它）。

            attn_bias (Optional[Tensor]):
                额外的注意力偏置（目前实现中暂不支持合并，
                如需支持可以在 BlockDiagonalMask 上再叠加 bias，需自定义）。

            p (float):
                dropout 概率，训练时可设为 0~0.1，推理时应为 0.0。
        """

        # ---------- 1. 基本检查 ----------
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
            "q, k, v 应为 [L_total, H, D] 形状的 3D 张量"

        L_total, n_heads, d_head = q.shape
        assert k.shape == (L_total, n_heads, d_head)
        assert v.shape[:2] == (L_total, n_heads), \
            "v 的前两维应为 [L_total, H]"
        # v 允许 value_dim != d_head，如果不同则 d_v = v.shape[-1]

        # cu_seqlens: [num_groups + 1]
        assert cu_seqlens.dim() == 1, "cu_seqlens 应为 1D prefix-sum 向量"
        num_groups = cu_seqlens.numel() - 1
        assert num_groups > 0, "cu_seqlens 长度必须 >= 2"

        # ---------- 2. 从 cu_seqlens 计算每个组长度 ----------
        # seqlens[i] = 第 i 个组的长度
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # 检查总长度是否匹配
        if cu_seqlens[-1].item() != L_total:
            raise ValueError(
                f"cu_seqlens[-1] = {cu_seqlens[-1].item()} 与 "
                f"L_total = {L_total} 不一致，说明排列或前缀和有问题"
            )

        # 可选：验证 max_seqlen
        if max_seqlen is not None:
            real_max = max(seqlens)
            if real_max > max_seqlen:
                raise ValueError(
                    f"给定的 max_seqlen={max_seqlen} 小于真实最大长度 {real_max}"
                )

        # ---------- 3. reshape 为 xFormers 期望的 [B, M, H, K] 结构 ----------
        # 这里把所有组 flatten 成一个 batch 里的大序列，B=1
        # q/k/v: [L_total, H, D] -> [1, L_total, H, D]
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)

        # ---------- 4. 利用 BlockDiagonalMask 构造“组内块对角” mask ----------
        # BlockDiagonalMask.from_seqlens 会根据 seqlens 构造块对角结构，
        # 这样第 i 个组的 token 只会跟第 i 组的 token 互相注意。
        # BlockDiagonalMask.from_seqlens accepts per-group sequence lengths.
        block_diag_bias = fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=seqlens,
            kv_seqlen=seqlens,          # 自注意力：Q/K/V 的分组一致
            device=q.device,
        )

        # 如果需要做 causal 组内注意力（如 decoder），可以在这里：
        # block_diag_bias = block_diag_bias.make_causal()

        if attn_bias is not None:
            # 如果你未来希望在组内 mask 之外再叠加一个外部 bias，
            # 这里可以用 block_diag_bias.materialize(...) 手动加到 attn_bias 上，
            # 再传入 memory_efficient_attention（注意这会略慢）。
            raise NotImplementedError(
                "当前实现未支持额外 attn_bias 与 BlockDiagonalMask 的组合，"
                "如有需要请在此处手工实现相加逻辑。"
            )

        # ---------- 5. 调用 xFormers memory-efficient attention ----------
        # xops.memory_efficient_attention 接受 [B, Mq, H, K] 形状输入，支持 AttentionBias。
        # The output shape is [B, Mq, H, Kv].
        out_4d = xops.memory_efficient_attention(
            q_4d,            # [1, L_total, H, D]
            k_4d,
            v_4d,
            attn_bias=block_diag_bias,
            p=p,             # dropout 概率
            # scale=None -> 自动用 1/sqrt(D)
        )

        # ---------- 6. 还原回 [L_total, H, D] ----------
        out = out_4d.squeeze(0)  # [L_total, H, D]

        return out
