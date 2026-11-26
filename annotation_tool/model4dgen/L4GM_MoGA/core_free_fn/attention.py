# Modified by HAT-4D; see the subtree MODIFICATIONS.md.
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

XFORMERS_ENABLED = (
    os.environ.get("XFORMERS_DISABLED") is None
    and os.environ.get("HAT4D_DISABLE_XFORMERS", "0") != "1"
)
try:
    if XFORMERS_ENABLED:
        from xformers.ops  import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_dropped = self.attn_drop(attn)
        
        if zero_out_cross_frame and num_frames is not None:
            tokens_per_frame = N // num_frames
            device = attn_dropped.device

            mask = torch.zeros((N, N), device=device)
            for f in range(num_frames):
                start = f * tokens_per_frame
                end = (f + 1) * tokens_per_frame
                mask[start:end, start:end] = 1.0

            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn_dropped = attn_dropped * mask  # zero out cross-frame attention
            print("Zero Out, attn_dropped:", attn_dropped)
        else:
            print("Without Zero Out,attn_dropped:", attn_dropped)
        
        
        
        if return_attn and num_frames is not None:
            # attn shape: [B, num_heads, N, N]

            B, num_heads, N, _ = attn.shape
            T = num_frames
            HW = N // T

            # reshape to [B, num_heads, T, HW, T, HW]
            attn = attn.reshape(B, num_heads, T, HW, T, HW)

            attn_down = attn.mean(dim=3).mean(dim=4)  # -> [B, num_heads, T, T]

            
            attn_return = attn_down.detach()
        else:
            attn_return = attn.detach()

        x = (attn_dropped @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn_return
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False, num_frames:int = None,zero_out_cross_frame:bool=False) -> Tensor:
        if return_attn :
            print("MemEffAttention Return Will Atten")
            pass
        if (not XFORMERS_AVAILABLE) and not return_attn and not zero_out_cross_frame:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv.unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(x))
        if return_attn or zero_out_cross_frame:
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
        attn_bias: Optional[Tensor] = None,
        p: float = 0.0,
    ) -> Tensor:

        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
            "q, k, and v must be 3D [L_total, H, D] tensors"

        L_total, n_heads, d_head = q.shape
        assert k.shape == (L_total, n_heads, d_head)
        assert v.shape[:2] == (L_total, n_heads), \
            "the first two dimensions of v must be [L_total, H]"

        # cu_seqlens: [num_groups + 1]
        assert cu_seqlens.dim() == 1, "cu_seqlens must be a 1D prefix-sum vector"
        num_groups = cu_seqlens.numel() - 1
        assert num_groups > 0, "cu_seqlens must contain at least two elements"

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        if cu_seqlens[-1].item() != L_total:
            raise ValueError(
                f"cu_seqlens[-1] = {cu_seqlens[-1].item()} does not match "
                f"L_total = {L_total}; check the ordering and prefix sums"
            )

        if max_seqlen is not None:
            real_max = max(seqlens)
            if real_max > max_seqlen:
                raise ValueError(
                    f"max_seqlen={max_seqlen} is smaller than the actual maximum {real_max}"
                )

        # q/k/v: [L_total, H, D] -> [1, L_total, H, D]
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)

        # BlockDiagonalMask.from_seqlens accepts per-group sequence lengths.
        block_diag_bias = fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=seqlens,
            kv_seqlen=seqlens,
            device=q.device,
        )


        if attn_bias is not None:
            raise NotImplementedError(
                "Combining an additional attn_bias with BlockDiagonalMask is not supported"
            )

        # The output shape is [B, Mq, H, Kv].
        out_4d = xops.memory_efficient_attention(
            q_4d,            # [1, L_total, H, D]
            k_4d,
            v_4d,
            attn_bias=block_diag_bias,
            p=p,
        )

        out = out_4d.squeeze(0)  # [L_total, H, D]

        return out
