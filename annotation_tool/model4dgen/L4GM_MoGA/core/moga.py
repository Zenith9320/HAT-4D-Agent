# Modified by HAT-4D; see the subtree MODIFICATIONS.md.
import torch
from torch import nn
from typing import Optional, Tuple
from core.attention import MemEffAttention
from core.unet import MVAttention, TempAttention, ResnetBlock
import numpy as np
from typing import Tuple, Literal
from functools import partial
import os
from xformers.ops import memory_efficient_attention, unbind
import torch.nn.functional as F
import torch
from typing import List

class Router(nn.Module):
    def __init__(self, dim: int, num_groups: int, router_bias: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.router_layer = nn.Linear(dim, num_groups, bias=router_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router_layer(x)
        return router_logits
    

class MoGAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_groups: int = 32,  # M
        T: float = 1.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        router_bias: bool = True,
        num_views: int = 4,
        num_frames: int = 8,
        tokenize=True, token_dim=None,
        capacity_factor=1.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.T = T
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.capacity_factor = capacity_factor
    
        self.tokenize = tokenize

        if tokenize:
            token_dim = token_dim or (dim * 4)
            self.tokenizer = nn.Conv2d(dim, token_dim, kernel_size=4, stride=4)
            self.unpatch = nn.ConvTranspose2d(token_dim, dim, kernel_size=4, stride=4)

            dim = token_dim
            
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_groups
        self.T = T
        self.num_frames = num_frames
        self.num_views = num_views
        
        self.router = Router(dim, num_groups, router_bias)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_op = MemEffAttention.forward_varlen

    def _moga_permute(self, q, k, v, router_logits, T=1.0):

        B_prime, L, H, Dk = q.shape
        M = self.num_groups
        
        weights = (router_logits / T).softmax(dim=-1)        # [B', L, M]
        weights_top1, groups = torch.topk(weights, k=1, dim=-1)
        
        weights_flat = weights_top1.reshape(-1)              # [N]
        groups_flat  = groups.reshape(-1)                    # [N]
        N = groups_flat.numel()

        # 2. Hard Capacity
        capacity_factor = getattr(self, "capacity_factor", 1.0)
        capacity = int(capacity_factor * (N / M))

        counts = torch.zeros(M, dtype=torch.int32, device=groups_flat.device)

        final_group = torch.empty_like(groups_flat)

        for idx in range(N):
            g = groups_flat[idx].item()
            
            if counts[g] < capacity:
                final_group[idx] = g
                counts[g] += 1
            else:
                alt = torch.argmin(counts).item()
                final_group[idx] = alt
                counts[alt] += 1

        sort_idx = torch.argsort(final_group, stable=True)   # [N]
        
        groups_sorted = final_group[sort_idx]                # [N]
        weights_sorted = weights_flat[sort_idx]              # [N]

        _, counts = torch.unique_consecutive(groups_sorted, return_counts=True)
        cu_seqlens = torch.cat(
            (counts.new_zeros(1, dtype=torch.int32),
            counts.cumsum(0, dtype=torch.int32))
        )   # [M+1]

        q_permuted = q.reshape(B_prime * L, H, Dk)[sort_idx]
        k_permuted = k.reshape(B_prime * L, H, Dk)[sort_idx]
        v_permuted = v.reshape(B_prime * L, H, Dk)[sort_idx]

        return q_permuted, k_permuted, v_permuted, sort_idx, weights_sorted, cu_seqlens


    def _moga_reperumte(self, out_permuted, sort_idx, weights_sorted):
        
        out_weighted = out_permuted * weights_sorted[:, None, None]
        
        out_ = torch.zeros_like(out_weighted)
        
        out_.index_add_(0, sort_idx.to(out_.device), out_weighted)
        
        return out_

    def forward(self, x: torch.Tensor, 
                    return_group_logit: bool = False, 
                    return_group_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        # -------- ensure submodules have same dtype/device as x -----------
    
        BTV, C, H, W = x.shape
            
        # ------------------- (A) Tokenizer --------------------
        if self.tokenize:
            x = self.tokenizer(x)        # [BTV, C', H/8, W/8]
            _, C, H, W = x.shape
        
        
        
        B = BTV // (self.num_frames * self.num_views)
        
        # [BTV, C, H, W] -> [B, T, V, C, H, W]
        x_reshaped = x.reshape(B, self.num_frames, self.num_views, C, H, W)
        
        # [B, T, V, C, H, W] -> [B, T, V, H, W, C] -> [B, L_full, C]
        x_flat = x_reshaped.permute(0, 1, 2, 4, 5, 3).contiguous().reshape(B, -1, C)
        
        B_prime, L, D = x_flat.shape # B' = B, L = T*V*H*W, D = C

        
        
    
        
        qkv = self.qkv(x_flat).reshape(B_prime, L, 3, self.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2) 

        router_logits = self.router(x_flat) # [B, L, M]
        
        gb_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if return_group_loss:
            gb_loss = _calculate_moga_loss_internal(router_logits, self.T)
            
        q_p, k_p, v_p, sort_idx, weights_sorted, cu_seqlens = \
            self._moga_permute(q, k, v, router_logits, self.T)
        
        max_seqlen = cu_seqlens.max().item()
        max_seqlen = cu_seqlens.max().item()


        out_permuted = self.attn_op(
            q_p, k_p, v_p, 
            cu_seqlens=cu_seqlens, 
            max_seqlen=max_seqlen
        )
        
        out_original_order = self._moga_reperumte(out_permuted, sort_idx, weights_sorted) 
        
        out_original_order = out_original_order.reshape(B_prime, L, D) 
        out_flat = self.proj(out_original_order)
        out_flat = self.proj_drop(out_flat)
        
        
        # [B, L_full, C] -> [B, T, V, H, W, C]
        out_reshaped = out_flat.reshape(B, self.num_frames, self.num_views, H, W, C)
        
        # [B, T, V, H, W, C] -> [B, T, V, C, H, W] -> [BTV, C, H, W]
        out = out_reshaped.permute(0, 1, 2, 5, 3, 4).contiguous().reshape(BTV, C, H, W)
        
        # ------------------ C. Unpatchify ------------------
        if self.tokenize:
            out = self.unpatch(out)     # [BTV, orig_C, orig_H, orig_W]
            
        out_logits = router_logits if return_group_logit else None
        
        
        return out, router_logits, gb_loss
    

    #     # qkv, proj, router

    #     # tokenizer & unpatch
            
    
class MoGADownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True,
        moga_atten : bool = False,
        moga_group: int = 32,
        moga_lambda: float = 0.5
    ):
        super().__init__()
 
        nets = []
        attns = []
        t_attns = []
        moga_attns = []
        
        for i in range(num_layers):
            in_channels_i = in_channels if i == 0 else out_channels
            
            nets.append(ResnetBlock(in_channels_i, out_channels, skip_scale=skip_scale))
            
            # 2. MV/Temp Attention
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
                
            # 3. MoGAttention
            if moga_atten:
                moga_attns.append(MoGAttention(out_channels, attention_heads, moga_group, num_views=num_views, num_frames=num_frames))
            else:
                moga_attns.append(None)
            
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        self.moga_attns = nn.ModuleList(moga_attns)
        self.moga_lambda = moga_lambda

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, return_group_logit: bool = False, return_group_loss: bool = False):
        xs = []
        router_logits = [] 
        accumulated_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for attn, t_attn, net, moga_attn in zip(self.attns, self.t_attns, self.nets, self.moga_attns):
            x = net(x) 

            if moga_attn:
                moga_x, moga_logit, moga_l = moga_attn(x, return_group_logit, return_group_loss)
                if return_group_logit: router_logits.append(moga_logit)
                if return_group_loss: accumulated_loss += moga_l
            
            if attn: x = attn(x)
            if t_attn: x = t_attn(x)
                
            if moga_attn:
                x = x + moga_x * self.moga_lambda
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)

        return x, xs, router_logits, accumulated_loss

class MoGAMidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True,
        moga_atten : bool = False,
        moga_group: int = 32,
        moga_lambda: float = 0.5
    ):
        super().__init__()


        nets = []
        attns = []
        t_attns = []
        moga_attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(in_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
            
            if moga_atten:
                moga_attns.append(MoGAttention(in_channels, attention_heads, moga_group, num_views=num_views, num_frames=num_frames)   )
            else:
                moga_attns.append(None)
            
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        self.moga_attns = nn.ModuleList(moga_attns)
        self.moga_lambda = moga_lambda
        
    def forward(self, x, return_group_logit: bool = False, return_group_loss: bool = False):
        x = self.nets[0](x)
        router_logits = []
        accumulated_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for attn, t_attn, net, moga_attn in zip(self.attns, self.t_attns, self.nets[1:], self.moga_attns):
            if moga_attn:
                moga_x, moga_logit, moga_l = moga_attn(x, return_group_logit, return_group_loss)
                if return_group_logit: router_logits.append(moga_logit)
                if return_group_loss: accumulated_loss += moga_l
            if attn: x = attn(x)
            if t_attn: x = t_attn(x)
            if moga_attn:
                x = x + moga_x * self.moga_lambda
            x = net(x)
            
        return x, router_logits, accumulated_loss
    
class MoGAUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True,
        moga_atten : bool = False,
        moga_group: int = 32,
        moga_lambda: float = 0.5
    ):
        super().__init__()

        nets = []
        attns = []
        t_attns = []
        moga_attns = []
        
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels
            
            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            
            # 2. MV/Temp Attention
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
                
            # 3. MoGAttention
            if moga_atten:
                moga_attns.append(MoGAttention(out_channels, attention_heads, moga_group, num_views=num_views, num_frames=num_frames))
            else:
                moga_attns.append(None)
                
                
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        self.moga_attns = nn.ModuleList(moga_attns)
        self.moga_lambda = moga_lambda

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


        
    def forward(self, x, xs, return_group_logit: bool = False, return_group_loss: bool = False):
        assert len(xs) == len(self.nets), \
            f"UpBlock received {len(xs)} skip connections for {len(self.nets)} layers"
            
        router_logits = []
        accumulated_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for attn, t_attn, net, moga_attn in zip(self.attns, self.t_attns, self.nets, self.moga_attns):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            
            if moga_attn:
                moga_x, moga_logit, moga_l = moga_attn(x, return_group_logit, return_group_loss)
                if return_group_logit: router_logits.append(moga_logit)
                if return_group_loss: accumulated_loss += moga_l
                
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x)
                
            if moga_attn:
                x = x + self.moga_lambda * moga_x
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x, router_logits, accumulated_loss
# it could be asymmetric!
class MoGA_UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256),
        up_attention: Tuple[bool, ...] = (True, True, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn: bool = True,
        use_moga_attn: bool = True,
        moga_lambda: float = 0.5,
        moga_group: int = 32,
        down_moga_attention: Tuple[bool, ...] = (True, True, True, True, True),
        up_moga_attention: Tuple[bool, ...] = (True, True, True, True, True),
        mid_moga_attention: bool = True,
    ):
        super().__init__()
        
        print("[UNet] Using MoGA Configure:")
        print(f"    use_temp_attn: {use_temp_attn}, use_moga_attn: {use_moga_attn}")
        print(f"    moga_lambda: {moga_lambda}, moga_group: {moga_group}")

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(MoGADownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn,
                moga_atten = down_moga_attention[i] and use_moga_attn,
                moga_group = moga_group,
                moga_lambda = moga_lambda
                
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MoGAMidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames, use_temp_attn=use_temp_attn, moga_atten = mid_moga_attention and use_moga_attn, moga_group = moga_group, moga_lambda = moga_lambda)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(MoGAUpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn,
                moga_atten = up_moga_attention[i] and use_moga_attn,
                moga_group = moga_group,
                moga_lambda = moga_lambda
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, return_mid_feature=False, return_group_logit=False, return_group_loss: bool = False):
        # x: [B, Cin, H, W]
        all_router_logits = []
        total_gb_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        for bid, block in enumerate(self.down_blocks) :
            x, xs, d_logits, d_loss = block(
                x, 
                return_group_logit=return_group_logit, 
                return_group_loss=return_group_loss
            )
            xss.extend(xs)
            if return_group_logit: all_router_logits.extend(d_logits)
            if return_group_loss: total_gb_loss += d_loss
        
        # 2. Mid
        x, m_logits, m_loss = self.mid_block(x,return_group_logit=return_group_logit, return_group_loss=return_group_loss)
        if return_group_logit: all_router_logits.extend(m_logits)
        if return_group_loss: total_gb_loss += m_loss
        mid_feature = (x, xss)
        
        # up
        for bid, block in  enumerate(self.up_blocks):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x, u_logits, u_loss = block(x, xs, return_group_logit=return_group_logit, return_group_loss=return_group_loss)
            if return_group_logit: all_router_logits.extend(u_logits)
            if return_group_loss: total_gb_loss += u_loss

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        res = {}
        if return_group_loss:
            res["loss"] = total_gb_loss
        if return_group_logit:
            res["logits"] = all_router_logits
        res["output"] = x

        if return_mid_feature:
            res["mid_feature"] = mid_feature
        return res if len(res) > 1 else x
    
    def set_capacity_factor(self, new_capacity: float):
        print(f"[MoGA_UNet] Updating capacity_factor -> {new_capacity}")

        # Down blocks
        for block in self.down_blocks:
            for moga_attn in block.moga_attns:
                if moga_attn is not None:
                    moga_attn.capacity_factor = new_capacity

        # Mid block
        for moga_attn in self.mid_block.moga_attns:
            if moga_attn is not None:
                moga_attn.capacity_factor = new_capacity

        # Up blocks
        for block in self.up_blocks:
            for moga_attn in block.moga_attns:
                if moga_attn is not None:
                    moga_attn.capacity_factor = new_capacity
        
    
#     Returns:
#     """
    
    
    
    
    
#     )
    
#     )
    
    
    
    


    

#     Args:

#     Returns:
#     """
    

#     ]
    
    
#     )
    

def _calculate_moga_loss_internal(
    router_logits: torch.Tensor,  # [B, N, M]
    T: float = 1.0
) -> torch.Tensor:

    B, N, M = router_logits.shape

    probs = (router_logits / T).softmax(dim=-1)  # [B, N, M]

    weights_top1, groups = torch.topk(probs, k=1, dim=-1)
    weights_top1 = weights_top1.squeeze(-1)  # [B, N]
    groups = groups.squeeze(-1)              # [B, N]

    # -------- (7) F_i = N_i / N --------
    counts = torch.zeros(B, M, device=probs.device, dtype=probs.dtype)
    counts.scatter_add_(1, groups, torch.ones_like(groups, dtype=probs.dtype))  # [B, M]
    f_i = counts / N  # [B, M]

    # -------- (8) P_i = (1/N) * sum_{g(x)=i} p(g(x)|x) --------
    p_i = torch.zeros(B, M, device=probs.device, dtype=probs.dtype)
    p_i.scatter_add_(1, groups, weights_top1)
    p_i = p_i / N  # [B, M]

    # -------- (6) L_gb(b) = M * sum(F_i(b) * P_i(b)) --------
    losses = M * (f_i * p_i).sum(dim=-1)   # [B]

    return losses.mean()

if __name__ =="__main__":
    moga = MoGAttention(
        dim=128, 
        num_heads=8,
        tokenize=True,
        token_dim=256,
    )

    x = torch.randn(32, 128, 512, 512)
    y = moga(x)
    print(y.shape)
