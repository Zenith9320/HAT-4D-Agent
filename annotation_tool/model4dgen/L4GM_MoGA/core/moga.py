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
    """
    令牌路由器（Token Router）模块，用于计算每个令牌分配给 M 个组的 Logits。
    这是 MoGA 论文中轻量级路由器的实现。
    """
    def __init__(self, dim: int, num_groups: int, router_bias: bool = True):
        super().__init__()
        self.num_groups = num_groups
        # Router 核心：一个简单的线性层 W_R
        self.router_layer = nn.Linear(dim, num_groups, bias=router_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算路由 Logits。

        Args:
            x (torch.Tensor): 输入令牌特征，形状通常为 [B', L, D]，
                              其中 B' 是批次大小（可能包含 B*T*V 的展平），
                              L 是注意力序列长度，D 是特征维度。

        Returns:
            torch.Tensor: 路由 Logits，形状为 [B', L, M]，M 是组数。
        """
        router_logits = self.router_layer(x)
        return router_logits
    

class MoGAttention(nn.Module):
    """
    MoGA（Mixture-of-Groups Attention）实现，适配 FlashAttention/xformers。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_groups: int = 32,  # M
        T: float = 1.0,        # 路由 Softmax 的温度参数
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        router_bias: bool = True,
        num_views: int = 4, # 新增：用于形状恢复
        num_frames: int = 8, # 新增：用于形状恢复
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
    
        # 空间tokenizer
        self.tokenize = tokenize
        # 如果不指定 C'，默认与 dim 相同

        # ★ tokenizer：8×8 conv + stride=8 下采样
        if tokenize:
            token_dim = token_dim or (dim * 4)
            self.tokenizer = nn.Conv2d(dim, token_dim, kernel_size=4, stride=4)
            # Unpatchify：转置卷积，恢复原尺寸
            self.unpatch = nn.ConvTranspose2d(token_dim, dim, kernel_size=4, stride=4)

            dim = token_dim  # MoGA 内部使用 C'
            
        
        # 用于形状恢复
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_groups
        self.T = T
        self.num_frames = num_frames
        self.num_views = num_views
        
        # 1. 令牌路由器
        self.router = Router(dim, num_groups, router_bias)
        
        # 2. QKV 投影和最终投影 (从 Attention/MemEffAttention 中复制)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 3. 注意力模块 (仅用于 FlashAttention 的 forward_varlen 调用)
        # 实际上不需要实例化 MemEffAttention 整个类，只需要其 QKV/Proj 和 forward_varlen 逻辑
        # 假设我们能访问到 MemEffAttention.forward_varlen
        self.attn_op = MemEffAttention.forward_varlen

    def _moga_permute(self, q, k, v, router_logits, T=1.0):
        """
        改进版 MoGA 排列函数：加入硬容量限制 Hard Capacity。
        q,k,v: [B', L, H, Dk]
        router_logits: [B', L, M]
        """

        B_prime, L, H, Dk = q.shape
        M = self.num_groups
        
        # -----------------------------
        # 1. softmax 得到路由权重
        # -----------------------------
        weights = (router_logits / T).softmax(dim=-1)        # [B', L, M]
        weights_top1, groups = torch.topk(weights, k=1, dim=-1)
        
        # 展平
        weights_flat = weights_top1.reshape(-1)              # [N]
        groups_flat  = groups.reshape(-1)                    # [N]
        N = groups_flat.numel()

        # -----------------------------
        # 2. Hard Capacity
        # -----------------------------
        # 容量 = α * (平均每组 token 数)
        # α=1 为严格均匀；你也可以用 1.2 之类让分配更灵活
        capacity_factor = getattr(self, "capacity_factor", 1.0)
        capacity = int(capacity_factor * (N / M))

        # 每组计数器
        counts = torch.zeros(M, dtype=torch.int32, device=groups_flat.device)

        # 最终 group 分配结果
        final_group = torch.empty_like(groups_flat)

        # 顺序遍历 token，执行 capacity 控制
        for idx in range(N):
            g = groups_flat[idx].item()
            
            if counts[g] < capacity:
                # 该组有空位
                final_group[idx] = g
                counts[g] += 1
            else:
                # 超出容量，必须找一个最空闲的组
                alt = torch.argmin(counts).item()
                final_group[idx] = alt
                counts[alt] += 1

        # -----------------------------
        # 3. 按 final_group 排序
        # -----------------------------
        sort_idx = torch.argsort(final_group, stable=True)   # [N]
        
        groups_sorted = final_group[sort_idx]                # [N]
        weights_sorted = weights_flat[sort_idx]              # [N]

        # -----------------------------
        # 4. 计算 cu_seqlens（flash-attn varlen 需要）
        # -----------------------------
        _, counts = torch.unique_consecutive(groups_sorted, return_counts=True)
        cu_seqlens = torch.cat(
            (counts.new_zeros(1, dtype=torch.int32),
            counts.cumsum(0, dtype=torch.int32))
        )   # [M+1]

        # -----------------------------
        # 5. 对 QKV 进行排序
        # -----------------------------
        q_permuted = q.reshape(B_prime * L, H, Dk)[sort_idx]
        k_permuted = k.reshape(B_prime * L, H, Dk)[sort_idx]
        v_permuted = v.reshape(B_prime * L, H, Dk)[sort_idx]

        return q_permuted, k_permuted, v_permuted, sort_idx, weights_sorted, cu_seqlens


    def _moga_reperumte(self, out_permuted, sort_idx, weights_sorted):
        # out_permuted 形状: [Total_Tokens, H, Dk]
        # Total_Tokens = B'*L
        
        # 1. 权重缩放 (Mixture)
        out_weighted = out_permuted * weights_sorted[:, None, None]
        
        # 2. 初始化目标张量
        out_ = torch.zeros_like(out_weighted)
        
        # 3. 重排列 (Repermutation)
        # 将排序后的结果，按照 sort_idx 恢复到原始顺序
        out_.index_add_(0, sort_idx.to(out_.device), out_weighted)
        
        return out_ # 形状: [Total_Tokens, H, Dk]

    def forward(self, x: torch.Tensor, 
                    return_group_logit: bool = False, 
                    return_group_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        MoGA 前向传播，支持特征图输入。
        
        Args:
            x (torch.Tensor): 输入特征图，形状 [B*T*V, C, H, W]。
            router_logits (Optional[torch.Tensor]): 可选的预计算 Logits。
            
        Returns:
            torch.Tensor: 输出特征图，形状 [B*T*V, C, H, W]。
        """

        # -------- ensure submodules have same dtype/device as x -----------
        # self._sync_dtype(x)
    
        BTV, C, H, W = x.shape
            
        # ------------------- (A) Tokenizer --------------------
        if self.tokenize:
            x = self.tokenizer(x)        # [BTV, C', H/8, W/8]
            _, C, H, W = x.shape         # 更新形状
        
        
        # ------------------- 1. 令牌化 (Flattening) -------------------
        
        # 1.1 获取原始 Batch Size B
        B = BTV // (self.num_frames * self.num_views)
        
        # 1.2 重塑维度，准备令牌化
        # [BTV, C, H, W] -> [B, T, V, C, H, W]
        x_reshaped = x.reshape(B, self.num_frames, self.num_views, C, H, W)
        
        # 1.3 令牌化：将 T, V, H, W 展平到 L 维度，并将 C 维度移到最后
        # [B, T, V, C, H, W] -> [B, T, V, H, W, C] -> [B, L_full, C]
        x_flat = x_reshaped.permute(0, 1, 2, 4, 5, 3).contiguous().reshape(B, -1, C)
        
        B_prime, L, D = x_flat.shape # B' = B, L = T*V*H*W, D = C

        # ------------------- 2. MoGA 核心计算 (使用 x_flat) -------------------
        
        
        # if self.qkv.weight.dtype != x.dtype:
        #     self.qkv = self.qkv.to(x.dtype)
        #     self.proj = self.proj.to(x.dtype)
        #     self.router = self.router.to(x.dtype)
    
        
        # 2.1 QKV 投影和分头
        qkv = self.qkv(x_flat).reshape(B_prime, L, 3, self.num_heads, self.head_dim)
        q, k, v = unbind(qkv, 2) 

        # 2.2 计算 Router Logits
        router_logits = self.router(x_flat) # [B, L, M]
        
        # --- 显存优化：内部计算 Loss 标量 ---
        gb_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if return_group_loss:
            gb_loss = _calculate_moga_loss_internal(router_logits, self.T)
            
        # 2.3 排列与分组
        q_p, k_p, v_p, sort_idx, weights_sorted, cu_seqlens = \
            self._moga_permute(q, k, v, router_logits, self.T)
        
        # 2.4 组内注意力计算 (Intra-Group Attention)
        max_seqlen = cu_seqlens.max().item()
        max_seqlen = cu_seqlens.max().item()

        # ★★★ 打印最长组长度 ★★★
        # print(f"[MoGA] Longest group length (max_seqlen): {max_seqlen}")

        out_permuted = self.attn_op(
            q_p, k_p, v_p, 
            cu_seqlens=cu_seqlens, 
            max_seqlen=max_seqlen
        )
        
        # 2.5 重排列 (_moga_reperumte)
        out_original_order = self._moga_reperumte(out_permuted, sort_idx, weights_sorted) 
        
        # 2.6 最终投影
        out_original_order = out_original_order.reshape(B_prime, L, D) 
        out_flat = self.proj(out_original_order)
        out_flat = self.proj_drop(out_flat)
        
        # ------------------- 3. 反令牌化 (Unflattening) -------------------
        
        # 3.1 恢复维度
        # [B, L_full, C] -> [B, T, V, H, W, C]
        out_reshaped = out_flat.reshape(B, self.num_frames, self.num_views, H, W, C)
        
        # 3.2 恢复特征图形状： C 维度移回第二位，并展平 B, T, V
        # [B, T, V, H, W, C] -> [B, T, V, C, H, W] -> [BTV, C, H, W]
        out = out_reshaped.permute(0, 1, 2, 5, 3, 4).contiguous().reshape(BTV, C, H, W)
        
        # ------------------ C. Unpatchify ------------------
        if self.tokenize:
            out = self.unpatch(out)     # [BTV, orig_C, orig_H, orig_W]
            
        # 记录需要返回的 Logit (如果 return_group_logit 为 False，则返回 None 节省显存引用)
        out_logits = router_logits if return_group_logit else None
        
        # print("MoGA out:", out.grad)
        
        return out, router_logits, gb_loss
    
    # def _sync_dtype(self, x):
    #     dtype = x.dtype
    #     device = x.device

    #     # qkv, proj, router
    #     for m in [self.qkv, self.proj, self.router]:
    #         m.to(dtype=dtype, device=device)

    #     # tokenizer & unpatch
    #     if self.tokenize:
    #         self.tokenizer.to(dtype=dtype, device=device)
    #         self.unpatch.to(dtype=dtype, device=device)
            
    
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
        moga_attns = [] # 新增
        
        for i in range(num_layers):
            in_channels_i = in_channels if i == 0 else out_channels
            
            # 1. ResNet 块
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
                # 注意：MoGA 的输入/输出维度是 out_channels
                moga_attns.append(MoGAttention(out_channels, attention_heads, moga_group, num_views=num_views, num_frames=num_frames))
            else:
                moga_attns.append(None)
            
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        self.moga_attns = nn.ModuleList(moga_attns) # 新增
        self.moga_lambda = moga_lambda # 新增

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    # --- 修改 MoGADownBlock ---
    def forward(self, x, return_group_logit: bool = False, return_group_loss: bool = False): # 增加参数
        xs = []
        router_logits = [] 
        accumulated_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for attn, t_attn, net, moga_attn in zip(self.attns, self.t_attns, self.nets, self.moga_attns):
            x = net(x) 

            if moga_attn:
                # 只有当需要时才接收并保存 logit
                moga_x, moga_logit, moga_l = moga_attn(x, return_group_logit, return_group_loss)
                if return_group_logit: router_logits.append(moga_logit)
                if return_group_loss: accumulated_loss += moga_l
            
            if attn: x = attn(x)
            if t_attn: x = t_attn(x)
                
            if moga_attn:
                x = x + moga_x * self.moga_lambda
                # print("moga_x * self.moga_lambda:",moga_x * self.moga_lambda)
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
        
    def forward(self, x, return_group_logit: bool = False, return_group_loss: bool = False): # 增加参数
        x = self.nets[0](x)
        router_logits = [] # ★ 新增：用于收集 Logits
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
        moga_attns = [] # 新增
        
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels
            
            # 1. ResNet 块 (输入维度需要加上跳跃连接的维度 cskip)
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
        self.moga_attns = nn.ModuleList(moga_attns) # 新增
        self.moga_lambda = moga_lambda # 新增

        self.upsample = None
        if upsample:
            # 保持不变，用于最后的上采样操作
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


        
    def forward(self, x, xs, return_group_logit: bool = False, return_group_loss: bool = False): # 增加参数
        # 假设 self.nets 是用来决定循环次数的
        # 跳跃连接的数量 (len(xs)) 必须等于 ResNet 块的数量 (len(self.nets))
        assert len(xs) == len(self.nets), \
            f"UpBlock中跳跃连接的数量 {len(xs)} 必须与层数 {len(self.nets)} 匹配。"
            
        router_logits = [] # ★ 新增：用于收集 Logits
        accumulated_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for attn, t_attn, net, moga_attn in zip(self.attns, self.t_attns, self.nets, self.moga_attns):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            
            if moga_attn:
                # ★ 核心修改：MoGA 返回 (输出, Logits) 
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

        # ★ 核心修改：返回 (输出, Logits列表)
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
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

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


    def forward(self, x, return_mid_feature=False, return_group_logit=False, return_group_loss: bool = False): # 增加参数
        # x: [B, Cin, H, W]
        all_router_logits = [] # ★ 新增：总 Logits 列表
        total_gb_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        for bid, block in enumerate(self.down_blocks) :
            # ★ 核心修改：接收 block 返回的 Logits
            # print(f"DownBlock {bid} x: {x.shape}")
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
            # ★ 核心修改：接收 block 返回的 Logits
            # print(f"UpBlock {bid} x: {x.shape}")
            x, u_logits, u_loss = block(x, xs, return_group_logit=return_group_logit, return_group_loss=return_group_loss)
            if return_group_logit: all_router_logits.extend(u_logits)
            if return_group_loss: total_gb_loss += u_loss

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        # --- 动态返回处理 ---
        # 动态组装返回元组
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
        """
        动态修改所有 MoGAttention 层的 capacity_factor。
        训练过程中可随时调用：unet.set_capacity_factor(v)
        """
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
        
# def _calculate_group_balancing_loss_from_flattened(
#     router_logits: torch.Tensor, 
#     num_groups: int, 
#     T: float = 1.0,
# ) -> torch.Tensor:
#     """
#     内部函数：计算 Group Balancing Loss 的核心部分 (M * sum(F_i * P_i))，
#     输入必须是 [N, M] 形状的展平 Logits。
    
#     Returns:
#         torch.Tensor: L_gb 标量损失的核心部分 (M * sum(F_i * P_i))。
#     """
    
#     # 将 Logits 展平为 [N, M]，N 是总 Token 数
#     N = router_logits.shape[0] 
    
#     # 1. 计算路由概率 P_route: p(g(x) | x)，形状 [N, M]
#     P_route = (router_logits / T).softmax(dim=-1)
    
#     # 2. 确定 Top-1 组分配 g(x) (硬分配)
#     _, g_x = torch.max(P_route, dim=-1) # g_x 形状: [N]
#     g_x_expanded = g_x.unsqueeze(-1) # [N, 1]
    
#     # 初始化累加器（形状 [M]）
#     # --- 修复点：确保使用 P_route 的 dtype ---
#     device = router_logits.device
#     # 使用 P_route 的 dtype，因为它是 Top1_Probs (src) 的 dtype。
#     dtype = P_route.dtype
#     N_i = torch.zeros(num_groups, dtype=dtype, device=device)
#     Sum_Prob_i = torch.zeros(num_groups, dtype=dtype, device=device)
    
#     # 3. 向量化计算 N_i (公式 7 的分子: Token 计数)
#     N_i.scatter_add_(
#         dim=0, 
#         index=g_x, 
#         src=torch.ones_like(g_x, dtype=dtype)
#     )
    
#     # 4. 向量化计算 Sum_Prob_i (公式 8 的分子: 软概率之和)
#     Top1_Probs = P_route.gather(dim=1, index=g_x_expanded).squeeze(-1) # [N]
#     Sum_Prob_i.scatter_add_(
#         dim=0, 
#         index=g_x, 
#         src=Top1_Probs
#     )
    
#     # 5. 计算 F_i (公式 7): F_i = N_i / N
#     F_i = N_i / N # [M]
    
#     # 6. 计算 P_i (公式 8): P_i = Sum_Prob_i / N_i
#     epsilon = 1e-6
#     N_i_safe = N_i.clamp(min=epsilon)
#     P_i = Sum_Prob_i / N_i_safe # [M]
    
#     # 7. 计算 L_gb 的核心部分 (公式 6 的 M * sum(F_i * P_i))
#     L_gb_core = num_groups * (F_i * P_i).sum()
    
#     return L_gb_core


# def calculate_group_balancing_loss_from_list(
#     group_logit_list: List[torch.Tensor], 
#     num_groups: int, 
#     T: float = 1.0, 
# ) -> torch.Tensor:
#     """
#     计算 Group Balancing Loss (L_gb) 的外部包装函数，接受 Logits 张量的列表输入。
#     返回 L_gb 的核心部分：M * sum(F_i * P_i)。
    
#     注意：此函数返回的值需要手动乘以损失权重 alpha。

#     Args:
#         group_logit_list (List[torch.Tensor]): 路由器的 Logits 张量列表，
#                                                每个形状为 [B, L, M] 或 [B', L', M]。
#         num_groups (int): 组数 M。
#         T (float): 路由 Softmax 的温度参数。

#     Returns:
#         torch.Tensor: L_gb 标量损失的核心部分 (M * sum(F_i * P_i))。
#     """
    
#     if not group_logit_list:
#         # 如果列表为空，返回 0 损失
#         device = group_logit_list[0].device if group_logit_list else ('cuda' if torch.cuda.is_available() else 'cpu')
#         return torch.tensor(0.0, device=device)

#     # 1. 预处理：使用列表推导式对每个张量执行展平 (B, L, M) -> (B*L, M)
#     flattened_logits = [
#         logit_tensor.reshape(-1, num_groups) 
#         for logit_tensor in group_logit_list
#     ]
    
#     # 2. 垂直拼接：将所有展平后的 Logits 在 Token 维度上堆叠起来
#     all_router_logits = torch.cat(flattened_logits, dim=0)
    
#     # 3. 调用内部函数进行损失核心部分的计算
#     overall_loss_core = _calculate_group_balancing_loss_from_flattened(
#         router_logits=all_router_logits,
#         num_groups=num_groups,
#         T=T,
#     )
    
#     return overall_loss_core

def _calculate_moga_loss_internal(
    router_logits: torch.Tensor,  # [B, N, M]
    T: float = 1.0
) -> torch.Tensor:
    """
    支持 Batch 并行的 Group Balancing Loss（MoGA L_gb）
    输入:
        router_logits: [B, N, M]
    输出:
        标量 loss (所有 batch 的 L_gb 求平均)
    """

    B, N, M = router_logits.shape

    # -------- Softmax 概率 p(g(x)|x) --------
    probs = (router_logits / T).softmax(dim=-1)  # [B, N, M]

    # -------- 找到硬分配组: groups[b, n] ∈ [0..M-1] --------
    weights_top1, groups = torch.topk(probs, k=1, dim=-1)
    weights_top1 = weights_top1.squeeze(-1)  # [B, N]
    groups = groups.squeeze(-1)              # [B, N]

    # -------- (7) F_i = N_i / N --------
    # N_i[b, i] = 每个 batch 中分配到 group i 的 token 数
    counts = torch.zeros(B, M, device=probs.device, dtype=probs.dtype)
    counts.scatter_add_(1, groups, torch.ones_like(groups, dtype=probs.dtype))  # [B, M]
    f_i = counts / N  # [B, M]

    # -------- (8) P_i = (1/N) * sum_{g(x)=i} p(g(x)|x) --------
    p_i = torch.zeros(B, M, device=probs.device, dtype=probs.dtype)
    p_i.scatter_add_(1, groups, weights_top1)  # 累加概率
    p_i = p_i / N  # [B, M]

    # -------- (6) L_gb(b) = M * sum(F_i(b) * P_i(b)) --------
    losses = M * (f_i * p_i).sum(dim=-1)   # [B]

    # -------- 对所有 batch 求平均 --------
    return losses.mean()

if __name__ =="__main__":
    moga = MoGAttention(
        dim=128, 
        num_heads=8,
        tokenize=True,
        token_dim=256,   # tokenizer 输出 C'
    )

    x = torch.randn(32, 128, 512, 512)
    y = moga(x)
    print(y.shape)