# Modified by HAT-4D; see the subtree MODIFICATIONS.md.
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language bgoverning permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial

from core.attention import MemEffAttention
from typing import Tuple, Literal, Dict, Optional


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_views: int = 4, 
        num_frames: int = 8
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_views = num_views
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x, return_attn=False):
        # x: [B*T*V, C, H, W]
        BTV, C, H, W = x.shape
        BT = BTV // self.num_views

        res = x
        x = self.norm(x)

        x = x.reshape(BT, self.num_views, C, H, W).permute(0, 1, 3, 4, 2).contiguous().reshape(BT, -1, C).contiguous()
        if not return_attn:
            x = self.attn(x, return_attn=False) 
        else:
            x, atten_weight = self.attn(x, return_attn=True, num_frames=self.num_views)
        x = x.reshape(BT, self.num_views, H, W, C).permute(0, 1, 4, 2, 3).contiguous().reshape(BTV, C, H, W).contiguous()


        if self.residual:
            x = (x + res) * self.skip_scale
        if not return_attn:
            return x
        else:
            return x, atten_weight


    

class TempAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_views = num_views
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x, return_attn=False,attn_resize: Optional[Tuple[int, int]] = None, zero_out_cross_frame=False):
        # x: [B*T*V, C, H, W]
        BTV, C, H, W = x.shape
        BV = BTV // self.num_frames
        B = BV // self.num_views

        res = x
        x = self.norm(x)

        # BTV -> BVT
        x = x.reshape(B, self.num_frames, self.num_views, C, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()

        x = x.reshape(BV, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).contiguous().reshape(BV, -1, C).contiguous().contiguous()
        if not return_attn:
            x = self.attn(x, return_attn= return_attn, zero_out_cross_frame=zero_out_cross_frame)
        else:
            x, attn_weight = self.attn(x, return_attn= return_attn, num_frames=self.num_frames, zero_out_cross_frame=zero_out_cross_frame)
            
        x = x.reshape(BV, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).contiguous().reshape(BTV, C, H, W).contiguous().contiguous()

        # BVT -> BTV
        x = x.reshape(B, self.num_views, self.num_frames, C, H, W).permute(0, 2, 1, 3, 4, 5).contiguous().reshape(BTV, C, H, W).contiguous()
        
        if not return_attn:
            if self.residual:
                x = (x + res) * self.skip_scale
            return x
        else:
            if self.residual:
                x = (x + res) * self.skip_scale
            return x, attn_weight
            


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
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
        use_temp_attn=True
    ):
        super().__init__()
 
        nets = []
        attns = []
        t_attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        
        self.num_views = num_views
        self.num_frames = num_frames

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, attn_rollout= False, zero_out_cross_frame=False):
        if attn_rollout:
            return self.forward_rollout(x)
        
        xs = []
        print(f"in downblock forward x_shape:",x.shape)

        for attn, t_attn, net in zip(self.attns, self.t_attns, self.nets):
            
            x = net(x)
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x, zero_out_cross_frame= zero_out_cross_frame)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
            

  
        return x, xs
    
    def forward_rollout(self,x):
        xs = []
        print(f"x_shape:",x.shape)
        
        BTV, C, H, W = x.shape
        
        V = self.num_views
        T = self.num_frames
        L = T * V 
        B = BTV // T // V
        
        
        rollout = torch.eye(L, device=x.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        rollout = rollout / rollout.sum(dim=-1, keepdim=True)
    
        for attn, t_attn, net in zip(self.attns, self.t_attns, self.nets):
            x = net(x)
            if attn:
                x, mv_attn_map = attn(x, return_attn=True)  # mv_attn_map: [BT, H, V, V]
                mv_attn_map = torch.mean(mv_attn_map, dim=1)
                mv_attn_map = torch.reshape(mv_attn_map, (B, T,V,V ))
                
                
                                
                visualize_attention_heatmap(
                    mv_attn_map[:, 0],
                    title="mv_attn_map - T0",
                    filename="mv_attn_map_T0.png"
                )
                mv_rollout = torch.zeros(B, L, L, device=x.device)
                for t in range(T):
                    idx_t = torch.arange(t * V , (t + 1) * V , device=x.device)
                    mv_rollout[:, idx_t[:, None], idx_t[None, :]] = mv_attn_map[:, t]
                    
                
                
                visualize_attention_heatmap(
                    mv_rollout,  # [B, L, L]
                    title="mv_rollout",
                    filename="mv_rollout.png"
                )
                
                mv_rollout = mv_rollout / mv_rollout.sum(dim=-1,keepdim=True)
                
                rollout = torch.bmm(rollout, mv_rollout)
                rollout = rollout / rollout.sum(dim=-1, keepdim=True)

                visualize_attention_heatmap(
                    rollout,  # [B, L, L]
                    title="final_rollout",
                    filename="after_mv_final_rollout.png"
                )
                
            if t_attn:
                x, temp_attn_map = t_attn(x, return_attn=True)  # [B, V, THW, THW]
                temp_attn_map = torch.mean(temp_attn_map, dim=1)
                temp_attn_map = torch.reshape(temp_attn_map, (B, V,T,T ))
                
                visualize_attention_heatmap(
                    temp_attn_map[:, 0],
                    title="temp_attn_map - V0",
                    filename="temp_attn_map_V0.png"
                )
                
                temp_rollout = torch.zeros(B, L, L, device=x.device)
                    
                for v in range(V):
                    idx_v = []
                    for t in range(T):
                        idx = torch.arange(t*V+ v, t *V+ (v+1), device=x.device)
                        idx_v.append(idx)
                    idx_v = torch.cat(idx_v, dim=0)  # [THW]
                    

                    temp_rollout[:, idx_v[:, None], idx_v[None, :]] = temp_attn_map[:, v]
     
                    visualize_attention_heatmap(
                        temp_rollout,  # [B, L, L]
                        title="temp_rollout",
                        filename="temp_rollout.png"
                    )
                                    

                temp_rollout = temp_rollout / temp_rollout.sum(dim=-1,keepdim=True)
                    
                rollout = torch.bmm(rollout, temp_rollout)
                
                rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        
                visualize_attention_heatmap(
                    rollout,  # [B, L, L]
                    title="final_rollout",
                    filename="after_temp_final_rollout.png"
                )
                
                
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
            
        print("after downblock forward:" , rollout.shape)
  
        return x, xs, rollout
        


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True
    ):
        super().__init__()

        nets = []
        attns = []
        t_attns = []
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
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, t_attn,net in zip(self.attns, self.t_attns, self.nets[1:]):
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
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
        use_temp_attn=True
    ):
        super().__init__()

        nets = []
        attns = []
        t_attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)



    def forward(self, x, xs, return_attn=False, attn_hook=None,attn_resize: Optional[Tuple[int, int]] = None, zero_out_cross_frame=False):
        # attn_hook: callable, (attn_weights: Tensor, layer_index: int) -> None
        
        temp_attn_weights = []
        view_attn_weights = []
        print("Upblock return_atten:", return_attn)

        for i, (attn, t_attn, net) in enumerate(zip(self.attns, self.t_attns, self.nets)):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)

            if attn:
                if return_attn:
                    x, attn_weights = attn(x, return_attn=True)
                    view_attn_weights.append(attn_weights)
                    
                    if attn_hook is not None:
                        attn_hook(attn_weights, i)  # external hook to process weights
                    
                else:
                    x = attn(x,return_attn=False)  # MVAttention does not return attention maps for now
                    

            if t_attn:
                if return_attn:
                    x, attn_weights = t_attn(x, return_attn=True, attn_resize= attn_resize, zero_out_cross_frame=zero_out_cross_frame)
                    print(f"{i}-th t_attn num_frames:", t_attn.num_frames)
                    temp_attn_weights.append(attn_weights)
                    
                    if attn_hook is not None:
                        attn_hook(attn_weights, i)  # external hook to process weights
                else:
                    x = t_attn(x, return_attn=False,  zero_out_cross_frame=zero_out_cross_frame)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        if return_attn:
            return x, temp_attn_weights, view_attn_weights
        else:
            return x

# it could be asymmetric!
class UNet(nn.Module):
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
        use_temp_attn: bool = True
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
            
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames, use_temp_attn=use_temp_attn)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
        
    
            
    def viz_view_attention(self, views_atten_weight, save_dir='views_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(views_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {views_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(views_atten_weight):
            B, H, V, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [V, V]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - T {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_T_{b}.png')
                plt.savefig(save_path)
                plt.close()

                

    def viz_temporal_attention(self, temporal_atten_weight, save_dir='temp_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(temporal_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {temporal_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(temporal_atten_weight):
            B, H, T, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [T, T]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - View {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_view{b}.png')
                plt.savefig(save_path)
                plt.close()

                
    def save_temporal_attention(self, temporal_atten_weight, save_dir="temp_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(temporal_atten_weight, attn_save_path)
        
    def save_view_attention(self, cross_atten_weight, save_dir="cross_view_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
       
        if not back:
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(cross_atten_weight, attn_save_path)
        
    def save_rollout_attention(self, rollout_atten_weight, save_dir="rollout_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{start_t}_to_{end_t}_block{layer_id}.pt')
        else:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_block{layer_id}.pt')
        torch.save(rollout_atten_weight, attn_save_path)
        print(f"[SAVED] Rollout attention weights saved to: {attn_save_path}")
            

    def forward(self, x, return_mid_feature=False, viz_last_up_attention=False, viz_atten_size=None, save_atten_dir="temp_attn_torch", start_t=0, end_t=1, downblock_rollout=False, back=False, back_num=0, add_mid_feature= None, mid_add_alpha = 0.5):
        # x: [B, Cin, H, W]
        B = 1
        
        
        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        if downblock_rollout:
            BTV, C, H, W = x.shape
            L = BTV 
            
            rollout = torch.eye(L, device=x.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        
            for bid, block in enumerate(self.down_blocks):
                x, xs, rollout_block = block(x, attn_rollout=True)
                xss.extend(xs)
                print("rollout_block:" , rollout_block.shape)
                print("rollout:" , rollout.shape)
                
                rollout = torch.bmm(rollout, rollout_block)
                self.save_rollout_attention(rollout, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=bid, back=back,back_num= back_num)
                
    
        else:
            for block in self.down_blocks:
                print("in Unet forward x_shape:", x.shape)
                x, xs = block(x)
                xss.extend(xs)
            
        # mid
        x = self.mid_block(x)
        mid_feature = (x, xss)
        
        if add_mid_feature != None:
            print("merge mid feature!")
            print("xss:", len(xss))

            print(mid_add_alpha)
            
            print("x.shape:", x.shape)
            print("mid_feature[0]:",  mid_feature[0].shape)
            
            save_feature_dir = os.environ.get("L4GM_MID_FEATURE_DIR")
            if save_feature_dir:
                os.makedirs(save_feature_dir, exist_ok=True)
                pt_count = sum(
                    1 for file in os.listdir(save_feature_dir)
                    if file.endswith(".pt")
                )
                
                torch.save(x.detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_back_{pt_count}.pt"))
                torch.save(mid_feature[0].detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_front_{pt_count}.pt"))
                
                print("Save Mid feature:", os.path.join(save_feature_dir, "blended_feature_x_back.pt"))
                
                
       
            
            
            x = mid_add_alpha *  mid_feature[0] + (1 - mid_add_alpha)*  x
            
            for idx in range(len(xss)):
                print(f"xss[{idx}]:" , xss[idx].shape)
                print(f"xss[{idx}]:" ,  mid_feature[1][idx].shape)
                xss[idx] =  mid_add_alpha * mid_feature[1][idx]  + (1- mid_add_alpha)  * xss[idx]
            
        

        # # up
            
        # up, Modified By jiaxin, Add viz
        print("[SHAPE] Upblocks:", len(self.up_blocks))
        temp_attn_weights = None
        view_attn_weights = None
        for i, block in enumerate(self.up_blocks):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]

            if viz_last_up_attention and i == len(self.up_blocks) - 3:
                x, temp_attn_weights,view_attn_weights = block(x, xs, return_attn=True, attn_resize= viz_atten_size )
                self.viz_temporal_attention(temp_attn_weights)
                self.save_temporal_attention(temp_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
                self.viz_view_attention(view_attn_weights, save_dir="view_atten_vis")
                self.save_view_attention(view_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
            else:
                x = block(x, xs)
                
        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']
        
        
        

        if return_mid_feature:
            return x, *mid_feature

        return x



# it could be asymmetric!
class UNet_CAG(nn.Module):
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
        use_temp_attn: bool = True
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames, use_temp_attn=use_temp_attn)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
        
    
            
    def viz_view_attention(self, views_atten_weight, save_dir='views_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(views_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {views_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(views_atten_weight):
            B, H, V, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [V, V]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - T {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_T_{b}.png')
                plt.savefig(save_path)
                plt.close()

                

    def viz_temporal_attention(self, temporal_atten_weight, save_dir='temp_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(temporal_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {temporal_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(temporal_atten_weight):
            B, H, T, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [T, T]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - View {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_view{b}.png')
                plt.savefig(save_path)
                plt.close()

                
    def save_temporal_attention(self, temporal_atten_weight, save_dir="temp_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(temporal_atten_weight, attn_save_path)
        
    def save_view_attention(self, cross_atten_weight, save_dir="cross_view_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
       
        if not back:
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(cross_atten_weight, attn_save_path)
        
    def save_rollout_attention(self, rollout_atten_weight, save_dir="rollout_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{start_t}_to_{end_t}_block{layer_id}.pt')
        else:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_block{layer_id}.pt')
        torch.save(rollout_atten_weight, attn_save_path)
        print(f"[SAVED] Rollout attention weights saved to: {attn_save_path}")
            

    def forward(self, x, return_mid_feature=False, viz_last_up_attention=False, viz_atten_size=None, save_atten_dir="temp_attn_torch", start_t=0, end_t=1, downblock_rollout=False, back=False, back_num=0, add_mid_feature= None, mid_add_alpha = 0.5,zero_out_epsilon = 0.1 ):
        # x: [B, Cin, H, W]
        B = 1
        
        
        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        xss_zero_out = [x]
        x_zero_out = x
        
        if downblock_rollout:
            BTV, C, H, W = x.shape
            L = BTV 
            
            rollout = torch.eye(L, device=x.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        
            for bid, block in enumerate(self.down_blocks):
                x, xs, rollout_block = block(x, attn_rollout=True)
                xss.extend(xs)
                print("rollout_block:" , rollout_block.shape)
                print("rollout:" , rollout.shape)
                
                rollout = torch.bmm(rollout, rollout_block)
                self.save_rollout_attention(rollout, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=bid, back=back,back_num= back_num)
                
    
        else:
            for block in self.down_blocks:
                print("in Unet forward x_shape:", x.shape)
                x, xs = block(x)
                xss.extend(xs)
                
        print("--------------Begin Zero Out --------------")
                
        
                
        if downblock_rollout:
            BTV, C, H, W = x_zero_out.shape
            L = BTV 
            
            rollout = torch.eye(L, device=x_zero_out.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        
            for bid, block in enumerate(self.down_blocks):
                x_zero_out, xs_zero_out, rollout_block = block(x_zero_out, attn_rollout=True, zero_out_cross_frame=True)
                xss_zero_out.extend(xs_zero_out)
                print("rollout_block:" , rollout_block.shape)
                print("rollout:" , rollout.shape)
                
                rollout = torch.bmm(rollout, rollout_block)
                self.save_rollout_attention(rollout, save_atten_dir + "_zero_out",start_t=start_t, end_t=end_t, layer_id=bid, back=back,back_num= back_num)
                
    
        else:
            for block in self.down_blocks:
                print("Zero Out, in Unet forward x_shape:", x.shape)
                x_zero_out, xs_zero_out = block(x_zero_out, zero_out_cross_frame=True)
                xss_zero_out.extend(xs_zero_out)
        
        
                
        x = x + zero_out_epsilon * (x - x_zero_out)
        
        
        
            
        # mid
        x = self.mid_block(x)
        mid_feature = (x, xss)
        
        if add_mid_feature != None:
            print("merge mid feature!")
            print("xss:", len(xss))

            print(mid_add_alpha)
            
            print("x.shape:", x.shape)
            print("mid_feature[0]:",  mid_feature[0].shape)
            
            save_feature_dir = os.environ.get("L4GM_MID_FEATURE_DIR")
            if save_feature_dir:
                os.makedirs(save_feature_dir, exist_ok=True)
                pt_count = sum(
                    1 for file in os.listdir(save_feature_dir)
                    if file.endswith(".pt")
                )
                
                torch.save(x.detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_back_{pt_count}.pt"))
                torch.save(mid_feature[0].detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_front_{pt_count}.pt"))
                
                print("Save Mid feature:", os.path.join(save_feature_dir, "blended_feature_x_back.pt"))
                
                
       
            
            
            x = mid_add_alpha *  mid_feature[0] + (1 - mid_add_alpha)*  x
            
            for idx in range(len(xss)):
                print(f"xss[{idx}]:" , xss[idx].shape)
                print(f"xss[{idx}]:" ,  mid_feature[1][idx].shape)
                xss[idx] =  mid_add_alpha * mid_feature[1][idx]  + (1- mid_add_alpha)  * xss[idx]
            
        

        # # up
            
        # up, Modified By jiaxin, Add viz
        print("[SHAPE] Upblocks:", len(self.up_blocks))
        temp_attn_weights = None
        view_attn_weights = None
        for i, block in enumerate(self.up_blocks):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]

            if viz_last_up_attention and i == len(self.up_blocks) - 3:
                x, temp_attn_weights,view_attn_weights = block(x, xs, return_attn=True, attn_resize= viz_atten_size )
                self.viz_temporal_attention(temp_attn_weights)
                self.save_temporal_attention(temp_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
                self.viz_view_attention(view_attn_weights, save_dir="view_atten_vis")
                self.save_view_attention(view_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
            else:
                x = block(x, xs)
                
        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']
        
        
        

        if return_mid_feature:
            return x, *mid_feature

        return x
    

# it could be asymmetric!
class UNet_CAG_UpBlock(nn.Module):
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
        use_temp_attn: bool = True
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames, use_temp_attn=use_temp_attn)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
        
    
            
    def viz_view_attention(self, views_atten_weight, save_dir='views_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(views_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {views_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(views_atten_weight):
            B, H, V, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [V, V]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - T {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_T_{b}.png')
                plt.savefig(save_path)
                plt.close()

                

    def viz_temporal_attention(self, temporal_atten_weight, save_dir='temp_attn_vis'):
        os.makedirs(save_dir, exist_ok=True)
        
        num_layers = len(temporal_atten_weight)
        print(f"[LEN] temporal_atten_weight: {num_layers}")
        print(f"[SHAPE] temporal_atten_weight[0]: {temporal_atten_weight[0].shape}")  # [B, H, T, T]

        for layer_idx, layer_attn in enumerate(temporal_atten_weight):
            B, H, T, _ = layer_attn.shape
            
            mean_attn = layer_attn.mean(dim=1)

            for b in range(B):
                attn_map = mean_attn[b].detach().cpu().numpy()  # shape [T, T]
                
                plt.figure(figsize=(4, 4))
                plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} - View {b}')
                plt.xlabel('Attending to Frame')
                plt.ylabel('Query Frame')

                save_path = os.path.join(save_dir, f'attn_layer{layer_idx}_view{b}.png')
                plt.savefig(save_path)
                plt.close()

                
    def save_temporal_attention(self, temporal_atten_weight, save_dir="temp_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'temporal_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(temporal_atten_weight, attn_save_path)
        
    def save_view_attention(self, cross_atten_weight, save_dir="cross_view_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
       
        if not back:
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{start_t}_to_{end_t}_layer{layer_id}.pt')
        else: 
            attn_save_path = os.path.join(save_dir, f'cross_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_layer{layer_id}.pt')
        torch.save(cross_atten_weight, attn_save_path)
        
    def save_rollout_attention(self, rollout_atten_weight, save_dir="rollout_attn_torch", start_t=0, end_t=1, layer_id=0, back=False, back_num=0):
        
        if not back:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{start_t}_to_{end_t}_block{layer_id}.pt')
        else:
            attn_save_path = os.path.join(save_dir, f'rollout_view_atten_from_{back_num - start_t}_to_{back_num - end_t}_block{layer_id}.pt')
        torch.save(rollout_atten_weight, attn_save_path)
        print(f"[SAVED] Rollout attention weights saved to: {attn_save_path}")
            

    def forward(self, x, return_mid_feature=False, viz_last_up_attention=False, viz_atten_size=None, save_atten_dir="temp_attn_torch", start_t=0, end_t=1, downblock_rollout=False, back=False, back_num=0, add_mid_feature= None, mid_add_alpha = 0.5,zero_out_epsilon = 0.1 ):
        # x: [B, Cin, H, W]
        B = 1
        
        
        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        xss_zero_out = [x]
        x_zero_out = x
        
        if downblock_rollout:
            BTV, C, H, W = x.shape
            L = BTV 
            
            rollout = torch.eye(L, device=x.device).unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]
        
            for bid, block in enumerate(self.down_blocks):
                x, xs, rollout_block = block(x, attn_rollout=True)
                xss.extend(xs)
                print("rollout_block:" , rollout_block.shape)
                print("rollout:" , rollout.shape)
                
                rollout = torch.bmm(rollout, rollout_block)
                self.save_rollout_attention(rollout, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=bid, back=back,back_num= back_num)
                
    
        else:
            for block in self.down_blocks:
                print("in Unet forward x_shape:", x.shape)
                x, xs = block(x)
                xss.extend(xs)
                
                
        
                
            
        
                
                
    
        
        
                
        
        
        
            
        # mid
        x = self.mid_block(x)
        
        mid_feature = (x, xss)
        
        
        
        if add_mid_feature != None:
            print("merge mid feature!")
            print("xss:", len(xss))

            print(mid_add_alpha)
            
            print("x.shape:", x.shape)
            print("mid_feature[0]:",  mid_feature[0].shape)
            
            save_feature_dir = os.environ.get("L4GM_MID_FEATURE_DIR")
            if save_feature_dir:
                os.makedirs(save_feature_dir, exist_ok=True)
                pt_count = sum(
                    1 for file in os.listdir(save_feature_dir)
                    if file.endswith(".pt")
                )
                
                torch.save(x.detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_back_{pt_count}.pt"))
                torch.save(mid_feature[0].detach().cpu(), os.path.join(save_feature_dir, f"blended_feature_x_front_{pt_count}.pt"))
                
                print("Save Mid feature:", os.path.join(save_feature_dir, "blended_feature_x_back.pt"))
                
                
       
            
            
            x = mid_add_alpha *  mid_feature[0] + (1 - mid_add_alpha)*  x
            
            for idx in range(len(xss)):
                print(f"xss[{idx}]:" , xss[idx].shape)
                print(f"xss[{idx}]:" ,  mid_feature[1][idx].shape)
                xss[idx] =  mid_add_alpha * mid_feature[1][idx]  + (1- mid_add_alpha)  * xss[idx]
            
        

        # # up
            
        # up, Modified By jiaxin, Add viz
        print("[SHAPE] Upblocks:", len(self.up_blocks))
        temp_attn_weights = None
        view_attn_weights = None
        
        
        x_zero_out = x
        xss_zero_out = xss
        
        
        for i, block in enumerate(self.up_blocks):
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]

            if viz_last_up_attention and i == len(self.up_blocks) - 3:
                x, temp_attn_weights,view_attn_weights = block(x, xs, return_attn=True, attn_resize= viz_atten_size )
                self.viz_temporal_attention(temp_attn_weights)
                self.save_temporal_attention(temp_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
                self.viz_view_attention(view_attn_weights, save_dir="view_atten_vis")
                self.save_view_attention(view_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
            else:
                x = block(x, xs)
                
                
        print("--------------Begin UpBlock Zero Out --------------")
                
        for i, block in enumerate(self.up_blocks):
            xs = xss_zero_out[-len(block.nets):]
            xss_zero_out = xss_zero_out[:-len(block.nets)]

            if viz_last_up_attention and i == len(self.up_blocks) - 3:
                x_zero_out, temp_attn_weights,view_attn_weights = block(x_zero_out, xs, return_attn=True, attn_resize= viz_atten_size )
                self.viz_temporal_attention(temp_attn_weights)
                self.save_temporal_attention(temp_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
                self.viz_view_attention(view_attn_weights, save_dir="view_atten_vis")
                self.save_view_attention(view_attn_weights, save_atten_dir,start_t=start_t, end_t=end_t, layer_id=i, back=back,back_num= back_num)
                
            else:
                x_zero_out = block(x_zero_out, xs)
                
        x = x + (zero_out_epsilon) * (x - x_zero_out)
                
        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']
        
        
        

        if return_mid_feature:
            return x, *mid_feature

        return x


def visualize_attention_heatmap(tensor, title, filename, save_dir="./attn_visualizations", index=0):
    os.makedirs(save_dir, exist_ok=True)
    heatmap_data = tensor[index].detach().cpu().numpy()

    heatmap_data = np.where(heatmap_data == 0, np.nan, heatmap_data)

    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, cmap='viridis', mask=np.isnan(heatmap_data), cbar_kws={"label": "Attention Value"})
    plt.title(title)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    
    
    
