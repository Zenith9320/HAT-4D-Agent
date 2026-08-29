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
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import kiui
from kiui.lpips import LPIPS
from kiui.cam import orbit_camera

from ..core.options_conf import Options
from ..core.gs import GaussianRenderer
import os
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm 
import torchvision.transforms.functional as TF
import imageio


from ..utils.viz_utils import save_images_clip_as_video
from ..utils.viz_utils import logits_to_groups_with_capacity, get_group_map_from_flat, visualize_moga_groups_multiview

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        self.num_clip_frames = self.opt.num_clip_frames
        
        if self.opt.unet_version == "v0":
            print("[VERSION] UNet Memory Attention: v0 (baseline)")
            from core_free_fn.unet import UNet
            # unet
            self.unet = UNet(
                9, 14 * self.opt.gaussian_perpixel, 
                down_channels=self.opt.down_channels,
                down_attention=self.opt.down_attention,
                mid_attention=self.opt.mid_attention,
                up_channels=self.opt.up_channels,
                up_attention=self.opt.up_attention,
                num_views=self.opt.num_input_views,
                num_frames=self.opt.num_clip_frames,
                use_temp_attn=self.opt.use_temp_attn
            )
            self.use_moga_attn = False
            self.return_group_logit = False
            self.moga_group = self.opt.moga_group
            self.group_loss_lambda = 0
            

            
        elif self.opt.unet_version == "moga_v1":
            print("[VERSION] UNet Memory Attention: MoGA_v1")
            print("self.opt.moga_lambda:", self.opt.moga_lambda)
            from core_free_fn.moga import MoGA_UNet
            # unet
            self.unet = MoGA_UNet(
                9, 14 * self.opt.gaussian_perpixel, 
                down_channels=self.opt.down_channels,
                down_attention=self.opt.down_attention,
                mid_attention=self.opt.mid_attention,
                up_channels=self.opt.up_channels,
                up_attention=self.opt.up_attention,
                num_views=self.opt.num_input_views,
                num_frames=self.opt.num_clip_frames,
                use_temp_attn=self.opt.use_temp_attn,
                use_moga_attn = self.opt.use_moga_attn, 
                moga_lambda =self.opt.moga_lambda, 
                moga_group = self.opt.moga_group, 
                down_moga_attention = self.opt.down_moga_attention,
                mid_moga_attention = self.opt.mid_moga_attention,
                up_moga_attention = self.opt.up_moga_attention
            )
            self.return_group_logit = True
            self.use_moga_attn = True
            self.moga_group = self.opt.moga_group
            self.group_loss_lambda = self.opt.group_loss_lambda
            

        # last conv
        self.conv = nn.Conv2d(14 * self.opt.gaussian_perpixel, 14 * self.opt.gaussian_perpixel, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / tan_half_fov
        self.proj_matrix[1, 1] = 1 / tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1
        

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, azimuth=[0, 90, 180, -90], elevation=[0, 0, 0, 0]):
        
        from kiui.cam import orbit_camera
        from core_free_fn.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation[0], azimuth[0], radius=self.opt.cam_radius),
            orbit_camera(elevation[1], azimuth[1], radius=self.opt.cam_radius),
            orbit_camera(elevation[2], azimuth[2], radius=self.opt.cam_radius),
            orbit_camera(elevation[3], azimuth[3], radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def evaluate_input_image(self, images, recur_clip = True, input_azis = [0, 90, 180, 270], viz_group_logit = False, name = None, num_frames = None, num_clip_frames=None):
        print("images:", images.shape)
        results = {}
        B, TV, C, H, W = images.shape
        T = self.opt.num_frames if num_frames is None else num_frames
        CN =  self.opt.num_clip_frames if num_clip_frames is None else num_clip_frames
        bg_color = torch.ones(3, dtype=torch.float32, device=images.device)
 
        gaussians = []
        input_other_views_azis = input_azis[1:]
        

        
        if not recur_clip:
            CLIP_NUMBER = int(T / CN)
            for clip_id in range(CLIP_NUMBER):
                images_clip = images[:, (clip_id) * 4 * CN : (clip_id+1) * 4 * CN , : , : , : ]
                
                viz_images_clip =  images_clip[:,:,:3,:,:]
                
                gaussians_clip =  self.forward_gaussians(images_clip) # [B * T, N, 15]
                gaussians.append(gaussians_clip.reshape(B, CN, -1, 14))
        else:
            self.proj_matrix = self.proj_matrix.to(images.device)
            CLIP_NUMBER = math.ceil((T-1) / (CN-1))
            print("CLIP_NUMBER:", CLIP_NUMBER)
            
            cam_view_ls = []
            cam_pos_ls = []
            cam_view_proj_ls = []
            
            for v, azi in enumerate(input_other_views_azis ):
    
                cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to(images.device)
                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]
                
                cam_pos_ls.append(cam_pos.unsqueeze(0))
                cam_view_proj_ls.append(cam_view_proj.unsqueeze(0))
                cam_view_ls.append(cam_view.unsqueeze(0))
                
       
            
            
            for clip_id in range(CLIP_NUMBER):
                
                images_clip_after =  images[:, (clip_id) * 4 * (CN-1) + 4 : (clip_id+1) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, -1, self.opt.num_input_views, *images.shape[-3:])
                print(f"{clip_id} clip,  images_clip_after:", images_clip_after.shape)
                if images_clip_after.shape[1] < CN-1:
                    curr_clip_num = images_clip_after.shape[1]
                    repeat_len = CN - 1 - curr_clip_num
                    images_final_repeat = images_clip_after[:, -1, :,:,:,:].repeat(1, repeat_len, 1,1,1,1)
                    images_clip_after = torch.concat([images_clip_after, images_final_repeat], dim = 1).reshape(B, CN-1, self.opt.num_input_views, *images.shape[-3:])
                else:
                    curr_clip_num = CN
                
                images_clip_first_frame =  images[:, (clip_id) * 4 * (CN-1) : (clip_id) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
                if clip_id == 0:
                    images_clip_first_frame = images[:, 0 : 4 , : , : , : ].reshape(B, 1,-1, *images.shape[-3:])
                else:

                    first_frame_gs = gaussians_clip.reshape(B, CN, -1, 14)[:,-1, :,: ].reshape(B, -1, gaussians_clip.shape[-1])
                    cam_view_input_first_frame = torch.concat(cam_view_ls, dim = 0)
                    cam_view_proj_input_first_frame = torch.concat(cam_view_proj_ls, dim=0) 
                    cam_pos_input_first_frame = torch.concat(cam_pos_ls , dim = 0)
                

                    images_clip_first_frame_other_views = self.gs.render(first_frame_gs.repeat(self.opt.num_input_views - 1,1,1), cam_view_input_first_frame, cam_view_proj_input_first_frame, cam_pos_input_first_frame, bg_color=bg_color)["image"]
                    images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(-1, *images_clip_first_frame_other_views.shape[-3:] )
                    images_clip_first_frame_other_views = TF.normalize(images_clip_first_frame_other_views, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                    images_clip_first_frame_other_views = F.interpolate(
                        images_clip_first_frame_other_views, size=(H, W), mode='bilinear', align_corners=False
                    )
                    images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(B,1, self.opt.num_input_views - 1,  *images_clip_first_frame_other_views.shape[-3:] )
                    
                    images_clip_first_frame = images_clip_first_frame.reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
                    images_clip_first_frame[:,:, 1:, :3, : , :] = images_clip_first_frame_other_views.reshape(B, 1, self.opt.num_input_views - 1, *images_clip_first_frame_other_views.shape[-3:])
                    
                images_clip = torch.concatenate( [ images_clip_first_frame, images_clip_after], dim = 1)
                images_clip[:,1:,1:] = images_clip[:,0:1,1:]
                images_clip = images_clip.reshape(B, CN* 4,*images.shape[-3:])
                
                viz_images_clip =  images_clip[:,:,:3,:,:].float()
                if not  viz_group_logit:
                    gaussians_clip =  self.forward_gaussians(images_clip) # [B * T, N, 15]
                else:
                    gaussians_clip, output =  self.forward_gaussians(images_clip,return_group_logit = viz_group_logit ) # [B * T, N, 15]
                    print("viz_group_logit:", len(output["logits"]))
                    print("viz_group_logit:", (output["logits"][0].shape))
                    print("image_clip:", images_clip.shape)
                    print(f"from {clip_id * (CN-1)} to { (clip_id+1) * (CN-1) }")
                    save_img_folder = "./debug/viz_group"
                    
                    
                    for layer_id in range(len(output["logits"])):
                        save_img_path = os.path.join(save_img_folder, f"f_{clip_id * (CN-1)}_t_{(clip_id+1) * (CN-1)}_layer_{layer_id}.png")
                        group_map =  logits_to_groups_with_capacity(output["logits"][layer_id])
                        token_L = group_map.shape[0]
                        sub_H = int(math.sqrt( token_L // CN // self.opt.num_input_views))
                        final_group_map = get_group_map_from_flat(CN, self.opt.num_input_views, group_map, B,  sub_H, sub_H)
                        visualize_moga_groups_multiview(final_group_map, (CN * B *self.opt.num_input_views, 1, H, W),CN  , self.opt.num_input_views , sample_index = 0, save_path = save_img_path)
                
                if clip_id == 0:
                    start_id = 0
                else:
                    start_id = 1
                gaussians.append(gaussians_clip.reshape(B, CN, -1, 14)[:, start_id: start_id + curr_clip_num, :, :])
                
                        
                    
        gaussians = torch.concat(gaussians, dim=1)
    
        # use the first view to predict gaussians

        gaussians = gaussians.reshape(B * T, -1, 14) # [B * T, N, 14]


        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        return gaussians
    
    
    
    def evaluate(self, data, num_frames, num_clip_frames):
        # images: [B, T, 4, 9, H, W]
        assert num_clip_frames == self.num_clip_frames
        
        images = data['input'] # [B, Tx4, 9, h, W], input features

        results = {}
        B, TV, C, H, W = images.shape
        T =  num_frames
        CN =  num_clip_frames
        gaussians = []
        results['images_pred'] = []
            
        cam_view_input = data["cam_view_input"].reshape(B,T, self.opt.num_input_views,  *data["cam_view_input"].shape[2:])
        cam_view_proj_input = data['cam_view_proj_input'].reshape(B,T, self.opt.num_input_views, *data['cam_view_proj_input'].shape[2:])
        cam_pos_input = data['cam_pos_input'].reshape(B,T, self.opt.num_input_views,  *data['cam_pos_input'].shape[2:])
        
        
        self.proj_matrix = self.proj_matrix.to(images.device)
        CLIP_NUMBER = math.ceil((T-1) / (CN-1))
        
        for clip_id in range(CLIP_NUMBER):
            clip_start_id = clip_id * (CN - 1)
            clip_end_id = (clip_id+1) * (CN - 1) + 1
            images_clip_after =  images[:, (clip_id) * 4 * (CN-1) + 4 : (clip_id+1) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, -1, self.opt.num_input_views, *images.shape[-3:])
            
            if images_clip_after.shape[1] < CN - 1:
                repeat_len = CN-1-images_clip_after.shape[1]
                repeat_final_images_clip = images_clip_after[:,-1:,:,:,:,:].repeat(1, repeat_len  ,1,1,1,1)
                images_clip_after = torch.concatenate([images_clip_after, repeat_final_images_clip], dim=1)
            else:
                repeat_len = 0
            assert images_clip_after.shape[1] == CN - 1 
            
            images_clip_first_frame =  images[:, (clip_id) * 4 * (CN-1) : (clip_id) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
            if clip_id == 0:
                images_clip_first_frame = images[:, 0 : 4 , : , : , : ].reshape(B, 1,-1, *images.shape[-3:])
            else:
                first_frame_gs = gaussians_clip.reshape(B,CN, -1 , gaussians_clip.shape[-1] )[:,-1, :,: ].reshape(B, -1, gaussians_clip.shape[-1]).detach().contiguous()
                cam_view_input_first_frame = cam_view_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_view_input.shape[3:]).unsqueeze(0)
                cam_view_proj_input_first_frame = cam_view_proj_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_view_proj_input.shape[3:]).unsqueeze(0)
                cam_pos_input_first_frame = cam_pos_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_pos_input.shape[3:]).unsqueeze(0)
                
                # always use white bg
                bg_color = torch.ones(3, dtype=torch.float32, device=first_frame_gs.device)
           
    
                images_clip_first_frame_other_views = self.gs.render(first_frame_gs, cam_view_input_first_frame, cam_view_proj_input_first_frame, cam_pos_input_first_frame, bg_color=bg_color)["image"]
                images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(-1, *images_clip_first_frame_other_views.shape[-3:] )
                images_clip_first_frame_other_views = TF.normalize(images_clip_first_frame_other_views, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                images_clip_first_frame_other_views = F.interpolate(
                    images_clip_first_frame_other_views, size=(H, W), mode='bilinear', align_corners=False
                )
                images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(B,1, self.opt.num_input_views - 1,  *images_clip_first_frame_other_views.shape[-3:] )
                
                
                images_clip_first_frame = images_clip_first_frame.reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
                images_clip_first_frame[:,:, 1:, :3, : , :] = images_clip_first_frame_other_views.reshape(B, 1, self.opt.num_input_views - 1, *images_clip_first_frame_other_views.shape[-3:])
                
            images_clip = torch.concatenate( [ images_clip_first_frame, images_clip_after], dim = 1)
            images_clip[:,1:,1:] = images_clip[:,0:1,1:]
            images_clip = images_clip.reshape(B, CN* 4,*images.shape[-3:])
            

            gaussians_clip =  self.forward_gaussians(images_clip) # [B * T, N, 15]
            curr_clip_num = CN-repeat_len
            
            if clip_id == 0:
                start_id = 0
            else:
                start_id = 1
            gaussians.append(gaussians_clip.reshape(B, CN, -1, 14)[:, start_id: start_id + curr_clip_num, :, :])

                 
        gaussians = torch.concat(gaussians, dim=1)
        
    
        # use the first view to predict gaussians

        gaussians = gaussians.reshape(B * T, -1, 14) # [B * T, N, 14]
        
        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
 
        
        # use the other views for rendering and supervision
        data['cam_view'] = data['cam_view'].reshape(B*T, -1, *data['cam_view'].shape[2:])
        data['cam_view_proj'] = data['cam_view_proj'].reshape(B*T, -1, *data['cam_view_proj'].shape[2:])
        data['cam_pos'] = data['cam_pos'].reshape(B*T, -1, *data['cam_pos'].shape[2:])
    
        
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B*T, V, C, output_size, output_size]
    
        pred_alphas = results['alpha'] # [B*T, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
  
        data['images_output'] = data['images_output'].reshape(B*T, -1, *data['images_output'].shape[2:])
        data['masks_output'] = data['masks_output'].reshape(B*T, -1, *data['masks_output'].shape[2:])


        gt_images = data['images_output'] # [B*T, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B*T, V, 1, output_size, output_size], ground-truth masks


        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        

            
        # Sanity check before LPIPS
        if torch.isnan(pred_images).any() or torch.isinf(pred_images).any():
            print("⚠️ pred_images contains NaN or Inf!")
        if torch.isnan(gt_images).any() or torch.isinf(gt_images).any():
            print("⚠️ gt_images contains NaN or Inf!")
            
        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        

        return results
    

                                
    

        

    def forward_gaussians(self, images, return_group_logit = False, return_group_loss = False, num_clip_frames = None ):
        # images: [B, T, 4, 9, H, W]
        B, TV, C, H, W = images.shape
        if num_clip_frames is None:
            T = self.num_clip_frames
        else:
            T = num_clip_frames
        V = TV // T
        images = images.reshape(B*T*V, C, H, W)
        
        output = {}
        print("forward images:", images.shape)

        if self.use_moga_attn:
            res =   self.unet(images, return_group_logit = return_group_logit, return_group_loss = return_group_loss, num_frames = num_clip_frames ) # [B*4, 14, h, w]
            if isinstance(res, dict):
                x = res["output"]
            else:
                x = res
            if return_group_logit:
                output["logits"] = res["logits"] 
                
            if return_group_loss:
                output["loss"] =  res["loss"] 
        else:
            x = self.unet(images, num_frames= num_clip_frames ) # [B*4, 14, h, w]
            
        
        
        
            
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B*T, V, 14 * self.opt.gaussian_perpixel, self.opt.splat_size, self.opt.splat_size)

        x = x.permute(0, 1, 3, 4, 2).reshape(B*T, -1, 14).contiguous()
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, T, N, 14]
        
        if return_group_logit or return_group_loss:
            return  gaussians, output
        
        return gaussians
    
    def forward_clip(
        self,
        data,
        step_ratio=1,
        num_frames=None,
        clip_id: int = 0,
        recur_clip: bool = None,
        return_group_logit: bool = False,
        vis_img: bool = False
    ):
        
        import torch.distributed as dist
        



        self.num_frames = num_frames if num_frames is not None else self.opt.num_frames
        assert self.num_frames > self.num_clip_frames

        self.recur_clip = recur_clip if recur_clip is not None else self.opt.recur_clip

        images = data['input']  # [B, Tx4, 9, H, W]
        B, TV, C, H, W = images.shape
        T = self.num_frames
        CN = self.num_clip_frames

        # reshape
        cam_view_input = data["cam_view_input"].reshape(
            B, T, self.opt.num_input_views, *data["cam_view_input"].shape[2:]
        )
        cam_view_proj_input = data["cam_view_proj_input"].reshape(
            B, T, self.opt.num_input_views, *data["cam_view_proj_input"].shape[2:]
        )
        cam_pos_input = data["cam_pos_input"].reshape(
            B, T, self.opt.num_input_views, *data["cam_pos_input"].shape[2:]
        )

        all_images_output = data['images_output'].reshape(
            B, T, -1, *data['images_output'].shape[2:]
        )
        all_masks_output = data['masks_output'].reshape(
            B, T, -1, *data['masks_output'].shape[2:]
        )

        all_cam_view = data['cam_view'].reshape(B, T, -1, *data['cam_view'].shape[2:])
        all_cam_view_proj = data['cam_view_proj'].reshape(
            B, T, -1, *data['cam_view_proj'].shape[2:]
        )
        all_cam_pos = data['cam_pos'].reshape(
            B, T, -1, *data['cam_pos'].shape[2:]
        )

        if not self.recur_clip:
            clip_start_id = clip_id * CN
            clip_end_id = (clip_id + 1) * CN
            curr_clip_num = CN

            images_clip = images[:, clip_id * 4 * CN : (clip_id+1) * 4 * CN]

            if self.use_moga_attn:
  
                gaussians_clip, res = self.forward_gaussians(
                    images_clip, return_group_logit=return_group_logit,  return_group_loss = True
                )
            else:
                gaussians_clip = self.forward_gaussians(images_clip)
                group_logit = None

            start_id = 0

        else:
            raise NotImplementedError(
                "forward_clip does not support recur_clip=True; "
                "use forward_recur for recurrent clip inference"
            )
            
        if vis_img:
            import os
            from torchvision.utils import save_image

            with torch.no_grad():
                vis_dir =  "./tmp/viz_pred_train_memory4d_v2"
                B_in, T4, C_in, H, W = images_clip.shape
                num_views = self.opt.num_input_views
                assert T4 == CN * num_views

                images_vis = images_clip[:, :, :3].detach().clamp(0, 1).cpu()

                for b in range(B_in):
                    
                    if "case_name" in data:
                        case_name = data["case_name"][b]
                    else:
                        case_name = data["image_folder"][b].split("/")[1]
                    input_vis_dir = os.path.join(vis_dir, case_name, "input")
                    os.makedirs(input_vis_dir, exist_ok=True)

                    for local_fid in range(CN):
                        global_frame_id = clip_start_id + local_fid

                        for view_id in range(num_views):
                            idx = local_fid * num_views + view_id
                            img = images_vis[b, idx]

                            save_path = os.path.join(
                                input_vis_dir,
                                f"input_clip_{clip_id}_f_{global_frame_id}_v_{view_id}.png"
                            )
                            save_image(img, save_path)
                            print("save:", save_path)
                            
                    

        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians_clip.device)

        clip_cam_view = all_cam_view[:, clip_start_id:clip_end_id].reshape(
            B * curr_clip_num, -1, *data['cam_view'].shape[2:]
        )
        clip_cam_view_proj = all_cam_view_proj[:, clip_start_id:clip_end_id].reshape(
            B * curr_clip_num, -1, *data['cam_view_proj'].shape[2:]
        )
        clip_cam_pos = all_cam_pos[:, clip_start_id:clip_end_id].reshape(
            B * curr_clip_num, -1, *data['cam_pos'].shape[2:]
        )

        results = self.gs.render(
            gaussians_clip, clip_cam_view, clip_cam_view_proj, clip_cam_pos, bg_color=bg_color
        )

        pred_images = results['image']     # [B*clip_num, V, 3, H, W]
        pred_alphas = results['alpha']     # [B*clip_num, V, 1, H, W]

        gt_images = all_images_output[:, clip_start_id:clip_end_id].reshape(
            B * curr_clip_num, -1, *data['images_output'].shape[2:]
        )
        gt_masks = all_masks_output[:, clip_start_id:clip_end_id].reshape(
            B * curr_clip_num, -1, *data['masks_output'].shape[2:]
        )

        gt_images = gt_images * gt_masks + bg_color.view(1,1,3,1,1)*(1-gt_masks)
        
        if vis_img:
            
            vis_dir = "./tmp/viz_pred_train_memory4d_v2"
            import os
            import torchvision
            from torchvision.utils import save_image

            os.makedirs(vis_dir, exist_ok=True)

            with torch.no_grad():
                # pred_images, gt_images: [B*clip_num, V, 3, H, W]
                Bc, V, _, H, W = pred_images.shape

                pred_vis = pred_images.detach().clamp(0, 1).cpu()
                gt_vis   = gt_images.detach().clamp(0, 1).cpu()

                for bc_id in range(Bc):
                    if "case_name" in data:
                        case_name = data["case_name"][b]
                    else:
                        case_name = data["image_folder"][b].split("/")[1]
                    for v_id in range(V):
                        pred_img = pred_vis[bc_id, v_id]  # [3, H, W]
                        gt_img   = gt_vis[bc_id, v_id]    # [3, H, W]

                        concat = torch.cat([pred_img, gt_img], dim=2)
                        os.makedirs(os.path.join( vis_dir,case_name ), exist_ok=True)

                        save_path = os.path.join(
                            vis_dir,case_name ,  f"bc_{clip_start_id + (bc_id % CN) }_v_{v_id}.png"
                        )

                        save_image(concat, save_path)

        # ================== Loss ======================
    
        loss_group = res["loss"]
        loss_recon = (
            F.mse_loss(pred_images, gt_images) +
            F.mse_loss(pred_alphas, gt_masks)
        )
        
        
        loss = loss_recon + self.group_loss_lambda * loss_group

        loss_lpips = torch.tensor(0.0, device=loss.device)
        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                F.interpolate(
                    gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size)*2 - 1,
                    (256,256), mode='bilinear', align_corners=False
                ),
                F.interpolate(
                    pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size)*2 - 1,
                    (256,256), mode='bilinear', align_corners=False
                )
            ).mean()
            loss += self.opt.lambda_lpips * loss_lpips

        # =============== psnr ===================
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images)**2))

        out_clip = {
            "images_pred": pred_images.detach(),
            "alphas_pred": pred_alphas.detach(),
            "conf_pred": None,
            "loss": loss.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_lpips": loss_lpips.detach(),
            "loss_group": loss_group.detach(),
            "psnr": psnr.detach(),
        }

        return loss, psnr, out_clip



    def forward(self, data, mode="default",**kwargs):
        
        if mode == "default":
            self.num_frames = num_frames  if num_frames != None else self.opt.num_frames
            assert self.num_frames <= self.num_clip_frames
            return self.forward_no_recur(data, **kwargs)
        
        elif mode == "test":
            return self.forward_recur_test(data,  **kwargs)
        elif mode == "clip":
            return self.forward_clip(data=data, **kwargs)
        elif mode == "eval":
            return self.evaluate(data=data, **kwargs)
        else:
            raise ValueError
    
    def forward_recur_test(self, data,  num_frames = None, recur_clip= None,return_group_logit = False):
        
        self.num_frames = num_frames
        assert self.num_frames > self.num_clip_frames
        torch.cuda.empty_cache()

        with torch.set_grad_enabled(False):

            self.recur_clip = recur_clip if recur_clip else self.opt.recur_clip
            
            results_all = {}
            results_all['images_pred'] = []
            results_all['alphas_pred']= []

            
            images = data['input'] # [B, Tx4, 9, h, W], input features
            B, TV, C, H, W = images.shape
            T =  num_frames if num_frames else  self.num_frames
            CN =  self.num_clip_frames
            CLIP_NUMBER = int(T / CN)
        
            print("in Test data['cam_view_input']:", data["cam_view_input"].shape)
            cam_view_input = data["cam_view_input"].reshape(B,T, self.opt.num_input_views,  *data["cam_view_input"].shape[2:])
            cam_view_proj_input = data['cam_view_proj_input'].reshape(B,T, self.opt.num_input_views, *data['cam_view_proj_input'].shape[2:])
            cam_pos_input = data['cam_pos_input'].reshape(B,T, self.opt.num_input_views,  *data['cam_pos_input'].shape[2:])

            
            all_images_output = data['images_output'].reshape(B,T, -1, *data['images_output'].shape[2:])
            all_masks_output = data['masks_output'].reshape(B,T, -1, *data['masks_output'].shape[2:])

            all_cam_view = data['cam_view'].reshape(B, T, -1, *data['cam_view'].shape[2:])
            all_cam_view_proj = data['cam_view_proj'].reshape(B, T, -1, *data['cam_view_proj'].shape[2:])
            all_cam_pos = data['cam_pos'].reshape(B, T , -1, *data['cam_pos'].shape[2:])
            
            data_balance_loss = 0
            data_loss = 0
            data_psnr = 0
            data_recon_loss = 0
            data_lpips = 0
        
            if not recur_clip:
                CLIP_NUMBER = int(T / CN)
            else:
                CLIP_NUMBER = math.ceil((T-1) / (CN-1))
                
            for clip_id in range(CLIP_NUMBER):
                                
                if not recur_clip:
                    clip_start_id = clip_id * CN
                    clip_end_id = (clip_id  + 1)* CN
                    curr_clip_num = CN
                    images_clip = images[:, (clip_id) * 4 * CN : (clip_id+1) * 4 * CN , : , : , : ]
                    viz_images_clip =  images_clip[:,:,:3,:,:]
            
                    if self.use_moga_attn:
                        gaussians_clip, res =  self.forward_gaussians(images_clip, return_group_logit = return_group_logit, return_group_loss = True) # [B * CN, N, 15]
                    else:
                        gaussians_clip =  self.forward_gaussians(images_clip, return_group_logit = False, return_group_loss =  False) # [B * CN, N, 15]
                        group_logit = None
                    start_id = 0
                else:
                    clip_start_id = clip_id * (CN - 1)
                    clip_end_id = (clip_id+1) * (CN - 1) + 1
                    images_clip_after =  images[:, (clip_id) * 4 * (CN-1) + 4 : (clip_id+1) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, -1, self.opt.num_input_views, *images.shape[-3:])
                    
                    
                    if images_clip_after.shape[1] < CN - 1:
                        repeat_len = CN-1-images_clip_after.shape[1]
                        repeat_final_images_clip = images_clip_after[:,-1:,:,:,:,:].repeat(1, repeat_len  ,1,1,1,1)
                        images_clip_after = torch.concatenate([images_clip_after, repeat_final_images_clip], dim=1)
                    else:
                        repeat_len = 0
                    assert images_clip_after.shape[1] == CN - 1 
                    
                    images_clip_first_frame =  images[:, (clip_id) * 4 * (CN-1) : (clip_id) * 4 * (CN-1) + 4 , : , : , : ].reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
                    if clip_id == 0:
                        images_clip_first_frame = images[:, 0 : 4 , : , : , : ].reshape(B, 1,-1, *images.shape[-3:])
                    else:
                        first_frame_gs = gaussians_clip.reshape(B,CN, -1 , gaussians_clip.shape[-1] )[:,-1, :,: ].reshape(B, -1, gaussians_clip.shape[-1]).detach().contiguous()
                        cam_view_input_first_frame = cam_view_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_view_input.shape[3:]).unsqueeze(0)
                        cam_view_proj_input_first_frame = cam_view_proj_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_view_proj_input.shape[3:]).unsqueeze(0)
                        cam_pos_input_first_frame = cam_pos_input[:, clip_id *(CN - 1)  , 1:].reshape(-1, *cam_pos_input.shape[3:]).unsqueeze(0)
                        
                        images_clip_first_frame_other_views = self.gs.render(first_frame_gs, cam_view_input_first_frame, cam_view_proj_input_first_frame, cam_pos_input_first_frame, bg_color=bg_color)["image"]
                        images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(-1, *images_clip_first_frame_other_views.shape[-3:] )
                        images_clip_first_frame_other_views = TF.normalize(images_clip_first_frame_other_views, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                        images_clip_first_frame_other_views = F.interpolate(
                            images_clip_first_frame_other_views, size=(H, W), mode='bilinear', align_corners=False
                        )
                        images_clip_first_frame_other_views = images_clip_first_frame_other_views.reshape(B,1, self.opt.num_input_views - 1,  *images_clip_first_frame_other_views.shape[-3:] )
                        
                        
                        images_clip_first_frame = images_clip_first_frame.reshape(B, 1, self.opt.num_input_views, *images.shape[-3:])
                        images_clip_first_frame[:,:, 1:, :3, : , :] = images_clip_first_frame_other_views.reshape(B, 1, self.opt.num_input_views - 1, *images_clip_first_frame_other_views.shape[-3:])
                        
                    images_clip = torch.concatenate( [ images_clip_first_frame, images_clip_after], dim = 1)
                    images_clip[:,1:,1:] = images_clip[:,0:1,1:]
                    images_clip = images_clip.reshape(B, CN* 4,*images.shape[-3:])
                    
                    viz_images_clip =  images_clip[:,:,:3,:,:].float()
                    
                    if self.use_moga_attn:
                        gaussians_clip, res =  self.forward_gaussians(images_clip, return_group_logit = return_group_logit, return_group_loss = True) # [B * CN, N, 15]
                    else:
                        gaussians_clip =  self.forward_gaussians(images_clip, return_group_logit = False, return_group_loss =  False) # [B * CN, N, 15
                        group_logit = None
                    gaussians_clip = gaussians_clip[ :B*(CN-repeat_len), :,:]
                    curr_clip_num = CN-repeat_len
                    

                    if clip_id == 0:
                        start_id = 0
                    else:
                        start_id = 1

                
                bg_color = torch.ones(3, dtype=torch.float32, device=gaussians_clip.device)
                
                clip_cam_view = all_cam_view[:,clip_start_id : clip_end_id  , : ,  : , :  ].reshape(B* curr_clip_num , -1,*data['cam_view'].shape[2:] )
                clip_cam_view_proj = all_cam_view_proj[:,  clip_start_id : clip_end_id , : ,  : , : ].reshape(B* curr_clip_num, -1,*data['cam_view_proj'].shape[2:] )
                clip_cam_pos = all_cam_pos[:,  clip_start_id : clip_end_id , : ,  :].reshape(B* curr_clip_num, -1,*data['cam_pos'].shape[2:] )
                
                results = self.gs.render(gaussians_clip, clip_cam_view, clip_cam_view_proj, clip_cam_pos, bg_color=bg_color)
                pred_images = results['image']
                pred_alphas = results['alpha']
        
                results_all['images_pred'].append(pred_images.reshape(B,curr_clip_num, -1,  *data['images_output'].shape[2:])[: , start_id:, :,:,:,:]   )
                results_all['alphas_pred'].append(pred_alphas.reshape(B,curr_clip_num, -1,  *data['masks_output'].shape[2:])[: , start_id:, :,:,:,:]  )
                

                gt_images =  all_images_output[:,  clip_start_id : clip_end_id , :, : , : ].reshape(B* curr_clip_num, -1, *data['images_output'].shape[2:])  
                gt_masks =  all_masks_output[:,  clip_start_id : clip_end_id  , :, : , : ].reshape(B* curr_clip_num, -1, *data['masks_output'].shape[2:])  
                gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
                loss_group = res["loss"]
                loss_mse =  F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks) 
                loss =  loss_mse + self.group_loss_lambda * loss_group 
                
                # Sanity check before LPIPS
                if torch.isnan(pred_images).any() or torch.isinf(pred_images).any():
                    print("⚠️ pred_images contains NaN or Inf!")
                if torch.isnan(gt_images).any() or torch.isinf(gt_images).any():
                    print("⚠️ gt_images contains NaN or Inf!")

                if self.opt.lambda_lpips > 0:
                    loss_lpips = self.lpips_loss(
                        F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                        F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    ).mean()
                    data_lpips  += loss_lpips
                    loss = loss + self.opt.lambda_lpips * loss_lpips
                    
                results['loss'] = loss
        
                with torch.no_grad():
                    psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
                    results['psnr'] = psnr
             
                data_recon_loss += loss_mse.item()
                data_loss += loss.item()
                data_psnr += psnr.item()
                data_balance_loss += loss_group.item()
            
            
            
            results_all["group_balance_loss"] = torch.tensor(data_balance_loss/ CLIP_NUMBER) 
            results_all["psnr"] = torch.tensor(data_psnr / CLIP_NUMBER)
            results_all["loss"] = torch.tensor(data_loss / CLIP_NUMBER) 
            results_all["loss_recon"] = torch.tensor(data_recon_loss / CLIP_NUMBER)
            results_all["loss_lpips"] = (data_lpips / CLIP_NUMBER)
            
            
            results_all['images_pred'] = torch.concat(results_all['images_pred'], dim = 1).reshape(B*T, -1,  *data['images_output'].shape[2:])
            results_all['alphas_pred'] = torch.concat(results_all['alphas_pred'], dim = 1).reshape(B*T, -1,  *data['masks_output'].shape[2:])

            return  results_all

    def forward_no_recur(self, data, step_ratio=1, return_group_logit = False):
        # data: output of the dataloader
        
        assert self.num_frames <= self.num_clip_frames

        results = {}
        loss = 0

        images = data['input'] # [B, Tx4, 9, h, W], input features

        B, TV, C, H, W = images.shape
        T = self.opt.num_frames
        
        # use the first view to predict gaussians
        if self.use_moga_attn:
            gaussians, res  = self.forward_gaussians(images, return_group_logit=return_group_logit,  return_group_loss = True)
            loss += res["loss"]
        else:
            gaussians = self.forward_gaussians(images) # [B * T, N, 14]

        results['gaussians'] = gaussians

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        data['cam_view'] = data['cam_view'].reshape(B*T, -1, *data['cam_view'].shape[2:])
        data['cam_view_proj'] = data['cam_view_proj'].reshape(B*T, -1, *data['cam_view_proj'].shape[2:])
        data['cam_pos'] = data['cam_pos'].reshape(B*T, -1, *data['cam_pos'].shape[2:])
        
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B*T, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B*T, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas


        data['images_output'] = data['images_output'].reshape(B*T, -1, *data['images_output'].shape[2:])
        data['masks_output'] = data['masks_output'].reshape(B*T, -1, *data['masks_output'].shape[2:])

        gt_images = data['images_output'] # [B*T, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B*T, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
