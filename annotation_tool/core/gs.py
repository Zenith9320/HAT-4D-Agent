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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.options_conf import Options

import kiui

try:
    from gsplat.rendering import rasterization
except (ImportError, OSError):
    rasterization = None


def _require_rasterization():  # noqa: ANN201
    if rasterization is None:
        raise RuntimeError(
            "gsplat is required for Gaussian rendering. Install the annotation-tool "
            "runtime dependencies before using GaussianRenderer.render()."
        )
    return rasterization

class GaussianRenderer:
    def __init__(self, opt, device):
        # self.opt.fovy , self.opt.output_size
        self.opt = opt
        self.device = device

        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)

        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        f = self.opt.output_size / (2 * self.tan_half_fov)
        self.K = torch.tensor([[f, 0., self.opt.output_size/2.], [0., f, self.opt.output_size/2.], [0., 0., 1.]], dtype=torch.float32, device=self.device)

    def render_points(self, points3D, cam_view, cam_view_proj, cam_pos, bg_color=None):
        # points3D: [B, N, 3] or [N, 3] (optionally [*, 6] with per-point RGB in [0, 1])
        device = points3D.device
        H = W = self.opt.output_size

        background = self.bg_color if bg_color is None else torch.tensor(bg_color, dtype=torch.float32, device=device)
        background = background.to(device).float()
        K = self.K.to(device)
        _ = cam_view_proj, cam_pos

        if points3D.dim() == 2:
            points3D = points3D.unsqueeze(0)
        assert points3D.dim() == 3, "points3D should be shaped as [B, N, 3] or [N, 3]"

        B = points3D.shape[0]
        V = cam_view.shape[1]

        images = []
        alphas = []

        for b in range(B):
            pts = points3D[b].float()
            coords = pts[:, :3]
            colors = pts[:, 3:6].clamp(0, 1) if pts.shape[1] >= 6 else torch.ones((coords.shape[0], 3), device=device)
            pts_h = torch.cat([coords, torch.ones((coords.shape[0], 1), device=device)], dim=1)

            for v in range(V):
                world_to_cam = cam_view[b, v].to(device=device, dtype=torch.float32).transpose(0, 1)
                pts_cam_h = pts_h @ world_to_cam
                pts_cam = pts_cam_h[:, :3]
                depth = pts_cam[:, 2]

                pos_front = depth > self.opt.znear
                neg_front = depth < -self.opt.znear
                use_negative = neg_front.sum() > pos_front.sum()
                depth_forward = -depth if use_negative else depth

                valid_depth = (depth_forward > self.opt.znear) & (depth_forward < self.opt.zfar)
                pts_cam = pts_cam[valid_depth]
                colors_valid = colors[valid_depth]
                depth_valid = depth_forward[valid_depth]

                if pts_cam.shape[0] == 0:
                    images.append(background.view(3, 1, 1).expand(3, H, W).clone())
                    alphas.append(torch.zeros((1, H, W), device=device))
                    continue

                pts_cam_proj = pts_cam.clone()
                pts_cam_proj[:, 2] = depth_valid
                proj = (pts_cam_proj @ K.T)
                proj_z = proj[:, 2].clamp(min=1e-6)
                u = proj[:, 0] / proj_z
                v_coord = proj[:, 1] / proj_z

                x_pix = torch.round(u).long()
                y_pix = torch.round(v_coord).long()
                in_bounds = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)

                if not torch.any(in_bounds):
                    images.append(background.view(3, 1, 1).expand(3, H, W).clone())
                    alphas.append(torch.zeros((1, H, W), device=device))
                    continue

                x_pix = x_pix[in_bounds]
                y_pix = y_pix[in_bounds]
                depth_valid = depth_valid[in_bounds]
                colors_valid = colors_valid[in_bounds]

                idx_flat = y_pix * W + x_pix
                flat_size = H * W

                # depth buffer via scatter-reduce; fall back to a CPU path if unavailable
                if hasattr(torch, "scatter_reduce"):
                    depth_init = torch.full((flat_size,), float("inf"), device=device)
                    depth_min = torch.scatter_reduce(depth_init, 0, idx_flat, depth_valid, reduce="amin", include_self=True)
                elif hasattr(torch.Tensor, "scatter_reduce_"):
                    depth_min = torch.full((flat_size,), float("inf"), device=device)
                    depth_min.scatter_reduce_(0, idx_flat, depth_valid, reduce="amin", include_self=True)
                else:
                    idx_np = idx_flat.detach().cpu().numpy()
                    depth_np = depth_valid.detach().cpu().numpy()
                    colors_np = colors_valid.detach().cpu().numpy()
                    order = np.lexsort((depth_np, idx_np))
                    idx_sorted = idx_np[order]
                    colors_sorted = colors_np[order]
                    keep = np.ones_like(idx_sorted, dtype=bool)
                    keep[1:] = idx_sorted[1:] != idx_sorted[:-1]
                    chosen_idx = idx_sorted[keep]
                    chosen_colors = colors_sorted[keep]
                    color_buffer_np = np.tile(background.detach().cpu().numpy().astype(np.float32), (flat_size, 1))
                    alpha_buffer_np = np.zeros((flat_size,), dtype=np.float32)
                    color_buffer_np[chosen_idx] = chosen_colors
                    alpha_buffer_np[chosen_idx] = 1.0
                    images.append(torch.from_numpy(color_buffer_np).to(device).transpose(0, 1).view(3, H, W))
                    alphas.append(torch.from_numpy(alpha_buffer_np).to(device).view(1, H, W))
                    continue

                depth_for_points = depth_min[idx_flat]
                min_mask = torch.isclose(depth_valid, depth_for_points, atol=1e-6)

                if not torch.any(min_mask):
                    images.append(background.view(3, 1, 1).expand(3, H, W).clone())
                    alphas.append(torch.zeros((1, H, W), device=device))
                    continue

                idx_min = idx_flat[min_mask]
                colors_min = colors_valid[min_mask]

                color_buffer = torch.zeros((flat_size, 3), device=device)
                weight_buffer = torch.zeros((flat_size,), device=device)
                color_buffer.index_add_(0, idx_min, colors_min)
                weight_buffer.index_add_(0, idx_min, torch.ones_like(idx_min, dtype=torch.float32))

                weight_buffer_clamped = torch.clamp(weight_buffer, min=1e-6).unsqueeze(1)
                color_buffer = color_buffer / weight_buffer_clamped
                color_buffer[weight_buffer == 0] = background

                alpha_flat = (weight_buffer > 0).float()

                images.append(color_buffer.transpose(0, 1).view(3, H, W))
                alphas.append(alpha_flat.view(1, H, W))

        images = torch.stack(images, dim=0).view(B, V, 3, H, W)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, H, W)

        return {
            "image": images,
            "alpha": alphas,
        }

    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            # render novel views
            view_matrix = cam_view[b].float()
            view_proj_matrix = cam_view_proj[b].float()
            campos = cam_pos[b].float()

            viewmat = view_matrix.transpose(2, 1) # [V, 4, 4]


            rendered_image_all, rendered_alpha_all, info = _require_rasterization()(
                means=means3D,
                quats=rotations,
                scales=scales,
                opacities=opacity.squeeze(-1),
                colors=rgbs,
                viewmats=viewmat,
                Ks=torch.stack([self.K for _ in range(V)]),
                width=self.opt.output_size,
                height=self.opt.output_size,
                near_plane=self.opt.znear,
                far_plane=self.opt.zfar,
                packed=False,
                backgrounds=torch.stack([self.bg_color for _ in range(V)]) if self.bg_color is not None else None,
                render_mode="RGB",
            )
            for rendered_image, rendered_alpha in zip(rendered_image_all, rendered_alpha_all):

                rendered_image = rendered_image.permute(2, 0, 1)
                rendered_image = rendered_image.clamp(0, 1)

                rendered_alpha = rendered_alpha.permute(2, 0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
