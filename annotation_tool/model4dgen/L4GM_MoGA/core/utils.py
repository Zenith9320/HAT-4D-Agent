import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import roma
from kiui.op import safe_normalize

def flip_pitch_towards_origin_stable(cam_poses):
    pos = cam_poses[:, :3, 3]
    R_old = cam_poses[:, :3, :3]

    # 球坐标
    x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:3]
    r = torch.norm(pos, dim=-1, keepdim=True)
    theta = torch.atan2(z, x)
    phi = torch.atan2(y, torch.sqrt(x**2 + z**2))
    phi = -phi  # 翻转俯仰

    # 新位置
    x_new = r * torch.cos(phi) * torch.cos(theta)
    y_new = r * torch.sin(phi)
    z_new = r * torch.cos(phi) * torch.sin(theta)
    pos_new = torch.cat([x_new, y_new, z_new], dim=-1)

    # forward 指向原点
    forward = -pos_new / torch.norm(pos_new, dim=-1, keepdim=True)

    # 用固定 world up 向量，处理极端俯仰
    world_up = torch.tensor([0,1,0], device=pos.device, dtype=pos.dtype).expand_as(forward)
    # 检查 forward 和 world_up 共线情况
    dot = (forward * world_up).sum(dim=-1, keepdim=True)
    mask = torch.abs(dot) > 0.999  # 接近共线

    # 对接近共线的 forward，用临时向量修正 up
    up = world_up.clone()
    up[mask.expand_as(up)] = torch.tensor([0,0,1], device=pos.device, dtype=pos.dtype)

    right = torch.linalg.cross(up, forward)
    right = right / torch.norm(right, dim=-1, keepdim=True)
    up = torch.linalg.cross(forward, right)

    R_new = torch.stack([right, up, forward], dim=-1)

    cam_poses_new = cam_poses.clone()
    cam_poses_new[:, :3, :3] = R_new
    cam_poses_new[:, :3, 3] = pos_new
    return cam_poses_new


def mirror_cam_xz(cam_poses: torch.Tensor) -> torch.Tensor:
    """
    对 cam_poses 在 x-z 平面做对称，保持相机朝向原点，
    并保证相机右方向和原始相机一致。
    
    Args:
        cam_poses: [B, 4, 4] 相机到世界变换矩阵
    
    Returns:
        mirrored_cam_poses: [B, 4, 4]
    """
    B = cam_poses.shape[0]
    
    cam_pos = cam_poses[:, :3, 3]  # [B, 3]
    cam_right = cam_poses[:, :3, 0]  # 原相机右方向
    cam_up = cam_poses[:, :3, 1]     # 原相机上方向
    
    # 对称位置
    mirrored_pos = cam_pos.clone()
    mirrored_pos[:, 1] *= -1  # y 轴取反
    
    # 重新计算朝向向量（指向原点）
    forward = -mirrored_pos  # look_at target=origin
    forward = forward / forward.norm(dim=-1, keepdim=True)
    
    # 保持右方向不变
    right = cam_right / cam_right.norm(dim=-1, keepdim=True)
    
    # 重新计算上方向
    up = torch.cross(forward, right, dim=-1)
    up = up / up.norm(dim=-1, keepdim=True)
    
    # 再次正交化 right
    right = torch.cross(up, forward, dim=-1)
    
    # 构建 4x4 矩阵
    mirrored_cam_poses = torch.eye(4, device=cam_poses.device).unsqueeze(0).repeat(B, 1, 1)
    mirrored_cam_poses[:, :3, 0] = right
    mirrored_cam_poses[:, :3, 1] = up
    mirrored_cam_poses[:, :3, 2] = forward
    mirrored_cam_poses[:, :3, 3] = mirrored_pos
    
    return mirrored_cam_poses


def flip_pitch_towards_origin(cam_poses):
    """
    cam_poses: [N,4,4] camera-to-world
    翻转相机围绕原点连线的俯仰角，同时保持朝向原点。
    """
    pos = cam_poses[:, :3, 3]  # [N,3]
    r = torch.norm(pos, dim=-1, keepdim=True)  # 半径
    
    x, y, z = pos[:,0:1], pos[:,1:2], pos[:,2:3]
    # 球坐标：theta = azimuth, phi = elevation
    theta = torch.atan2(z, x)
    phi = torch.atan2(y, torch.sqrt(x**2 + z**2))
    
    # 翻转俯仰
    phi = -phi
    
    # 转回笛卡尔
    x_new = r * torch.cos(phi) * torch.cos(theta)
    y_new = r * torch.sin(phi)
    z_new = r * torch.cos(phi) * torch.sin(theta)
    pos_new = torch.cat([x_new, y_new, z_new], dim=-1)
    
    # 生成新的旋转矩阵，让相机朝向原点
    forward = -pos_new / torch.norm(pos_new, dim=-1, keepdim=True)
    tmp_up = torch.tensor([0,1,0], device=cam_poses.device, dtype=cam_poses.dtype).expand_as(forward)
    right = torch.linalg.cross(tmp_up, forward)
    right = right / torch.norm(right, dim=-1, keepdim=True)
    up = torch.cross(forward, right)
    
    R = torch.linalg.cross([right, up, forward], dim=-1)  # [N,3,3]
    
    cam_poses_new = cam_poses.clone()
    cam_poses_new[:, :3, :3] = R
    cam_poses_new[:, :3, 3] = pos_new
    return cam_poses_new




def matrix_to_euler_xyz(R):
    # 输入: R [..., 3, 3]
    # 输出: [..., 3] -> [roll(x), pitch(y), yaw(z)]
    sy = torch.sqrt(R[...,0,0]**2 + R[...,1,0]**2)
    singular = sy < 1e-6

    x = torch.where(singular,
                    torch.atan2(-R[...,1,2], R[...,1,1]),
                    torch.atan2(R[...,2,1], R[...,2,2]))
    y = torch.where(singular,
                    torch.atan2(-R[...,2,0], sy),
                    torch.atan2(-R[...,2,0], sy))
    z = torch.where(singular,
                    torch.zeros_like(x),
                    torch.atan2(R[...,1,0], R[...,0,0]))
    return torch.stack([x, y, z], dim=-1)

def euler_xyz_to_matrix(euler):
    x, y, z = euler[...,0], euler[...,1], euler[...,2]
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    R = torch.zeros((*euler.shape[:-1], 3, 3), device=euler.device, dtype=euler.dtype)
    R[...,0,0] = cy * cz
    R[...,0,1] = cz * sx * sy - cx * sz
    R[...,0,2] = sx * sz + cx * cz * sy
    R[...,1,0] = cy * sz
    R[...,1,1] = cx * cz + sx * sy * sz
    R[...,1,2] = cx * sy * sz - cz * sx
    R[...,2,0] = -sy
    R[...,2,1] = cy * sx
    R[...,2,2] = cx * cy
    return R


def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d) # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images

def apply_random_mask(image_tensor, mask_ratio=0.3):
    """
    对 (C, H, W) 图像 Tensor 应用 Patch Masking，将遮掩区域设置为白色 (1.0)。
    
    参数:
        image_tensor (torch.Tensor): 输入图像 Tensor，形状为 (C, H, W)。
        mask_ratio (float): 遮掩区域占总面积的比例。
        
    返回:
        torch.Tensor: 应用了白色 Patch Masking 的 Tensor。
    """
    _, H, W = image_tensor.shape
    device = image_tensor.device
    
    # 生成一个与图像大小相同的随机 Mask
    mask = torch.rand(H, W, device=device) < mask_ratio
    
    # 将 Mask 扩展到所有通道 (C, H, W)
    mask = mask.unsqueeze(0).expand_as(image_tensor)
    
    # 创建一个白色张量 (值 1.0)
    white_tensor = torch.ones_like(image_tensor, device=device)
    
    # 逻辑操作：
    # 1. 原始图像 * (~mask)：保留未被遮掩的区域。
    # 2. 白色张量 * mask：获取被遮掩区域的白色值。
    # 3. 两者相加得到最终结果。
    masked_tensor = (image_tensor * (~mask)) + (white_tensor * mask)
    
    return masked_tensor

import torch
import random
from typing import Tuple, Union
# 导入 Kornia 库
try:
    from kornia.filters import GaussianBlur2d
except ImportError:
    print("Warning: Kornia not installed. Please install it using 'pip install kornia'.")
    # 如果 Kornia 未安装，我们使用一个替代函数，但性能和功能可能受限
    # 在生产环境中，强烈推荐安装 Kornia。
    def GaussianBlur2d(*args, **kwargs):
        raise NotImplementedError("Kornia is required for this function.")


def apply_gaussian_blur(
    image_tensor: torch.Tensor, 
    kernel_size: Union[int, Tuple[int, int]] = 5, 
    sigma: Tuple[float, float] = (0.1, 2.0)
) -> torch.Tensor:
    """
    对 PyTorch Tensor 应用随机参数的高斯模糊。
    
    参数:
        image_tensor (torch.Tensor): 输入图像 Tensor，形状为 (C, H, W) 或 (B, C, H, W)，
                                     通常是 L4GM 中的 (3, H, W)。
        kernel_size (int | Tuple[int, int]): 模糊核的大小 (例如 5 代表 5x5)。
        sigma (Tuple[float, float]): Sigma 值的范围，将从中随机选择一个值。
        
    返回:
        torch.Tensor: 应用了高斯模糊的 Tensor。
    """
    
    # 确保 kernel_size 是奇数，这是高斯模糊核的要求
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if kernel_size[0] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1)
        
    # ----------------------------------------------------
    # 1. 形状处理：Kornia 的 GaussianBlur2d 要求 (B, C, H, W) 格式
    # ----------------------------------------------------
    original_shape = image_tensor.shape
    if len(original_shape) == 3:
        # 如果是 (C, H, W)，添加 batch 维度：(1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
    elif len(original_shape) != 4:
        raise ValueError(f"Input tensor must be (C, H, W) or (B, C, H, W), but got {original_shape}")

    # ----------------------------------------------------
    # 2. 随机选择 Sigma 值
    # ----------------------------------------------------
    sigma_val = random.uniform(*sigma)
    
    # ----------------------------------------------------
    # 3. 创建并应用高斯模糊操作
    # ----------------------------------------------------
    
    # 创建 GaussianBlur2d 实例。注意 sigma 必须是 float 或 Tuple[float, float]
    # 我们对 x 和 y 轴使用相同的随机 sigma 值
    blur_op = GaussianBlur2d(kernel_size, (sigma_val, sigma_val))
    
    # 确保 blur_op 和 tensor 在同一个设备上
    blur_op = blur_op.to(image_tensor.device)
    
    # 应用模糊
    blurred_tensor = blur_op(image_tensor)
    
    # ----------------------------------------------------
    # 4. 恢复原始形状
    # ----------------------------------------------------
    if len(original_shape) == 3:
        # 移除添加的 batch 维度：(C, H, W)
        return blurred_tensor.squeeze(0)
    else:
        return blurred_tensor