"""
Dataloader for after_resort_subset dataset.

This dataloader is designed specifically for the after_resort_subset dataset structure,
which contains reorganized data from with_gt_memory and with_gt_resample datasets.
The dataset follows a simplified structure without tar files.

Dataset Structure:
文件结构：
with_gt_memory/
├── Capture_Real_wg_memory_week3/
│   ├── input/
│   │   ├── 0141_cup_cover_object/
│   │   └── 0144_cups_cover_object/
│   └── eval_gt/
│       ├── 0141_cup_cover_object/
│       │   ├── eval_0/
│       │   └── eval_1/
│       └── 0144_cups_cover_object/
├── Render_sync_wg_memory_week1/
├── Render_sync_wg_memory_week2/
└── Render_sync_wg_memory_week3/
└── split/
    ├── render_train_v1.txt
    └── render_test_v1.txt
    └── capture_test_v1.txt
    └── capture_train_v1.txt

    
split.txt 文件内容示例：
Capture_Real_wg_memory_week11/0295_box_open_5
Capture_Real_wg_memory_week11/0299_book_open_3
Capture_Real_wg_memory_week11/0312_bag_open_4



Authors:
    Jiaxin Li

Date:
    2025-10-12
"""

import os
import cv2
import random
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options_conf import Options

from core.utils import get_rays, grid_distortion, orbit_camera_jitter
from core.utils import apply_gaussian_blur, apply_random_mask


from kiui.cam import orbit_camera

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Memory4DSubsetDataset(Dataset):
    """Dataset for after_resort_subset data structure.
    
    This dataset loads data from the reorganized after_resort_subset directory,
    which contains both training and test subsets with a simplified folder structure.
    
    Camera Setup:
        - Input view: elevation=0°, azimuth=0°
        - eval_0 to eval_3: elevation=0°, azimuth=[-75°, 15°, 105°, 195°]
    """

    def __init__(self, opt: Options, training=True, evaluating=False, clip_num_frame = None):
        """Initialize the dataset.
        
        Args:
            opt: Options object containing dataset configuration.
            training: Whether this is for training.
            evaluating: Whether this is for evaluation.
        """
        
        self.opt = opt
        self.training = training
        self.evaluating = evaluating
        self.type = opt.data_type # "render" or "capture"
        self.split_version = opt.split_version if hasattr(opt, 'split_version') else 'v1'
        self.train_subset = opt.train_subset
        
        # if self.training:
        if clip_num_frame is None:
            self.clip_num = self.opt.num_clip_frames
        else:
            self.clip_num = clip_num_frame
            

        # Camera parameters for different views
        # self.input_camera = {'elevation': 0.0, 'azimuth': 0.0}
        # self.eval_cameras = [
        #     {'elevation': 0.0, 'azimuth': -75.0},  # eval_0
        #     {'elevation': 0.0, 'azimuth': 15.0},   # eval_1
        #     {'elevation': 0.0, 'azimuth': 105.0},  # eval_2
        #     {'elevation': 0.0, 'azimuth': 195.0}   # eval_3
        # ]
        # 在这里input_images用的是eval_0 到eval_3的四个视角, 以 其中一个作为主视角,
        if self.type == "render":
            self.input_camera = {'elevation': 0.0, 'azimuth': 0.0}
            self.eval_cameras = [
                {'elevation': 0.0, 'azimuth': -75.0},  # eval_0
                {'elevation': 0.0, 'azimuth': 15.0},   # eval_1
                {'elevation': 0.0, 'azimuth': 105.0},  # eval_2
                {'elevation': 0.0, 'azimuth': 195.0}   # eval_3
            ]
        else:
            # 未支持
            raise NotImplementedError("Only 'render' type is currently supported.")

        # Load case list
        self.items = []
        if hasattr(opt, 'data_root'):
            self.data_root = opt.data_root
        else:
            assert False
        
        
        # self.subset_names = self.find_subset_name(self.type)
     
        self.split_file = self.get_split_file(self.type, self.training, self.split_version, self.train_subset)
        self.items = self.get_items(self.split_file)

        # Default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

    def __len__(self):
        if self.opt.debug_mode:
            return 16
        return len(self.items)
    
    def get_items(self, split_file):
        """
        items: {
        "0141_cup_cover_object": [<input_images_path>, <eval_gt_path>],
        ...
        }
        """
        case_list = []
        for line in open(split_file, 'r'):
            line = line.strip()
            if line:
                case_list.append(line)

        items = {}
        for case_name in case_list:
            items[case_name] = [os.path.join(self.data_root, case_name.split('/')[0], 'input', case_name.split('/')[1]),
                                os.path.join(self.data_root, case_name.split('/')[0], 'eval_gt', case_name.split('/')[1]),
                                os.path.join(self.data_root +"_mask" , case_name.split('/')[0], 'input', case_name.split('/')[1]),
                                 os.path.join(self.data_root +"_mask" , case_name.split('/')[0], 'eval_gt', case_name.split('/')[1])]
            
        
        
        return items

    def get_split_file(self, data_type, training, split_version, train_subset = None):
        """Get the split file path based on type and training flag.
        
        Args:
            type: 'render' or 'capture'.
            training: Boolean indicating training or test split.
            split_version: Version of the split file.
        Returns:
            str: Path to the split file.
        """
        split_type = 'train' if training else 'test' 
        
        if train_subset and training:
            split_version += f"_{train_subset}" 
        split_file = os.path.join(self.data_root, 'split', f"{data_type}_{split_type}_{split_version}.txt")
        
        
        
        print("Split path: ", split_file )
        return split_file

    

    def _load_image(self, image_path):
        """Load and preprocess image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            torch.Tensor: Preprocessed image tensor [3, H, W].
        """
        if not os.path.exists(image_path):
            # Return white image if file not found
            return torch.ones(3, 512, 512, dtype=torch.float32)
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return torch.ones(3, 512, 512, dtype=torch.float32)
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Handle different channel configurations
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Extract alpha for mask
                alpha = image[:, :, 3:4]
                image = image[:, :, :3]
                # Apply alpha blending with white background
                image = image * alpha + (1 - alpha)
            elif image.shape[2] == 3:  # RGB
                pass  # Use as is
            
            # Convert BGR to RGB if needed
            image = image[:, :, [2, 1, 0]]  # BGR to RGB
        else:
            # Grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
        
        # Convert to tensor and rearrange dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        return image
    
    def _load_image_masks(self, image_path):
        """Load and preprocess image.
        
        Args:
            image_path: Path to the image file. [R G B A]
            
        Returns:
            torch.Tensor: Preprocessed image tensor [3, H, W].
            tensor.Tensor: Preprocessed mask tensor [1, H, W].
        """
        if not os.path.exists(image_path):
            # Return white image if file not found
            Warning("Image path not found: {}".format(image_path))
            return torch.ones(3, 512, 512, dtype=torch.float32), torch.zeros(1, 512, 512, dtype=torch.float32)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            Warning("Image is None: {}".format(image_path))
            return torch.ones(3, 512, 512, dtype=torch.float32), torch.zeros(1, 512, 512, dtype=torch.float32)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Handle different channel configurations
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Extract alpha for mask
                alpha = image[:, :, 3:4]
                image = image[:, :, :3]
                # Apply alpha blending with white background
                image = image * alpha + (1 - alpha)
                mask = torch.from_numpy(alpha).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]
            elif image.shape[2] == 3:  # RGB    
                mask = torch.ones(1, image.shape[0], image.shape[1], dtype=torch.float32)  # Full mask
 
            # Convert BGR to RGB if needed
            image = image[:, :, [2, 1, 0]]  # BGR to RGB
        else:
            # Grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
            mask = torch.ones(1, image.shape[0], image.shape[1], dtype=torch.float32)  # Full mask
        
        # Convert to tensor and rearrange dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        return image, mask
    

    def _get_camera_pose(self, elevation, azimuth):
        """Get camera pose from elevation and azimuth angles.
        
        Args:
            elevation: Elevation angle in degrees.
            azimuth: Azimuth angle in degrees.
            
        Returns:
            torch.Tensor: Camera-to-world transformation matrix [4, 4].
        """
        return torch.from_numpy(orbit_camera(elevation, azimuth, radius=1.5, opengl=True))

    def _get_frame_count(self, case_name):
        """Get the number of frames in a case.
        
        Args:
            case_name: Name of the case.
            
        Returns:
            int: Number of frames.
        """
        input_case_path = self.items[case_name][0]
        if not os.path.exists(input_case_path):
            return 32  # Default frame count
        
        # Count PNG files
        png_files = glob.glob(os.path.join(input_case_path, "*.png"))
        return len(png_files) if png_files else 32

    def _get_batch(self, idx):
        """Get a batch of data for the given index.
        
        Args:
            idx: Dataset index.
            
        Returns:
            dict: Dictionary containing all required outputs matching the original dataloader.
        """
        case_name = list(self.items.keys())[idx]
        results = {}

        # Get frame count for this case
        total_frames = self._get_frame_count(case_name)
        T = min(self.opt.num_frames, total_frames)

        # Storage for all data
        images = []
        masks = []
        cam_poses = []
        
        if self.training:
            # 随机选择基准视角 (eval_0 到 eval_3)
            base_eval_idx = random.randint(0, 3)
            base_azimuth = self.eval_cameras[base_eval_idx]['azimuth']
            # 创建轮转排序的eval索引列表 (从基准视角开始，按eval_id从小到大轮转)
            eval_indices = [(base_eval_idx + i) % 4 for i in range(4)]
        else:
            eval_indices = [0,1,2,3]
            base_azimuth = 0
            

        
        # Load data for each time frame
        for t_idx in range(T):
            # 临时存储当前帧的数据
            frame_images = []
            frame_masks = []
            frame_cam_poses = []
            
            # 按轮转排序加载eval views (eval_0到eval_3，前四位)
            for eval_idx in eval_indices:
                eval_camera = self.eval_cameras[eval_idx]
                # 调整azimuth相对于基准视角，创建新的值而不修改原始数据
                adjusted_azimuth = (eval_camera['azimuth'] - base_azimuth) % 360
                
                eval_case_path = os.path.join(self.items[case_name][1], f"eval_{eval_idx}")
                eval_image_path = os.path.join(eval_case_path, f"{t_idx:06d}.png")
                
                # Load eval image
                # eval_image = self._load_image(eval_image_path)
                eval_image, eval_mask = self._load_image_masks(eval_image_path)
                eval_pose = self._get_camera_pose(
                    eval_camera['elevation'], 
                    adjusted_azimuth
                )
                
                frame_images.append(eval_image)
                frame_masks.append(eval_mask)
                frame_cam_poses.append(eval_pose)
            

            # 加载input view并放在最后一位
            input_case_path = self.items[case_name][0]
            input_image_path = os.path.join(input_case_path, f"{t_idx:06d}.png")
            # print("input_image_path:", input_image_path)
        
            
            # Load input image
            # input_image = self._load_image(input_image_path)
            input_image, input_mask = self._load_image_masks(input_image_path)
            # input视角的azimuth也要相对于基准视角调整
            input_azimuth = (self.input_camera['azimuth'] - base_azimuth) % 360
            input_pose = self._get_camera_pose(
                self.input_camera['elevation'], 
                input_azimuth
            )
            if self.training:
                # Add input view (最后一位)
                frame_images.append(input_image)
                frame_masks.append(input_mask)
                frame_cam_poses.append(input_pose)
            else:
                # 不是训练的时候，加载第一个视角在第一位
                frame_images = [input_image] + frame_images
                frame_masks = [input_mask] + frame_masks
                frame_cam_poses = [input_pose] + frame_cam_poses
                
            # 将当前帧的数据添加到总列表
            images.extend(frame_images)
            masks.extend(frame_masks)
            cam_poses.extend(frame_cam_poses)
            
        assert self.opt.num_views == 5 # 这里的相机视角太少了，所以需要都用上; 并且这里仅支持num_views == 5, 其他值会导致视角对不齐

        # Ensure we have enough views
        target_view_count = self.opt.num_views * T
        while len(images) < target_view_count:
            # Duplicate last view if not enough views
            images.append(images[-1].clone())
            masks.append(masks[-1].clone())
            cam_poses.append(cam_poses[-1].clone())

        # Truncate if too many views
        images = images[:target_view_count]
        masks = masks[:target_view_count]
        cam_poses = cam_poses[:target_view_count]

        # Convert to tensors
        images = torch.stack(images, dim=0)  # [V, C, H, W]
        masks = torch.stack(masks, dim=0)   # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]

        # Resize images to 512x512 if needed
        images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)
        masks = F.interpolate(masks, size=(512, 512), mode='bilinear', align_corners=False)

        # Normalize camera poses (transform first pose to fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], 
                                dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses

        # Prepare input images (first num_input_views for each frame)
        images_input = images.reshape(T, self.opt.num_views, *images.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *images.shape[1:]).clone()
        cam_poses_input = cam_poses.reshape(T, self.opt.num_views, *cam_poses.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *cam_poses.shape[1:]).clone()

        # Resize input images
        images_input = F.interpolate(images_input, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)

        # Data augmentation (only during training)
        if self.training:
            images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])
            cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])

            # Apply random grid distortion
            if random.random() < self.opt.prob_grid_distortion:
                for t in range(T):
                    if self.opt.num_input_views > 1:
                        images_input[t, 1:] = grid_distortion(images_input[t, 1:])

            # Apply camera jittering
            if random.random() < self.opt.prob_cam_jitter:
                for t in range(T):
                    if self.opt.num_input_views > 1:
                        cam_poses_input[t, 1:] = orbit_camera_jitter(cam_poses_input[t, 1:])

            images_input = images_input.reshape(-1, *images_input.shape[2:])
            cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])

        # Masking other views (same as original implementation)
        # 掩码其他视图（时间一致性处理）
        images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])
        cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])
        
        CN = T // self.clip_num
        for clip_id in range(CN):
            start_clip_id = clip_id  * self.clip_num
            images_input[start_clip_id+1 : start_clip_id + self.clip_num, 1:] = images_input[start_clip_id :start_clip_id+1, 1:]
            cam_poses_input[start_clip_id+1 :start_clip_id + self.clip_num, 1:] = cam_poses_input[start_clip_id :start_clip_id+1, 1:] 
            # print(f"copy from {start_clip_id} to {start_clip_id+1}: {start_clip_id + self.clip_num}")
        # assert False
        # images_input[1:, 1:] = images_input[0:1, 1:]
        # cam_poses_input[1:, 1:] = cam_poses_input[0:1, 1:]
        
        images_input = images_input.reshape(-1, *images_input.shape[2:])
        cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])

        
        # 数据增强v2（Jx 的数据增强,发现把数据增强放在mask之前作用不大）
        if self.training and self.opt.diy_data_aug:
            images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])
            cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])

            # 应用网格变形
            if random.random() < self.opt.prob_grid_distortion:
                for t in range(T):
                    images_input[t, 1:] = grid_distortion(images_input[t, 1:])
            
            # 应用相机抖动
            if random.random() < self.opt.prob_cam_jitter:
                for t in range(T):
                    cam_poses_input[t, 1:] = orbit_camera_jitter(cam_poses_input[t, 1:])
                    
            # ----------------------------------------------------------------------
            # 核心修正：按 Clip 进行 DIY Jx 增强
            # ----------------------------------------------------------------------
            
            clip_len = self.clip_num # 每个 Clip 的长度
            CN = T // clip_len        # Clip 的数量
            V = self.opt.num_input_views # 视角数量
            
            for clip_id in range(CN):
                # 确定当前 Clip 的起始帧索引 (该 Clip 中的 '第一帧')
                start_frame_idx = clip_id * clip_len
                                
                # ---------------------------------------------------
                # **1. 随机mask掉 Clip 起始帧的另外三个视角 (diy jx)**
                # ---------------------------------------------------
                # ... (概率检查和索引选择代码不变) ...
                if random.random() < self.opt.prob_mask_v0_views:
                    num_views_to_mask = random.randint(1, V - 1)
                    masked_view_indices = random.sample(range(1, V), num_views_to_mask)
                    print("mask views:", masked_view_indices)
                    
                    # --- 关键修正：使用全白张量进行替换 ---
                    # 创建一个全白张量，形状与单个图像 (3, H, W) 相同
                    white_view = torch.ones_like(images_input[start_frame_idx :start_frame_idx + clip_len, 0]) 
                    
                    for v_idx in masked_view_indices:
                        images_input[start_frame_idx:start_frame_idx + clip_len, v_idx] = white_view
                    
                # ---------------------------------------------------
                # **2. 随机给 Clip 起始帧的另外三个视角加入强高斯模糊 (diy jx)**
                # ---------------------------------------------------
                # 目标：降低 Clip 内部初始化几何信息的精度
                if random.random() < self.opt.prob_blur_v0_views:
                    # 随机选择一个辅助视角进行模糊
                    view_idx = random.randint(1, V - 1) 
                    print("gs views:", view_idx)
                    images_input[start_frame_idx:start_frame_idx + clip_len, view_idx] = apply_gaussian_blur(
                        images_input[start_frame_idx:start_frame_idx + clip_len, view_idx], 
                        kernel_size=10, 
                        sigma=(3.0, 5.0)
                    )
                
                # ---------------------------------------------------
                # **3. 随机mask掉 第一个视角的某些帧 （diy jx)**
                # ---------------------------------------------------
                # ... (概率检查和索引选择代码不变) ...
                if random.random() < self.opt.prob_mask_temporal_frames:
                    
                    clip_indices_to_sample = list(range(start_frame_idx + 1, start_frame_idx + clip_len))
                    
                    if len(clip_indices_to_sample) > 0:
                        num_frames_to_mask = random.randint(1, max(1, len(clip_indices_to_sample) // 3)) 
                        frame_indices_to_mask = random.sample(clip_indices_to_sample, num_frames_to_mask)
                        print("mask frame:", frame_indices_to_mask )
                        
                        for t in frame_indices_to_mask:
                            # 调用修正后的函数，应用白色 Patch Masking (只对 V=0 的单目视频流)
                            images_input[t, 0] = apply_random_mask(images_input[t, 0], mask_ratio=0.4)

            images_input = images_input.reshape(-1, *images_input.shape[2:])
            cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])
           


        # Normalize input images
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # Prepare output images and masks
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
        # print("masks:", masks.shape)
        results['masks_output'] = F.interpolate(masks, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)

        # Build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views * T):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()

        # Final input combining images and rays
        final_input = torch.cat([images_input, rays_embeddings], dim=1)
        results['input'] = final_input



        cam_poses_input[:, :3, 1:3] *= -1  # 翻转上方向和前方向
        # 准备渲染器所需的相机参数
        cam_view_input = torch.inverse(cam_poses_input).transpose(1, 2)
        cam_view_proj_input = cam_view_input @ self.proj_matrix
        cam_pos_input = - cam_poses_input[:, :3, 3]
        
        results['cam_view_input'] = cam_view_input
        results['cam_view_proj_input'] = cam_view_proj_input
        results['cam_pos_input'] = cam_pos_input
        
        # Convert to COLMAP camera coordinate system
        cam_poses[:, :3, 1:3] *= -1

        # Prepare camera matrices for Gaussian rasterization
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ self.proj_matrix
        cam_pos = -cam_poses[:, :3, 3]

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        results["case_name"] = case_name
        
        results["input_mask_folder_root"] = self.items[case_name][2]
        results["eval_mask_folder_root"] = self.items[case_name][3]
        
        results = self.add_conf(results)

        return results
    
    def add_conf(self, results):
        # TODO
        # 需要用blender去生成对应的conf, 然后用来微调带有置信度的模型
        results["confs_output"] = torch.zeros_like(results["masks_output"])
        return results

    def __getitem__(self, idx):
        """Get item with error handling.
        
        Args:
            idx: Dataset index.
            
        Returns:
            dict: Data dictionary with all required fields.
        """
        while True:
            # print("self.items:",len(self.items))
            # print("idx:", idx)
            try:
                results = self._get_batch(idx)
                break
            except Exception as e:
                key = self.items.keys()[idx]
                print(f"Error loading case {self.items[key]}: {e}")
                idx = random.randint(0, len(self.items) - 1)
        return results


# Usage example
if __name__ == "__main__":
    # Example usage
    from core.options import Options
    
    # Create mock options for testing
    opt = Options()
    opt.data_root = './data/example_dataset'
    opt.num_frames = 4
    opt.num_views = 5
    opt.num_input_views = 2
    opt.input_size = 256
    opt.output_size = 512
    opt.fovy = 40
    opt.cam_radius = 1.5
    opt.znear = 0.1
    opt.zfar = 100.0
    opt.prob_grid_distortion = 0.5
    opt.prob_cam_jitter = 0.5
    
    # Create datasets
    train_dataset = AfterResortSubsetDataset(opt, training=True)
    test_dataset = AfterResortSubsetDataset(opt, training=False)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test loading a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print("Sample keys:", sample.keys())
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
