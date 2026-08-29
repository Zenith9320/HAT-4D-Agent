# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# dist
# ributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import json
import kiui
from core.options_conf import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter, flip_pitch_towards_origin,flip_pitch_towards_origin_stable,mirror_cam_xz
from core.utils import apply_gaussian_blur, apply_random_mask
from tqdm import tqdm
from kiui.cam import orbit_camera

import tarfile
from io import BytesIO

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
"""
相对于v0。 把原本的红色 灰色 黑色背景的rgb图片改成 单通道的 1 0 -1 的图


"""


def load_np_array_from_tar(tar, path):
    array_file = BytesIO()
    array_file.write(tar.extractfile(path).read())
    array_file.seek(0)
    return np.load(array_file)

def save_image(img: torch.Tensor, path: str):
    """
    保存张量为图片。
    支持形状:
      - [3, H, W]: RGB 单张
      - [N, 3, H, W]: RGB 多张（水平拼接）
      - [1, H, W] 或 [N, 1, H, W]: 单通道置信度热图
    可视化规则（置信度图时）:
      -1 → 黑色
       0 → 灰色
       1 → 红色
    """
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # 统一为 [N, C, H, W]
    if img.ndim == 5:
        img = img.reshape(-1, *img.shape[-3:])
    elif img.ndim == 3:
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")

    img = img.detach().cpu()
    out_imgs = []

    for im in img:  # [C, H, W]
        im_np = im.numpy()
        C = im_np.shape[0]
        minv, maxv = float(im_np.min()), float(im_np.max())

        if C == 3:
            # ===== RGB 图像处理 =====
            if maxv > 1.0 or minv < 0.0:
                if maxv > minv:
                    im_np = (im_np - minv) / (maxv - minv)
                else:
                    im_np = np.zeros_like(im_np)
            im_np = np.clip(im_np, 0.0, 1.0)
            im_np = (im_np * 255.0 + 0.5).astype(np.uint8)
            im_np = np.transpose(im_np, (1, 2, 0))  # [H, W, 3]

        elif C == 1:
            # ===== 单通道置信度热图 =====
            conf = im_np[0]
            conf = np.clip(conf, -1.0, 1.0)

            # 初始化 RGB 输出
            H, W = conf.shape
            color_img = np.zeros((H, W, 3), dtype=np.uint8)

            # 三个区间分别赋值
            mask_bg = conf < -0.5       # 背景
            mask_gray = (conf >= -0.5) & (conf < 0.5)  # 中间灰
            mask_red = conf >= 0.5      # 高置信度

            color_img[mask_bg] = (0, 0, 0)             # 黑
            color_img[mask_gray] = (128, 128, 128)     # 灰
            color_img[mask_red] = (255, 0, 0)          # 红

            im_np = color_img

        else:
            raise ValueError(f"Unsupported channel count: {C}")

        out_imgs.append(im_np)

    # 多张拼接
    if len(out_imgs) == 1:
        grid = out_imgs[0]
    else:
        grid = np.concatenate(out_imgs, axis=1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 保存
    try:
        from PIL import Image
        Image.fromarray(grid).save(path)
    except Exception:
        plt.imsave(path, grid)

    print(f"✅ 已保存图像: {path}")
    
class SafeLPIPS(nn.Module):
    def __init__(self, lpips_fn):
        super().__init__()
        self.lpips = lpips_fn

    def forward(self, pred, gt):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        gt = torch.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=-1.0)
        with torch.cuda.amp.autocast(enabled=False):
            return self.lpips(pred.float(), gt.float())
        
        
        
class ObjaverseDataset_Conf_V2(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True, evaluating=False, clip_num_frame = None):
        
        self.opt = opt
        self.training = training
        self.evaluating = evaluating
        # if self.training:
        if clip_num_frame is None:
            self.clip_num = self.opt.num_clip_frames
        else:
            self.clip_num = clip_num_frame
   
     
        self.load_meta(opt.datalist, training= training, evaluating=evaluating)



        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        
        # conf_get_parameter
        self.red_thresh = 0.5
        self.diff_thresh = 0.2
        


    def __len__(self):
        # return 32
        if self.opt.debug_mode:
            return 16
        return len(self.items)
    
        # return 15 # for debug
        
    def get_ani_idx(self, idx):
        return len(self.items[idx])
    
    def get_speical_item_ani_idx(self, idx, anim_idx):
        return self._get_batch(idx, anim_idx)
        


    def load_meta(self, meta_path, training=True, evaluating=False):
        """
        加载元数据文件
        
        Args:
            meta_path (str): 元数据文件路径
            training (bool): 是否为训练模式
            evaluating (bool): 是否为评估模式
        
        Returns:
            list: 元数据列表
        """
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"元数据文件 {meta_path} 不存在")
        
        with open(meta_path, 'r') as f:
            dataset_meta = json.load(f)
        
        # 这里按照sha256分组，将相同sha256的不同视角合并在一起
        anim_map = {}
        for item in dataset_meta:
            key = item["sha256"]
            if key in anim_map:
                anim_map[key].append(item)
            else:
                anim_map[key] = [item]
        
        self.items = list(anim_map.values())
        
        # 按训练/评估/测试进行分割
        if training:
            # self.items = self.items[:-self.opt.batch_size * 40]
            self.items = self.items[:-1 * 64] # 为了确保test set保持一致
            print("training dataloader:", len(self.items))
        elif evaluating:
            self.items = self.items[::1000]
            print("evlaution:", len(self.items))
        else:
            # self.items = self.items[-self.opt.batch_size * 40:]
            self.items = self.items[-1 * 64:]  # 为了确保test set保持一致
            print("test:", len(self.items))

        
        # if True: # for debug
        #     self.items = self.items[:30]
        #     # self.items = self.items
            
        print(f"加载了 {len(self.items)} 个数据项") 
        
        return self.items

    def _get_available_views(self, base_path):
        """
        通过扫描目录结构获取所有可用的视角ID
        
        Args:
            base_path (str): 数据基础路径，到动画文件夹的路径
            
        Returns:
            list: 可用视角ID列表，如 ["000", "001", ...]
        """
        available_views = []
        
        try:
            # 列出base_path下的所有目录
            items = os.listdir(base_path)
            
            # 筛选出render_view_开头的目录
            for item in items:
                if item.startswith("render_view_") and os.path.isdir(os.path.join(base_path, item)):
                    # 提取视角ID
                    view_id = item[len("render_view_"):]
                    available_views.append(view_id)
            
            # 按视角ID排序
            available_views.sort()
            
            if not available_views:
                print(f"警告：在 {base_path} 中没有找到任何render_view_XXX目录")
        except Exception as e:
            print(f"扫描视角目录时出错: {e}")
        
        return available_views

    def _get_batch(self, idx, anim_idx = None):
        # 获取当前样本的元数据
        item_group = self.items[idx]
        
        # 训练时随机选择该sha256下的一个视角，否则使用第一个
        if self.training:
            item = random.choice(item_group)
        else:
            item = item_group[0]
            
        if anim_idx:
            item = item_group[anim_idx]
        
        sha256 = item["sha256"]
        image_folder = item["image_folder"]
        animation_idx = item["animation_idx"]
        input_view_id = item["input_view_id"]
        source_folder = item["source_folder"]
        
        results = {}
        
        # 基础路径构建
        base_path = os.path.join(
            self.opt.train_dataset_path,
            # os.environ.get("DATASET_PATH", ),
            source_folder,
            sha256,
            image_folder
        )
        
        # 加载相机参数
        camera_info_path = os.path.join(base_path, "camera_info.pkl")
        camera_K_path = os.path.join(base_path, "camera_K_info.pkl")
        
        import pickle
        try:
            with open(camera_info_path, 'rb') as f:
                # print("camera_info_path:", camera_info_path)
                camera_info = pickle.load(f)
            
            with open(camera_K_path, 'rb') as f:
                camera_K_info = pickle.load(f)
                #! 目前内参还是使用option里面的默认内参
        except Exception as e:
            print(f"无法加载相机参数文件: {e} : {camera_info_path}")
            raise e
        
        # 准备数据容器
        images = []
        masks = []
        confs = []
        cam_poses = []
        
        
        
        # 确定要使用的视角列表
        # 假设有16个视角，从0到15
        # 扫描目录获取可用的视角ID
        available_views = self._get_available_views(base_path)
        
        # 如果没有找到视角，使用默认范围
        if not available_views:
            print(f"警告：在 {base_path} 中没有找到视角目录，使用默认视角范围")
            available_views = [f"{i:03d}" for i in range(16)]
            
        
        
        # 如果有指定输入视角，确保它在列表中
        if input_view_id not in available_views:
            available_views.append(input_view_id)
        
        # 根据训练或评估模式选择视角
        if self.training and self.opt.shuffle_input:
            # 随机选择视角，但确保input_view_id在前self.opt.num_input_views个
            other_views = [v for v in available_views if v != input_view_id and int(v) > 8] # 8-24的视角是随机视角，作为监督视角
            other_input_viwes =  [v for v in available_views if v != input_view_id and int(v) < 8 ] # 八个固定视角作为预备的输入视角
            random.shuffle(other_views)
            random.shuffle(other_input_viwes)
            vids = [input_view_id] + other_input_viwes[:self.opt.num_input_views - 1] +  other_views[:self.opt.num_views-self.opt.num_input_views]
        else:
            # 固定选择，input_view_id作为第一个视角
            input_other = [  str((vid + int(input_view_id)) % 8).zfill(3) for vid in range(2,8,2)] # 以input_view_id为起点，选取差距为90°的另外三个视角
            fixed_output_num = 2
            fixed_output = [v for v in available_views if v != input_view_id and v not in input_other and int(v) < 8][:fixed_output_num]
            random_output = [v for v in  available_views if v != input_view_id and v not in input_other and int(v) > 8][:(self.opt.num_views- self.opt.num_input_views - len(fixed_output)) ]
            vids = [input_view_id] + input_other + fixed_output + random_output
        # print("vids:", vids)
        
        # 处理多个时间帧
        T = self.opt.num_frames
        for t_idx in (range(T)):
            vid_cnt = 0
            for view_id in vids:
                # 构建视角路径
                view_path = os.path.join(base_path, f"render_view_{view_id}")
                
                # 图像和掩码路径
                image_path = os.path.join(view_path, "img", f"{t_idx:03d}.jpg")
                # print(f"get images from: {image_path}")
                mask_path = os.path.join(view_path, "mask", f"{t_idx:03d}.png")
                conf_path = os.path.join(view_path, "confidence", f"input_view_{input_view_id}" , f"{t_idx:03d}.png")
                
                # 检查文件是否存在
                if not os.path.exists(image_path) or not os.path.exists(mask_path) or not os.path.exists(conf_path):
                    print(f"警告：找不到图像或掩码: {image_path} / {mask_path}")
                    continue
                
                # 读取图像和掩码
                image = cv2.imread(image_path)  # 默认BGR格式
                if image is None:
                    print(f"警告：无法读取图像: {image_path}")
                    continue
                
                
                    
                image = torch.from_numpy(image.astype(np.float32) / 255)  # [H, W, 3] in [0, 1]
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"警告：无法读取掩码: {mask_path}")
                    continue
                    
                mask = torch.from_numpy(mask.astype(np.float32) / 255)  # [H, W]
                
                conf = cv2.imread(conf_path) 
                
                if conf is None:
                    print(f"警告：无法读取置信度热图: {conf_path}")
                    continue
                    
                conf = torch.from_numpy(conf.astype(np.float32) / 255) # [H, W, 3] in [0, 1]
                

                
                # 从camera_info中获取相机参数
                try:
                    self.tmp_print = camera_info_path
                    c2w = self._get_camera_pose_from_info(camera_info,  view_id, t_idx)
                except Exception as e:
                    print(f"无法获取相机参数: {e}")
                    continue
                
                # 处理图像格式
                image = image.permute(2, 0, 1)  # [3, H, W]
                mask = mask.unsqueeze(0)  # [1, H, W]
                conf = conf.permute(2, 0, 1)  # [3, H, W]
                

                # 取红通道或亮度作为置信度值（推荐取 max 更稳健）
                # red, green, blue = conf[..., 2], conf[..., 1], conf[..., 0]
                # # 红色区域（关键前景）
                # red_mask = (red > self.red_thresh) & (red - green > self.diff_thresh) & (red - blue > self.diff_thresh)
                # # 背景区域（mask中为0 或几乎全黑）
                # bg_mask = ((mask < 0.5) | ((red < 1e-3) & (green < 1e-3) & (blue < 1e-3)))[0]

                # # 构建输出
                # conf_map = torch.zeros_like(red, dtype=torch.float32)
                # conf_map[red_mask] = 1.0   # 红色区域
                # # print("bg_mask.shape:",bg_mask.shape)
                # # print("red_mask:",red_mask.shape)
                # conf_map[~red_mask & ~bg_mask] = 0.0  # 前景非红色
                # conf_map[bg_mask] = -1.0   # 背景



                # 调整大小
                image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                mask = F.interpolate(mask.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                conf = F.interpolate(conf.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
                
                
                # 白背景处理
                image = image * mask + (1 - mask)  # [3, 512, 512], 白背景
                image = image[[2,1,0]].contiguous()  # BGR转RGB
                
                # 白背景处理 
                # ! 后续看看要不要直接这里把他转成[0,1]的 一维的值
                conf = conf * mask + (1 - mask)  # [3, 512, 512], 白背景
                # conf = conf[[2,1,0]].contiguous()  # BGR转RGB
            
                
                
                images.append(image)
                confs.append(conf)
                masks.append(mask.squeeze(0))
                cam_poses.append(c2w)
                
                vid_cnt += 1
                if vid_cnt == self.opt.num_views:
                    break
            
            # 处理视角不足的情况
            if vid_cnt < self.opt.num_views:
                print(f'[WARN] dataset {sha256}: not enough valid views, only {vid_cnt} views found!')
                if vid_cnt == 0:
                    # 如果一个有效视角都没有，跳过这个样本
                    raise ValueError(f"No valid views found for {sha256}")
                
                # 复制最后一个视角来填充
                n = self.opt.num_views - vid_cnt
                images = images + [images[-1]] * n
                masks = masks + [masks[-1]] * n
                confs = confs + [confs[-1]] * n
                cam_poses = cam_poses + [cam_poses[-1]] * n
 
        # 以下处理与原始代码保持一致
        images = torch.stack(images, dim=0)  # [V, C, H, W]
        masks = torch.stack(masks, dim=0)  # [V, H, W]
        confs = torch.stack(confs, dim=0)  # [V, C, H, W]
        

        # conf 的特别处理
        # [V, C, H, W] → 分离颜色通道
        red   = confs[:, 2, :, :]  # [V, H, W]
        green = confs[:, 1, :, :]
        blue  = confs[:, 0, :, :]
        
        # print("red:", red.shape)
        # print("red:", torch.unique(red))
        
        # print("green:", green.shape)
        # print("green:", torch.unique(green))
    
        # print("blue:", blue.shape)
        # print("blue:", torch.unique(blue))

        # 计算红色区域掩膜
        red_mask = (red > self.red_thresh) & (red - green > self.diff_thresh) & (red - blue > self.diff_thresh)
        
        # print("red_masks:", torch.unique(red_mask))

        # 背景掩膜（mask<0.5 或接近黑色）
        bg_mask = (masks < 0.5) | ((red < 1e-3) & (green < 1e-3) & (blue < 1e-3))

        # 初始化输出 [V, H, W]
        conf_maps = torch.zeros_like(red, dtype=torch.float32)

        # 根据规则赋值
        conf_maps[red_mask] = 1.0
        conf_maps[~red_mask & ~bg_mask] = 0.0
        conf_maps[bg_mask] = -1.0

        # 增加 channel 维度以统一 shape
        confs = conf_maps.unsqueeze(1)  # [V, 1, H, W]
        # print("DONE")


        
        cam_poses = torch.stack(cam_poses, dim=0)  # [V, 4, 4]

        # 标准化相机位姿
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        # 处理输入图像
        images_input = F.interpolate(
            images.reshape(T, self.opt.num_views, *images.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *images.shape[1:]).clone(), 
            size=(self.opt.input_size, self.opt.input_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        cam_poses_input = cam_poses.reshape(T, self.opt.num_views, *cam_poses.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *cam_poses.shape[1:]).clone()

        # 数据增强
        if self.training:
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

            images_input = images_input.reshape(-1, *images_input.shape[2:])
            cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])

        # 掩码其他视图（时间一致性处理）
        images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])
        cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])
        
        CN = T // self.clip_num
        for clip_id in range(CN):
            start_clip_id = clip_id  * self.clip_num
            images_input[start_clip_id+1 : start_clip_id + self.clip_num, 1:] = images_input[start_clip_id :start_clip_id+1, 1:]
            cam_poses_input[start_clip_id+1 :start_clip_id + self.clip_num, 1:] = cam_poses_input[start_clip_id :start_clip_id+1, 1:] 
            # print("change clip:", start_clip_id)   
        
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
            

        # 图像标准化
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # 调整渲染的GT图像大小
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)
        results["confs_output"] = F.interpolate(confs, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False)


        
        
        # 生成光线嵌入
        rays_embeddings = []
        for i in range(self.opt.num_input_views * T):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
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

        # OpenGL到COLMAP的相机转换
        cam_poses[:, :3, 1:3] *= -1  # 翻转上方向和前方向
                
        # cam_poses = flip_pitch_towards_origin(cam_poses) V2
        
        # cam_poses = flip_pitch_towards_origin_stable(cam_poses) # v4
        
        
        # 这个是对的
        # cam_poses = mirror_cam_xz(cam_poses) # V5
        
        
        # print("in dataloader gt_conf[i]:", torch.unique(results["confs_output"][0]))
        # save_image(results["confs_output"][0], f"./debug_train_viz/gt_confs/frame_{i}_dataloader.png")
        # print("save_images:", f"./debug_train_viz/gt_confs/frame_{i}_dataloader.png")
        
        # 准备渲染器所需的相机参数
        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ self.proj_matrix
        cam_pos = - cam_poses[:, :3, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        
        results["sha256"]= item["sha256"]
        results["image_folder"] = item["image_folder"]
        results["animation_idx"] = item["animation_idx"]
        results["input_view_id"] = item["input_view_id"]
        results["source_folder"] = item["source_folder"]

        return results
        


    def __getitem__(self, idx):
        while True:
            try:
                results = self._get_batch(idx)
                break
            except Exception as e:
                print(f"get item error:{e}")
            idx = random.randint(0, len(self.items) - 1)
        return results
    


    def _get_camera_pose_from_info(self, camera_info, view_id, frame_id):
        """
        根据实际的camera_info.pkl结构提取相机姿态
        
        Args:
            camera_info (dict): 相机外参信息，包含各视角的旋转、仰角、半径等
            camera_K_info (dict): 相机内参信息，包含K矩阵、焦距等
            view_id (str): 视角ID，如 "000"
            frame_id (int): 帧ID
            
        Returns:
            torch.Tensor: 4x4相机姿态矩阵
        """
        # 根据camera_info中的视角信息构建相机姿态
        view_key = f"render_view_{view_id}"
        
        # 检查视角是否存在
        if view_key not in camera_info:
            print("cam_info_path:", self.tmp_print)
            print(f"警告：视角 {view_key} 在camera_info中不存在，使用默认相机姿态")
            assert False
            # 默认相机姿态，基于视角ID的均匀分布
            view_id_int = int(view_id)
            angle = (view_id_int / 16) * 2 * np.pi
            elevation = 0.0
            c2w = torch.from_numpy(orbit_camera(elevation, angle, radius=1.5, opengl=True))
            return c2w
            
        view_info = camera_info[view_key]
        
        # # 检查是否包含rt_matrix（相机矩阵）
        # if 'rt_matrix' in view_info:
        #     # 直接使用提供的相机矩阵
        #     rt_matrix = view_info['rt_matrix']
            
        #     # 检查是否有多帧
        #     if isinstance(rt_matrix, list) and len(rt_matrix) > frame_id:
        #         # 如果是多帧列表，选择对应帧
        #         c2w = torch.tensor(rt_matrix[frame_id], dtype=torch.float32)
        #     else:
        #         # 单一矩阵或帧数不足
        #         c2w = torch.tensor(rt_matrix, dtype=torch.float32)
            
        #     # 扩展为4x4矩阵（如果不是）
        #     if c2w.shape == (3, 4):
        #         # 3x4矩阵扩展为4x4
        #         c2w_4x4 = torch.eye(4, dtype=torch.float32)
        #         c2w_4x4[:3, :4] = c2w
        #         return c2w_4x4
        #     elif c2w.shape == (4, 4):
        #         return c2w
        #     else:
        #         print(f"警告：相机矩阵形状异常 {c2w.shape}，使用默认值")
        
        # 如果没有直接的rt_matrix，使用角度信息构建
        rotation = view_info.get('rotation', 0.0)
        elevation = view_info.get('elevation', 0.0) * -1 
        if int(view_id) < 8 and self.opt.force_fixed_ele_zero  : 
            elevation = 0
        # print(f"{view_id} elevation:", elevation)
        radius = view_info.get('radius', 1.5)
        
        # print("view:", )
        # print("radius:", radius)
        # print("rotat:", rotation)
        # print('ele:', elevation)
        
        # 检查是否有多帧角度信息
        if isinstance(rotation, list) and len(rotation) > frame_id:
            rotation = rotation[frame_id]
        if isinstance(elevation, list) and len(elevation) > frame_id:
            elevation = elevation[frame_id]
        
        # 使用角度信息构建相机姿态
        c2w = torch.from_numpy(orbit_camera(elevation, rotation, radius=radius, opengl=True))
        return c2w

    
