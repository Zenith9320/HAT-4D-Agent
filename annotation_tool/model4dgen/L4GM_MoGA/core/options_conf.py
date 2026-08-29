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

import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional


@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    down_moga_attention: Tuple[bool, ...] = (False, False, False, True, True, False)
    
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    up_moga_attention: Tuple[bool, ...] =  (False, True, True, False, False)
    
    mid_moga_attention: bool = False
    # Unet output size, dependent on the input_size and U-Net structure!
    splat_size: int = 64
    # gaussian render size
    output_size: int = 256
    debug_mode:bool = False

    ### dataset
    # data mode (only support s3 now)
    data_mode: str = '4d_conf'
    # fovy of the dataset
    fovy: float = 49.1
    # camera near plane
    znear: float = 0.1
    # camera far plane
    zfar: float = 100
    # number of all views (input + output)
    num_views: int = 12
    # number of views
    num_input_views: int = 4
    # camera radius
    cam_radius: float = 1.5 # to better use [-1, 1]^3 space
    # num workers
    num_workers: int = 8
    datalist: str='core/options.py'
    train_dataset_path: str = ""
    
    
    force_fixed_ele_zero: bool = False
    data_root: str = ""
    data_type: str = "render"
    split_version: str = "v1"
    train_subset: Optional[str] = None
    memory_data_root: Optional[str] = None
    memory_data_split_version: Optional[str] = "v1"
    diy_data_aug: bool = False
    prob_blur_v0_views: float = 0.4
    prob_mask_v0_views: float = 0.4
    prob_mask_temporal_frames: float = 0.4
    
    # exp
    exp_name: str = "debug"
    exp_log_path: Optional[str] = None
    log_tensorboard: bool = False

    ### training
    # workspace
    workspace: str = './workspace'
    # resume
    resume: Optional[str] = None
    training_stage: str = "stage_1"
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 1.0
    # conf loss weight
    conf_version:str = "bce_masked"  # "bce_masked". "bce_weighted".  "bce_alpha_weighted". "bce_calibrated". "focal"
    conf_pos_weight: float = 2.0  # (for 'bce_calibrated'  "bce_weighted"
    
    conf_lambda : float = 1e-2
    conf_mse_lambda: float = 0.5 # (for 'bce_calibrated')
    conf_gamma: float = 2.0 # focal loss parameters
    conf_alpha: float = 0.25 # focal loss parameters
    # use confidence guassians
    use_conf_gs: bool = True
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = 'bf16'
    lr: float = 4e-5
    prob_grid_distortion: float = 0.5
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.5
    # number of gaussians per pixel
    gaussian_perpixel: int = 1
    # freeze the initlize L4GM
    freeze_L4GM: bool = False
    # scheduler
    diy_scheduler: str = "v1"

    ### testing
    # test image path
    test_path: Optional[str] = None
    input_path: Optional[str] = None
    ### misc
    # nvdiffrast backend setting
    force_cuda_rast: bool = False
    # render fancy video with gaussian scaling effect
    fancy_video: bool = False

    # 4D
    num_frames: int = 32
    num_clip_frames: int = 8
    recur_clip: bool = False
    use_temp_attn: bool = True
    shuffle_input: bool = True
    
    # s3
    sample_by_anim: bool = True

    # interp
    interpresume: Optional[str] = None
    interpolate_rate: int = 3

    # remove_bg
    remove_bg: bool = False

    # evaluation
    eval_azimuth: Optional[list[float]] = None
    use_azimuth: Optional[list[float]] = None
    eval_elevation: Optional[list[float]] = None
    eval_distance: Optional[list[float]] = None
    eval_size: Optional[Tuple[int, int]] = None
    eval_path: Optional[str] = None
    fps: int = 30
    seed: int = 123
    process_window: int = 32
    folder_id: int =0
    folder_before_id: int =0
    
    # viz_feature
    viz_tempoaral_attention: bool = False
    viz_atten_size: Optional[Tuple[int, int]] = (128,128)
    viz_memory_attn: bool = False
    save_gs: bool = False
    
    # encoder
    encode_patch_size: int = 14
    encode_num_view: int = 8
    encode_batch: int = 24
    encoder: str = "v0_1"
    encoder_feature_dim: int = 1024
    
    use_memory: bool = True
    l4gm_model: str = "v4"
    memory_size: int = 12
    memory_dim: int = 512
    slots_per_item: int = 64
    memory_heads: int = 8
    memory_layers: int = 2
    memory_mlp_ratio: float = 4.0
    memory_dropout: float = 0.0
    memory_lambda: float = 0.1
    memory_bank: str = "v1"
    injector_num_layers: int = 2
    
    unet_version: str = "v0" 
    memory_cross_upblock_id: int = 0
    
    infer_static_noise: float = 0

    seed: int = 123
    
    moga_lambda: float = 0.5
    group_loss_lambda:  float = 0.1
    use_moga_attn: bool = True
    moga_group: int = 32
    
    return_group_logit: bool = False


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()


config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    resume='pretrained/model_fp16_fixrot.safetensors',
)

config_doc['big_skip_mid'] = 'big model with higher resolution Gaussians'
config_defaults['big_skip_one'] = Options(
    input_size=256,
    up_channels=(1024,512, 256, 128), # one more decoder
    up_attention=(True, True, False, False),
    down_channels = (64,128, 256, 512, 1024),
    down_attention = (False, False, False, True, True),
    mid_attention = False,
    splat_size=128,
    output_size=512, # render & supervise Gaussians at a higher resolution.
    batch_size=1,
    num_views=8,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
    resume='pretrained/model_fp16_fixrot.safetensors',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
