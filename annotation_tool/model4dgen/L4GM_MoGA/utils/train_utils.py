"""
训练过程中用到的辅助函数
"""


import os
import time
import json
import random
import contextlib
import psutil
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch import amp
from safetensors.torch import load_file
from accelerate import Accelerator
from PIL import Image
import imageio
import tyro
import kiui
from core.options_conf import AllConfigs
import copy
import shutil
import os
import sys


# 聚合所有 GPU/机器的结果（安全版本）
def safe_gather_mean(accelerator, tensor):
    # 确保 tensor 是 Tensor 类型（有时可能是 Python 数字）
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32, device=accelerator.device)
    # 确保 tensor 位于正确设备（GPU）
    if not tensor.is_cuda:
        tensor = tensor.to(accelerator.device)
    # 聚合并取均值
    return accelerator.gather_for_metrics(tensor).mean()

def adapt_batch_to_model(batch, model):
    """
    自动将 batch 中的张量转为模型的 dtype 和 device。
    非张量数据（如字符串、int）会被原样保留。
    """
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def to_device_dtype(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=model_device, dtype=model_dtype)
        elif isinstance(x, dict):
            return {k: to_device_dtype(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return [to_device_dtype(v) for v in x]
        else:
            return x

    return to_device_dtype(batch)

# ================================================================
# ⚙️ 分布式初始化（多机多卡）
# ================================================================
def init_distributed_env():
    """初始化多机多卡环境，兼容 accelerate 与 torchrun"""
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        print(f"🚀 Distributed initialized | Rank {rank}/{world_size} | Local rank {local_rank}")
    else:
        print("⚠️ No distributed environment detected, running single-node mode.")


def seed_everything(seed):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")
    
    
# =============== 冻结特定权重的函数 ============
def load_json_opt(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

def update_dataclass_from_dict(dc_instance, data_dict):
    """递归更新 dataclass 实例"""
    for k, v in data_dict.items():
        if hasattr(dc_instance, k):
            cur = getattr(dc_instance, k)
            if isinstance(v, dict) and hasattr(cur, "__dict__"):
                update_dataclass_from_dict(cur, v)
            else:
                setattr(dc_instance, k, v)
    return dc_instance

def get_user_provided_args() -> set:
    """从 sys.argv 中提取用户实际输入的参数名"""
    args = sys.argv[1:]
    keys = set()
    for arg in args:
        if arg.startswith("--"):
            keys.add(arg.lstrip("-").split("=")[0])
    return keys
