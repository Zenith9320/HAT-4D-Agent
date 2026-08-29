# Modified by HAT-4D; see the subtree MODIFICATIONS.md.

import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
)