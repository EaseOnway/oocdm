import torch.nn as nn
import torch
import utils
from core import ObjectOrientedEnv


class RLModel(nn.Module):

    def __init__(self, env: ObjectOrientedEnv, args: utils.Struct,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
