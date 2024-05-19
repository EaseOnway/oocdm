from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import numpy as np

from core import VType, Taskinfo, VarID
from utils.typings import ObjectTensors, NamedTensors, TransitionModel
import utils
import alg.functional as F

from .base import RLModel


class DistributionDecoder(nn.Module):
    def __init__(self, dim_in: int, vtype: VType,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self._vtype = vtype
        self._ptype = vtype.ptype
        self.sub_decoders = {
            key: nn.Linear(dim_in, dim_param, device=device, dtype=dtype)
            for key, dim_param in self._ptype.param_sizes.items()}
        for param, decoder in self.sub_decoders.items():
            self.add_module(f"{param} decoder", decoder)

    def forward(self, x: Tensor) -> NamedTensors:
        params = {k: decoder(x) for k, decoder in self.sub_decoders.items()}
        return params


class MLPModel(RLModel):
    
    class Args(utils.Struct):
        def __init__(self, dim_h1=256, dim_h2=256, dim_h3=256):
            self.dim_h1 = dim_h1
            self.dim_h2 = dim_h2
            self.dim_h3 = dim_h3

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        self.info = info = env.taskinfo
        self.dtype = dtype
        self.device = device

        d_in = sum(info.v(var).size for var in info.input_variables)

        self.f = nn.Sequential(
            nn.Linear(d_in, args.dim_h1, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(args.dim_h1, args.dim_h2, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(args.dim_h2, args.dim_h3, device=device, dtype=dtype),
            nn.ReLU(),
        )
        self.decoders = [DistributionDecoder(args.dim_h3, info.v(var), device, dtype)
                         for var in info.output_variables]
        for idx, decoder in zip(info.output_variables, self.decoders):
            self.add_module(f"{idx}_decoder", decoder)

    def forward(self, raw_attributes: ObjectTensors):
        envinfo = self.info.envinfo
        inputs = {
            clsname: {
                fieldname: envinfo.v(clsname, fieldname).raw2input(raw)
                for fieldname, raw in temp.items()}
            for clsname, temp in raw_attributes.items()}
        inputs = [vidx(inputs) for vidx in self.info.input_variables]
        x: Tensor = torch.cat(inputs, dim=1)
        x = self.f(x)

        variable_params: Dict[VarID, NamedTensors] = {}
        for i, decoder in enumerate(self.decoders):
            params = decoder.forward(x)
            variable_params[self.info.output_variables[i]] = params

        return variable_params

    def make_transition_model(self) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            variables = self.forward(raw_attributes)
            state = self.info.get_obj_distr(variables)
            return state
        return f
