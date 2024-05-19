from typing import Final, Dict, Optional,  List, TypeVar, Sequence, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import numpy as np


from core import VType, Taskinfo, VarID, CausalGraph
from utils.typings import ObjectTensors, NamedTensors, TransitionModel
import utils
import alg.functional as F

from .base import RLModel
from .modules import MultiLinear, Attention


def binary_gumbel_softmax(x: torch.Tensor, tau: Optional[float]):
    '''
    Arg:
    - x: (*batchshape)
    - tau: positive temperature parameter. 0 for hard sampling. None for maxlikelihood.
    Return:
    - Tensor(*batchshape): value in [0, 1].
    '''
    
    if tau is None:
        return (x > 0).to(dtype = x.dtype, device = x.device)

    g = -torch.log(-torch.log(torch.rand(2, *x.shape, device=x.device, dtype=x.dtype)))
    u1, u0 = x + g[1], g[0]
    cond = u1 > u0
    
    if tau <= 0:
        return cond.to(dtype = x.dtype, device = x.device)
    
    v1 = torch.where(cond, 0, u1 - u0)
    v0 = torch.where(cond, u0 - u1, 0)
    s1 = torch.exp(v1 / tau)
    s0 = torch.exp(v0 / tau)
    return s1 / (s0 + s1)



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


class VariableEncoder(nn.Module):

    def __init__(self, info: Taskinfo, dim_out: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.info = info
        self.encoders = {
            idx: nn.Sequential(
                nn.Linear(info.v(idx).size, dim_out, device=device, dtype=dtype),
                nn.ReLU(),
            )
            for idx in info.input_variables
        }
        for idx, module in self.encoders.items():
            self.add_module('%s_encoder' % idx, module)

    def forward(self, raw_attributes: ObjectTensors) -> Tensor:
        envinfo = self.info.envinfo
        inputs = {
            clsname: {
                fieldname: envinfo.v(clsname, fieldname).raw2input(raw)
                for fieldname, raw in temp.items()}
            for clsname, temp in raw_attributes.items()}
        inputs = {vidx: vidx(inputs) for vidx in self.info.input_variables}

        temp: List[Tensor] = []
        for vidx, x in inputs.items():
            encoder = self.encoders[vidx]
            temp.append(encoder.forward(x))
        
        return torch.stack(temp, dim=1)  # (batch, n_in, dim_v)
 

class TISCAModel(RLModel):
    
    class Args(utils.Struct):
        def __init__(self, dim_v = 4, dim_h1=64, dim_h2=32, dim_h3=32):
            self.dim_v = dim_v
            self.dim_h1 = dim_h1
            self.dim_h2 = dim_h2
            self.dim_h3 = dim_h3

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        self.info = info = env.taskinfo
        self.dtype = dtype
        self.device = device

        self.encoder = VariableEncoder(info,  args.dim_v, device, dtype)

        self.__d_in = args.dim_v * info.n_input_variable
        self.p = nn.Parameter(
            torch.randn(info.n_output_variable, info.n_input_variable,
                        device=device, dtype=dtype))
        self.f = nn.Sequential(
            MultiLinear.auto([info.n_output_variable], self.__d_in, args.dim_h1, dtype, device),
            nn.LeakyReLU(),
            MultiLinear.auto([info.n_output_variable], args.dim_h1, args.dim_h2, dtype, device),
            nn.LeakyReLU(),
            MultiLinear.auto([info.n_output_variable], args.dim_h2, args.dim_h3, dtype, device),
            nn.ReLU(),
        )
        self.decoders = [DistributionDecoder(args.dim_h3, info.v(var), device, dtype)
                         for var in info.output_variables]
        for idx, decoder in zip(info.output_variables, self.decoders):
            self.add_module(f"{idx}_decoder", decoder)
    
    def _mask(self, varenc: Tensor, tau: Optional[float]):
        # (batch, n_in, dim_v)
        b = varenc.shape[0]
        m = binary_gumbel_softmax(self.p.expand(b, -1, -1), tau)
        
        # m: (batch, n_out, n_in)
        x = varenc.unsqueeze(1) * m.unsqueeze(-1)  # (batch, n_out, n_in, dim_v)
        x = x.reshape(b, self.info.n_output_variable, self.__d_in)  # (batch, n_out, n_in * dim_v)
        return x, m

    def forward(self, raw_attributes: ObjectTensors, tau: Optional[float]):
        varenc = self.encoder.forward(raw_attributes)
        x, m = self._mask(varenc, tau)
        x = self.f(x)

        variable_params: Dict[VarID, NamedTensors] = {}
        for i, decoder in enumerate(self.decoders):
            params = decoder.forward(x[:, i])
            variable_params[self.info.output_variables[i]] = params
        
        return variable_params, m
    
    def extract_causal_graph(self, sampler: Literal['hard', 'max'] = 'max'):
        if sampler == 'max':
            m = (self.p > 0)
        elif sampler == 'hard':
            m = binary_gumbel_softmax(self.p, 0) > 0
        else:
            raise ValueError(sampler)

        g = CausalGraph(self.info)
        g.matrix[:] = m.detach().cpu().numpy()
        return g

    def make_transition_model(self, tau: Optional[float]) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            variables, _ = self.forward(raw_attributes, tau)
            state = self.info.get_obj_distr(variables)
            return state
        return f
