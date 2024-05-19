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


_T = TypeVar('_T')
def _index_dict(seq: Sequence[_T]) -> Dict[_T, int]:
    return {x: i for i, x in enumerate(seq)}


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

    def __init__(self, info: Taskinfo, dim_hidden: int, dim_out: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.info = info
        self.encoders = {
            idx: nn.Sequential(
                nn.Linear(info.v(idx).size, dim_hidden, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_out, device=device, dtype=dtype),
                nn.ReLU()
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
        
        return torch.stack(temp, dim=1)


class VariableInferer(DistributionDecoder):
    def __init__(self, taskinfo: Taskinfo, j: int,
                 dim_in, dim_gru: int, dim_decoder_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__(dim_decoder_hidden, 
                         taskinfo.v(taskinfo.output_variables[j]), device, dtype)

        self.dim_gru = dim_gru

        self.gru = nn.GRU(dim_in, dim_gru, batch_first=True,
                          bidirectional=True, device=device, dtype=dtype)
        self.f = nn.Linear(2 * dim_gru, dim_decoder_hidden, device=device, dtype=dtype)
        self.h0 = nn.parameter.Parameter(torch.zeros(2, dim_gru, device=device, dtype=dtype))
    
    def forward(self, encoding: Tensor):
        # encoding: batchsize * n_parents * dim_enc
        b = encoding.shape[0]
        h0 = self.h0.unsqueeze(1).expand(2, b, self.dim_gru).contiguous()
        if encoding.shape[1] > 0:
            _, h = self.gru.forward(encoding, h0)
        else:
            h = h0

        # h: (2, batchsize, dim_gru)
        assert h.shape == (2, b, self.dim_gru)
        h = h.transpose(0, 1).reshape(b, 2 * self.dim_gru)
        
        x = torch.relu(self.f.forward(h))
        return super().forward(x)


class GRUModel(RLModel):
    
    class Args(utils.Struct):
        def __init__(self,
                dim_variable_encoding: int = 64,
                dim_variable_encoder_hidden: int = 64,
                dim_gru_hidden: int = 64,
                dim_decoder_hidden: int = 32):

            self.dim_variable_encoding = dim_variable_encoding
            self.dim_variable_encoder_hidden = dim_variable_encoder_hidden
            self.dim_gru_hidden = dim_gru_hidden
            self.dim_decoder_hidden = dim_decoder_hidden

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        self.info = info = env.taskinfo
        self.device = device
        self.variable_encoder = VariableEncoder(
            info, args.dim_variable_encoder_hidden, args.dim_variable_encoding,
            device, dtype)
        self.inferers = [VariableInferer(info, j,
                                         args.dim_variable_encoding,
                                         args.dim_gru_hidden,
                                         args.dim_decoder_hidden,
                                         device, dtype)
                         for j in range(info.n_output_variable)]
        for idx, inferer in zip(info.output_variables, self.inferers):
            self.add_module(f"{idx}_inferer", inferer)

    def infer(self, encoding: Tensor,
              mask: Optional[Tensor] = None):
        
        out: Dict[VarID, NamedTensors] = {}
        for j, var in enumerate(self.info.output_variables):
            if mask is not None:
                parents = mask[j]
                x = encoding[:, parents, :]
            else:
                x = encoding
            
            out[var] = self.inferers[j].forward(x)
        return out

    def forward(self, raw_attributes: ObjectTensors,
                causal_graph: Optional[CausalGraph] = None):
        
        if causal_graph is None:
            mask = None
        else:
            mask = torch.tensor(causal_graph.matrix, dtype=torch.bool, device=self.device)

        enc = self.variable_encoder.forward(raw_attributes)
        return self.infer(enc, mask)
    
    def make_transition_model(self, causal_graph: Optional[CausalGraph] = None) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            variables = self.forward(raw_attributes, causal_graph)
            state = self.info.get_obj_distr(variables)
            return state
        return f
