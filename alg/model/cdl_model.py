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


class Kernel(nn.Module):
    def __init__(self, info: Taskinfo, device: torch.device):
        super().__init__()
        self.device = device
        self.info = info
        self._n_out = info.n_output_variable
        self._n_in = info.n_input_variable

    def full(self, x: Tensor):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''

        x = torch.amax(x, dim=1, keepdim=True)
        return x.expand(-1, self._n_out, -1)
    
    def drop(self, x: Tensor, i_drop: int):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''
        mask = torch.ones(self._n_in, dtype=torch.bool, device=self.device)
        mask[i_drop] = False
        x = F.masked_retain(x, mask.unsqueeze(-1), value=0)
        x = torch.amax(x, dim=1, keepdim=True)
        return x.expand(-1, self._n_out, -1)
    
    def random_drop(self, x: Tensor):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''
        b = x.shape[0]
        to_drop: Tensor = nn.functional.one_hot(
            torch.randint(self._n_in, (b, self._n_out)), self._n_in).\
            to(device=self.device, dtype=torch.bool)
        mask = to_drop.logical_not()  # b * n_out * n_in
        x = F.masked_retain(x.unsqueeze(1), mask.unsqueeze(-1), value=0)
        x = torch.amax(x, dim=2)
        return x
    
    def graph(self, x: Tensor, g: CausalGraph):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''
        mask = torch.tensor(g.matrix, device=self.device)  # n_out * n_in
        x = x.unsqueeze(dim=1)
        x = F.masked_retain(x, mask.unsqueeze(-1), value=0)  # b * n_out * n_in * d
        x = torch.amax(x, dim=2)
        return x
    

class AttnKernel(Kernel):
    def __init__(self, dim_in: int, dim_k: int,
                 n_head: Optional[int], info: Taskinfo,
                 device: torch.device, dtype: torch.dtype):
        super().__init__(info, device)
        self.attn = Attention(device, dtype, n_head, None, (dim_in, dim_k), None)
        self.qs = nn.Parameter(torch.randn(self._n_out, dim_k, device=device, dtype=dtype))

    def _kernel(self, x: Tensor, mask: Optional[torch.Tensor]):
        '''
        x: Tensor(batch_size, n_in, dim_variable_enc)
        mask: None, or can be braodcast to Tensor(batch_size, n_out, n_in)
        '''
        return self.attn.forward(self.qs, x, x, mask)

    def full(self, x: Tensor):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_v)
        '''

        return self._kernel(x, None)


    def drop(self, x: Tensor, i_drop: int):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''

        mask = torch.ones(self._n_in, dtype=torch.bool, device=self.device)
        mask[i_drop] = False
        
        return self._kernel(x, mask)
    
    def random_drop(self, x: Tensor):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''
        b = x.shape[0]
        to_drop: Tensor = nn.functional.one_hot(
            torch.randint(self._n_in, (b, self._n_out)), self._n_in).\
            to(device=self.device, dtype=torch.bool)
        mask = to_drop.logical_not()  # b * n_out * n_in
        
        return self._kernel(x, mask)
    
    def graph(self, x: Tensor, g: CausalGraph):
        '''
        Args:
            x: Tensor(batch_size, n_in, dim_variable_enc)
        Returns:
            Tensor: (batch_size, n_out, dim_variable_enc)
        '''
        mask = torch.tensor(g.matrix, device=self.device)  # n_out * n_in
        return self._kernel(x, mask)


class Inferer(nn.Module):
    def __init__(self, info: Taskinfo, dim_in: int, dim_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.info = info
        self.f = nn.Sequential(
            MultiLinear.auto([info.n_output_variable], dim_in, dim_hidden, dtype, device),
            nn.ReLU(),
        )
        self.decoders = [DistributionDecoder(dim_hidden, info.v(idx), device, dtype)
                         for idx in info.output_variables]
        for idx, decoder in zip(info.output_variables, self.decoders):
            self.add_module(f"{idx}_decoder", decoder)

    def forward(self, x: Tensor):
        '''
        x: Tensor(batch_size, n_variable_out, dim_variable_enc)
        '''
    
        x = self.f(x)

        variable_params: Dict[VarID, NamedTensors] = {}
        for i, decoder in enumerate(self.decoders):
            params = decoder.forward(x[:, i])
            variable_params[self.info.output_variables[i]] = params
        
        return variable_params
 

class CDLModel(RLModel):
    
    class Args(utils.Struct):
        def __init__(self, dim_variable_encoding: int = 64,
                     dim_variable_encoder_hidden: int = 64,
                     dim_inferer_hidden: int = 32,
                     kernel: Literal['max', 'attn'] = 'max',
                     attn_dim_k: int = 64,
                     attn_nhead: Optional[int] = 2,
                     ):
            self.dim_variable_encoding = dim_variable_encoding
            self.dim_variable_encoder_hidden = dim_variable_encoder_hidden
            self.dim_inferer_hidden = dim_inferer_hidden
            self.kernel = kernel
            self.attn_dim_k = attn_dim_k
            self.attn_nhead = attn_nhead

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)
        
        dim_variable_encoding = args.dim_variable_encoding
        dim_variable_encoder_hidden = args.dim_variable_encoder_hidden
        dim_inferer_hidden = args.dim_inferer_hidden
        self.info = info = env.taskinfo

        self.variable_encoder = VariableEncoder(
            info, dim_variable_encoder_hidden, dim_variable_encoding, device, dtype)
        if args.kernel == 'max':
            self.kernel = Kernel(info, device)
        elif args.kernel == 'attn':
            self.kernel = AttnKernel(
                dim_variable_encoding, args.attn_dim_k, args.attn_nhead,
                info, device, dtype)
        self.inferer = Inferer(
            info, dim_variable_encoding, dim_inferer_hidden, device, dtype)

    def infer(self, encoding: Tensor,
              mask: Literal['full', 'drop', 'graph', 'random'],
              i_drop: Optional[int] = None,
              causal_graph: Optional[CausalGraph] = None):
        
        if mask == 'full':
            x = self.kernel.full(encoding)
        elif mask == 'drop':
            assert i_drop is not None
            x = self.kernel.drop(encoding, i_drop)
        elif mask == 'graph':
            assert causal_graph is not None
            x = self.kernel.graph(encoding, causal_graph)
        elif mask == 'random':
            x = self.kernel.random_drop(encoding)
        else:
            assert False
        
        return self.inferer.forward(x)

    def forward(self, raw_attributes: ObjectTensors,
                mask: Literal['full', 'drop', 'graph', 'random'],
                i_drop: Optional[int] = None,
                causal_graph: Optional[CausalGraph] = None):
        
        enc = self.variable_encoder.forward(raw_attributes)
        return self.infer(enc, mask, i_drop, causal_graph)
    
    def make_transition_model(
            self, mask: Literal['full', 'drop', 'graph', 'random'],
            i_drop: Optional[int] = None,
            causal_graph: Optional[CausalGraph] = None) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            variables = self.forward(raw_attributes, mask, i_drop, causal_graph)
            state = self.info.get_obj_distr(variables)
            return state
        return f
