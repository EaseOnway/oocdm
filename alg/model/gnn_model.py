from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import numpy as np

from core import VType, Taskinfo, EnvInfo, VarID, Role
from utils.typings import ObjectTensors, NamedTensors, TransitionModel
import utils
import alg.functional as F
from .modules.linears import MultiLinear

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


class ObjectEncoder(nn.Module):
    def __init__(self, taskinfo: Taskinfo, role: Role, 
                 dim_hidden: int, dim_out: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.info = taskinfo
        self.role = role
        for c in taskinfo.envinfo.classes:
            n = taskinfo.counts[c.name]
            d_in = sum(c.v(fname).size for fname in c.fieldnames(role))
            self.add_module(c.name + "_encoder",
                nn.Sequential(            
                    MultiLinear.auto([n], d_in, dim_hidden, dtype, device),
                    nn.LeakyReLU(),
                    MultiLinear.auto([n], dim_hidden, dim_out, dtype, device),
            ) )
        
        self.__device = device
        self.__dtype = dtype
        self.__dim_out = dim_out
    
    def get_encoder(self, cname: str):
        return self.get_submodule(cname + "_encoder")
    
    def forward(self, raw_attributes: ObjectTensors):
        envinfo = self.info.envinfo
        lis = []

        for c in envinfo.classes:
            try:
                temp = raw_attributes[c.name]
            except KeyError:
                continue
            
            fnames = c.fieldnames(self.role) # type: ignore
            if len(fnames) > 0:
                x = torch.cat([  # batch * n_obj * d_in
                    c.v(fieldname).raw2input(temp[fieldname])  # batch * n_obj * size
                    for fieldname in fnames], dim=-1)
                enc = self.get_encoder(c.name)
                lis.append(enc.forward(x))  # batch * n_obj * d_out
            else:
                shape = next(iter(temp.values())).shape[:2]
                lis.append(torch.zeros(*shape, self.__dim_out,
                                       dtype=self.__dtype,
                                       device=self.__device))  # batch * n_obj * d_out
            
        out = torch.cat(lis, dim=-2)  # batch * n_obj_all * d_out
        return out


class ObjectDecoder(nn.Module):
    def __init__(self, taskinfo: Taskinfo, dim_in: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.info = taskinfo

        self.decoders = [[
                DistributionDecoder(dim_in, taskinfo.v(var), device, dtype)
                for var in taskinfo.variables_of_obj(i_obj, role="state")
            ] for i_obj in range(taskinfo.n_allobj)]
        
        for i_obj, decoders_i in enumerate(self.decoders):
            for i_attr, decoder in enumerate(decoders_i):
                self.add_module(f"decoder_{i_obj}_{i_attr}", decoder)

    def forward(self, x: Tensor):
        out: Dict[VarID, NamedTensors] = {}

        # x: batch * n_obj_all * d_in
        for i_obj in range(self.info.n_allobj):
            x_obj = x[..., i_obj, :]  # batch * d_in
            decoders_obj = self.decoders[i_obj]
            variables_obj = self.info.variables_of_obj(i_obj, "state")
            for decoder, var in zip(decoders_obj, variables_obj):
                out[var] = decoder.forward(x_obj)  

        return out


class GNNModel(RLModel):

    class Args(utils.Struct):
        def __init__(self,
                dim_z=64, dim_h_z=64,
                dim_a=16, dim_h_a=16,
                dim_e=64,
                dim_h_edge=100, dim_h_node=100):
            
            self.dim_z = dim_z
            self.dim_h_z = dim_h_z
            self.dim_a = dim_a
            self.dim_h_a = dim_h_a
            self.dim_e = dim_e
            self.dim_h_edge = dim_h_edge
            self.dim_h_node = dim_h_node


    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        self.info = taskinfo = env.taskinfo
        self.z_encoder = ObjectEncoder(taskinfo, "state", args.dim_h_z, args.dim_z, device, dtype)
        self.a_encoder = ObjectEncoder(taskinfo, "action", args.dim_h_a, args.dim_a, device, dtype)
        
        self.f_edge = nn.Sequential(
            nn.Linear(2 * args.dim_z, args.dim_h_edge, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(args.dim_h_edge, args.dim_e, device=device, dtype=dtype),
        )
        self.f_node = nn.Sequential(
            nn.Linear(args.dim_z + args.dim_a + args.dim_e, args.dim_h_node,
                      device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(args.dim_h_node, args.dim_z, device=device, dtype=dtype),
        )
        self.decoder = ObjectDecoder(taskinfo, args.dim_z, device=device, dtype=dtype)

        self.__mask = torch.eye(taskinfo.n_allobj, device=device, dtype=torch.bool).unsqueeze(-1)

    def forward(self, raw_attributes: ObjectTensors):
        z = self.z_encoder.forward(raw_attributes)  # batch * n_obj_all * d_z
        a = self.a_encoder.forward(raw_attributes)  # batch * n_obj_all * d_a

        n = z.shape[-2]

        zi = z.unsqueeze(-2).expand((-1, -1, n, -1))
        zj = zi.transpose(-2, -3)
        zizj = torch.cat((zi, zj), dim=-1)  # batch * n_obj_all * n_obj_all * d_e

        e = self.f_edge.forward(zizj)
        e = torch.masked_fill(e, self.__mask, 0)
        e = torch.sum(e, dim=-3)  # batch * n_obj_all * d_e

        x = torch.cat((z, a, e), dim=-1)
        dz = self.f_node.forward(x)  # batch * n_obj_all * d_z
        next_z = z + dz

        out = self.decoder.forward(next_z)
        return out

    def make_transition_model(self) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            variables = self.forward(raw_attributes)
            state = self.info.get_obj_distr(variables)
            return state
        return f
