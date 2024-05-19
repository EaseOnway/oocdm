from typing import Final, Dict, Optional, Iterable

import torch
import torch.nn as nn
import numpy as np

from core import VType, DType, EnvObjClass, ObjectOrientedEnv
from utils.typings import NamedTensors, ObjectTensors


class DistributionDecoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, vtype: VType,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self._vtype = vtype
        self._ptype = vtype.ptype

        self.sub_decoders = {key: nn.Sequential(
            nn.Linear(dim_in, dim_hidden, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_param, device=device, dtype=dtype),
        ) for key, dim_param in self._ptype.param_sizes.items()}
        for param, decoder in self.sub_decoders.items():
            self.add_module(f"{param} decoder", decoder)

    def forward(self, x: torch.Tensor):
        params = {k: decoder(x) for k, decoder in self.sub_decoders.items()}
        out = self._ptype(**params)
        return out


class RelationalInferer(nn.Module):
    def __init__(self, vtype: VType,
                 dim_in: int, dim_k: int, dim_v: int,
                 dim_decoder_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        # self.classes: Final = env.classes
        self.vtype = vtype

        self.kmap = nn.Linear(dim_in, dim_k, device=device, dtype=dtype)
        self.qmap = nn.Linear(dim_in, dim_k, device=device, dtype=dtype)
        self.vmap = nn.Linear(dim_in, dim_v, device=device, dtype=dtype)
        self.decoder = DistributionDecoder(dim_in + dim_v, dim_decoder_hidden, self.vtype,
                                           device, dtype)
        # self.activate = nn.LeakyReLU()
        self._sqrt_dk = np.sqrt(dim_k)

    def forward(self, clsname: str, local_encoding: torch.Tensor,
                global_encodings: NamedTensors,
                obj_masks: Optional[NamedTensors] = None):
        
        # combine all objects
        temp = []
        temp.append(global_encodings[clsname])        
        for clsname_, e in global_encodings.items():
            if clsname_ != clsname:
                temp.append(e)
        g = torch.cat(temp, dim=1)
        n_obj_cls = local_encoding.shape[1]
        n_obj = g.shape[1]

        # compute query, key and value
        q: torch.Tensor = self.qmap(local_encoding)  # batchsize * n_obj_cls * dim_k
        k: torch.Tensor = self.kmap(g)  # batchsize * n_obj * dim_k
        v: torch.Tensor = self.vmap(g)  # batchsize * n_obj * dim_v

        # attention
        a = torch.matmul(q, k.transpose(1, 2)) / self._sqrt_dk  # batchsize * n_obj_cls * n_obj
        a = torch.exp(a - torch.amax(a, dim=-1, keepdim=True))

        # mask
        if obj_masks is not None:
            temp = []
            temp.append(obj_masks[clsname])
            for clsname_, obj_mask in obj_masks.items():
                if clsname_ != clsname:
                    temp.append(obj_mask)
            m = torch.cat(temp, dim=1)  # batchsize * n_obj
            m = m.unsqueeze(1).repeat(1, n_obj_cls, 1)  # batchsize * n_obj_cls * n_obj
        else:
            m = torch.ones_like(a, dtype=torch.bool)
        m[:, range(n_obj_cls), range(n_obj_cls)] = False  # no self-attention

        # apply the mask
        a = torch.masked_fill(a, torch.logical_not(m), 0.)

        # normalize
        a = a / (torch.sum(a, dim=-1, keepdim=True) + 1e-6)  # batchsize * n_obj_cls * n_obj

        # aggregate
        v_ = torch.matmul(a, v)  # batchsize * n_obj_cls * dim_v

        x = torch.cat((v_, local_encoding), dim=2)  # batchsize * n_obj_cls * (dim_v + dim_in)
        out = self.decoder.forward(x)
        return out

 
# self.sub_decoders = {key: nn.Sequential(
#     nn.LeakyReLU(),
#     nn.Linear(dim_in, dh_dec, **self.torchargs),
#     nn.PReLU(dh_dec, **self.torchargs),
#     nn.Linear(dh_dec, dim, **self.torchargs),
# ) for key, dim in self._ptype.param_dims.items()}
# for param, decoder in self.sub_decoders.items():
#     self.add_module(f"{param} decoder", decoder)
