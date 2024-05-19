from typing import Final, Dict, Optional, Iterable

import torch
import torch.nn as nn
import numpy as np

from utils.typings import NamedTensors, ObjectTensors

class RewardPredictor(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int,
                 n_head: int, dim_decoder_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        if dim_v % n_head != 0:
            new_dim_v = (dim_v // n_head + 1) * n_head
            print(f"dim_v={dim_v} is not a multiple of n_head={n_head}. "
                  f"Use dim_v = {new_dim_v} instead.")
            dim_v = new_dim_v
            assert dim_v % n_head == 0
        
        self.n_head: Final = n_head
        self.q = nn.parameter.Parameter(
            torch.randn(n_head, dim_k, dtype=dtype, device=device))
        self.w_k = nn.parameter.Parameter(
            torch.randn(n_head, dim_in, dim_k, dtype=dtype, device=device))
        self.w_v = nn.parameter.Parameter(
            torch.randn(n_head, dim_in, dim_v//n_head, dtype=dtype, device=device))
        self.decoder = nn.Sequential(
            nn.Linear(dim_v, dim_decoder_hidden, dtype=dtype, device=device),
            nn.LeakyReLU(),
            nn.Linear(dim_decoder_hidden, 1, device=device, dtype=dtype),
        )
        self._sqrt_dk = np.sqrt(dim_k)

    def __multihead_attn(self, objects: torch.Tensor, obj_mask: Optional[torch.Tensor]):
        # objects: batchsize * n_obj * dim_in
        # obj_mask: batchsize * n_obj
        temp = []
        batchsize, n_obj, dim_in = objects.shape
        n_head = self.n_head

        # objects: batchsize * n_obj * n_head * 1 * dim_in
        objects = objects.unsqueeze(2).expand(batchsize, n_obj, n_head, dim_in).unsqueeze(3)
        w_k = self.w_k  # n_head * dim_in * dim_k
        w_v = self.w_v  # n_head * dim_in * (dim_v / n_head)

        k = torch.matmul(objects, w_k).squeeze(3)  # batchsize * n_obj * n_head * dim_k
        v = torch.matmul(objects, w_v).squeeze(3)  # batchsize * n_obj * n_head * (dim_v / n_head)

        q = self.q  # n_head * dim_q
        a = torch.sum(q * k, dim = -1) / self._sqrt_dk  # batchsize * n_obj * n_head
        a = torch.exp(a - torch.amax(a, dim=1, keepdim=True))
        
        # apply mask
        if obj_mask is not None:
            m = obj_mask.unsqueeze(2)  # batchsize * n_obj * 1
            a = torch.masked_fill(a, torch.logical_not(m), 0.)  # batchsize * n_obj * n_head
        
        # normalize
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-6)  # batchsize * n_obj * n_head
        
        # weighted sum
        v_ = torch.sum(v * a.unsqueeze(3), dim=1)  # batchsize * n_head * (dim_v / n_head)

        # concat heads
        v_ = torch.flatten(v_, start_dim=1)  # batchsize * dim_v

        return v_

    def forward(self, encodings: NamedTensors,
                obj_masks: Optional[NamedTensors] = None):
        
        # combine all objects
        temp = []  
        for clsname, e in encodings.items():
            temp.append(e)
        objects = torch.cat(temp, dim=1)  # batchsize * n_obj * dim_in

        # combine masks
        if obj_masks is None:
            obj_masks_ = None
        else:
            temp = []
            for clsname, e in encodings.items():
                m = obj_masks[clsname]
                temp.append(m)
            obj_masks_ = torch.cat(temp, dim=1)  # batchsize * n_obj

        x = self.__multihead_attn(objects, obj_masks_)  # batchsize * dim_v

        out: torch.Tensor = self.decoder.forward(x)  # batchsize * 1
        out = out.squeeze(1)  # batchsize
        return out
