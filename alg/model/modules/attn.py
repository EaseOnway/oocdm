from typing import Final, Optional, Tuple

import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.parameter import Parameter


def attention(q: Tensor, k: Tensor, v: Tensor,
              n_head: Optional[int] = None,
              mask: Optional[Tensor] = None):
    '''
    Args:
        q: (*batch, nq, dim_k)
        k: (*batch, nk, dim_k)
        v: (*batch, nk, dim_v)
        mask: can be broadcasted to (*batch, nq, nk)

    return:
        Tensor: (*batch, nq, dim_v)
    '''
    nk, dk = k.shape[-2:]
    dv = v.shape[-1]
    nq = q.shape[-2]

    if mask is not None and mask.ndim == 1:
        mask = mask.expand(nq, nk)

    if n_head is not None:
        if dk % n_head != 0 or dv % n_head != 0:
            raise ValueError(f"dimension of key ({dk}) and value ({dv}) "
                             f"should both be a multiple of n_head ({n_head})")
        q = q.view(*q.shape[:-1], n_head, dk//n_head).transpose(-2, -3)  # (*batch, n_head, nq, dk//n_head)
        k = k.view(*k.shape[:-1], n_head, dk//n_head).transpose(-2, -3)  # (*batch, n_head, nk, dk//n_head)
        v = v.view(*v.shape[:-1], n_head, dv//n_head).transpose(-2, -3)  # (*batch, n_head, nk, dv//n_head)

        if mask is not None:
            mask = mask.unsqueeze(-3)  # (*batch, 1, nq, dim_v)

    a = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(dk)  # (*batch, [n_head], nq, nk)
    if mask is not None:  # mask off non-existing objects.
        a = torch.masked_fill(a, torch.logical_not(mask), -np.inf)
    b = torch.amax(a, dim=-1, keepdim=True)  # base
    b = torch.masked_fill(b, torch.isinf(b), 0.)
    a = torch.exp(a - b)  # subtract b to prevent inf
    a = a / (torch.sum(a, dim=-1, keepdim=True) + 1e-6)  # (*batch, [n_head], nq, nk)
    
    # v: (*batch, nv, [n_head], dv//n_head)
    if n_head is not None:
        x = torch.matmul(a, v)  # (*batch, n_head, nq, dv//n_head)
        x = x.transpose(-2, -3).flatten(-2)  # (*batch, nq, dv)
    else:
        x = torch.matmul(a, v)  # (*batch, nq, dv)

    return x


class Attention(nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype,
                 n_head: Optional[int] = None,
                 transform_q: Optional[Tuple[int, int]] = None,
                 transform_k: Optional[Tuple[int, int]] = None,
                 transform_v: Optional[Tuple[int, int]] = None):
        super().__init__()
        
        self.n_head = n_head
        
        def make_f(d_in, d_out):
            return nn.Linear(d_in, d_out, device=device, dtype=dtype)
        
        self.fq = None if (transform_q is None) else make_f(*transform_q)
        self.fk = None if (transform_k is None) else make_f(*transform_k)
        self.fv = None if (transform_v is None) else make_f(*transform_v)
    
    def __apply_transform(self, x: Tensor, f: Optional[nn.Linear]):
        # x: (batch, len, din)
        if f is None:  # no transform
            return x
        else:
            return f.forward(x)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        '''
        Args:
            q: (*batch, nq, dim_q)
            k: (*batch, nk, dim_k)
            v: (*batch, nk, dim_v)
            mask: can be broadcasted to (*batch, nq, nk)

        return:
            Tensor: (*batch, nq, dim_v)
        '''

        q = self.__apply_transform(q, self.fq)
        k = self.__apply_transform(k, self.fk)
        v = self.__apply_transform(v, self.fv)

        out = attention(q, k, v, self.n_head, mask)
        return out
