from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def _iter_tuple(*ns: int):
    if len(ns) == 0:
        assert False
    elif len(ns) == 1:
        for i in range(ns[0]):
            yield (i,)
    else:
        for i in range(ns[0]):
            for js in _iter_tuple(*ns[1:]):
                yield (i,) + js


class MultiLinear(nn.Module):
    def __init__(self, size: Sequence[int], dim_in: int, dim_out: int,
                 dtype: torch.dtype, device: torch.device, bias=True):
        '''
        size: a sequence of integars indicate the size of linear transforms.
            1 for shared transform.
        '''

        super().__init__()
        
        self.size = tuple(size)
        self.__indices = list(_iter_tuple(*size))
        self.__n = len(self.__indices)

        self.fs = [
            nn.Linear(dim_in, dim_out, bias=bias, device=device, dtype=dtype)
            for _ in range(self.__n)
        ]
        for index, f in zip(self.__indices, self.fs):
            self.add_module('sub_module' + str(index), f)
        
    def forward(self, x: Tensor):
        '''
        input: [..., *size, dim_in]
        output: [..., *size, dim_out]
        '''
        
        temp = []
        for i, index in enumerate(self.__indices):
            temp.append(
                self.fs[i].forward(x[(..., *index, slice(None))])
        )
        y = torch.stack(temp, dim=-2)
        y = y.view(*y.shape[:-2], *self.size, y.shape[-1])
        return y
    
    @staticmethod
    def auto(size: Sequence[int], dim_in: int, dim_out: int,
             dtype: torch.dtype, device: torch.device, bias=True) -> 'MultiLinear':
        if dim_in * dim_out * np.prod(size) <= 1024:  # type: ignore
            return MultiLinearParallel(size, dim_in, dim_out, dtype, device, bias)
        else:
            return MultiLinear(size, dim_in, dim_out, dtype, device, bias)


class MultiLinearParallel(MultiLinear):
    def __init__(self, size: Sequence[int], dim_in: int, dim_out: int,
                 dtype: torch.dtype, device: torch.device, bias=True):
        '''
        size: a sequence of integars indicate the size of linear transforms.
            1 for shared transform.
        '''

        nn.Module.__init__(self)
        
        self.size = tuple(size)
        self.weight = nn.parameter.Parameter(torch.empty(
            *size, dim_in, dim_out,
            device=device, dtype=dtype))
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(
                *size, dim_out,
                device=device, dtype=dtype))
        else:
            self.bias = None
    
    def forward(self, x: Tensor):
        '''
        input: [..., *size, dim_in]
        output: [..., *size, dim_out]
        '''
        
        # x: ..., *size, dim_in
        x = x.unsqueeze(-2)  # ..., *size, 1, dim_in
        w = self.weight  # *size, dim_in, dim_out
        b = self.bias  # *size, dim_out

        y = torch.matmul(x, w)  # ..., *size, 1, dim_out
        y = y.squeeze(-2)  # ..., *size, dim_out
        
        if b is not None:
            y = y + b
        return y



class HeterogenousLinear(nn.Linear):
    def __init__(self, in_features: Sequence[int], out_feature: int,
                 dtype: torch.dtype, device: torch.device, bias=True):

        self.total_infeature = sum(in_features)
        self.n_input = n_in = len(in_features)
        if n_in == 0:
            raise ValueError("needs at least one input.")
        
        super().__init__(self.total_infeature, out_feature,
                         bias=False, dtype=dtype, device=device)

        if bias:
            self.hetero_bias = nn.parameter.Parameter(torch.empty(
                n_in, out_feature, device=device, dtype=dtype))
        else:
            self.hetero_bias = None
        
        
    def forward(self, xs: Sequence[Tensor]):
        # xs[i]: (*batchshape, in_features[i])
        assert len(xs) == self.n_input
        batchshape = xs[0].shape[:-1]

        # xs[i]: (b, in_features[i])     
        xs = [x.reshape(-1, x.shape[-1]) for x in xs]
        
        # (n_input * b, sum_in_features)
        x: Tensor = torch.block_diag(*xs)  

        n_input, d = self.n_input, self.total_infeature

        # (*batchshape, n_input, total_infeature)
        x = x.reshape(n_input, -1, d).transpose(0, 1).reshape(*batchshape, n_input, d)
        
        # (*batchshape, n_input, out_feature)
        y = super().forward(x)

        if self.hetero_bias is not None:
            y = y + self.hetero_bias
        
        return y
