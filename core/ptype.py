from typing import Any, Dict, List, Optional, Set, Tuple, final
import numpy as np
import abc
import torch.distributions as D
import torch
from torch import Tensor

from utils import torchutils
from utils.typings import ShapeLike, Shape
import utils


EPSILON = 1e-5


class PType(abc.ABC):
    '''Parameterized probability model
    '''

    def __init__(self, **param_sizes: int):
        self.__param_sizes = param_sizes
        self.__param_keys = tuple(param_sizes.keys())
    
    @property
    @final
    def param_sizes(self):
        return self.__param_sizes
    
    @property
    def param_names(self):
        return self.__param_keys

    @abc.abstractmethod
    def __call__(self, **params: Tensor) -> D.Distribution:
        raise NotImplementedError
    
    @abc.abstractmethod
    def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def default_params(self, device: torch.device, dtype: torch.dtype,
                       batchshape: ShapeLike) -> Dict[str, Tensor]:
        raise NotImplementedError


class Normal(PType):
    def __init__(self, size: int, scale: Optional[float] = 1.0):
        if scale is None:
            super().__init__(mean=size, scale=size)
        else:
            super().__init__(mean=size)
        
        self.size = size
        self.scale = scale
        
    def __call__(self, mean: Tensor, scale: Optional[Tensor] = None):
        if self.scale is not None:
            _scale = self.scale
        else:
            assert scale is not None
            _scale = torch.nn.functional.softplus(scale) + EPSILON

        return D.Normal(mean, _scale)

    def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
        scale, mean = torch.std_mean(samples, dim=dims)
        scale = torchutils.inv_softplus(scale - EPSILON)
        
        if self.scale is not None:
            return {'mean': mean}
        else:
            return {'mean': mean, 'scale': scale}

    def default_params(self, device: torch.device, dtype: torch.dtype,
                       batchshape: ShapeLike) -> Dict[str, Tensor]:
        size = utils.Shaping.as_shape(batchshape) + (self.size,)
        return {
            'mean': torch.zeros(size, device=device, dtype=dtype),
            'scale': torchutils.inv_softplus(torch.ones(size, device=device, dtype=dtype))
        }


class TanhNormal(Normal):
    def __init__(self, size: int, scale: Optional[float] = 1.0,
                 _range: Optional[Tuple[Any, Any]] = None):
        super().__init__(size, scale)
        
        self.scale = scale

        self._ranged = False
        if _range is not None:
            self._ranged = True
            low = torch.tensor(_range[0], dtype=torch.float)
            high = torch.tensor(_range[1], dtype=torch.float)
            self._rad = (high - low)/ 2
            self._mid = (high + low)/ 2

    def __call__(self, mean: Tensor, scale: Optional[Tensor] = None):
        mean = torch.tanh(mean)
        if self._ranged:
            self._rad = self._rad.to(mean.device, mean.dtype)
            self._mid = self._mid.to(mean.device, mean.dtype)
            mean = self._mid + mean * self._rad
        return super().__call__(mean, scale)
    
    def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
        params = super().estimate_params(samples, dims)
        mean = params['mean']
        if self._ranged:
            self._rad = self._rad.to(mean.device, mean.dtype)
            self._mid = self._mid.to(mean.device, mean.dtype)
            mean = (mean - self._mid) / self._rad
        mean = torch.arctanh(mean)
        params['mean'] = mean
        return params


class SoftplusNormal(Normal):
    def __init__(self, size: int, scale: Optional[float] = 1.0,
                 start: Optional[Any] = 0., negative=False):
        super().__init__(size, scale)
        
        self.scale = scale

        self._start = torch.tensor(start, dtype=torch.float)
        self._negative = negative

    def __call__(self, mean: Tensor, scale: Optional[Tensor] = None):
        self._start = self._start.to(mean.device, mean.dtype)
        if self._negative:
            mean = self._start - torch.nn.functional.softplus(mean)
        else:
            mean = self._start + torch.nn.functional.softplus(mean)
        return super().__call__(mean, scale)
    
    def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
        params = super().estimate_params(samples, dims)
        mean = params['mean']
        
        self._start = self._start.to(mean.device, mean.dtype)
        if self._negative:
            mean = torchutils.inv_softplus(self._start - mean)
        else:
            mean = torchutils.inv_softplus(mean - self._start)

        params['mean'] = mean
        return params


class Categorical(PType):
    def __init__(self, k: int):
        super().__init__(weights=k)
        self.__k = k
        
    def __call__(self, weights: Tensor):
        weights = torch.softmax(weights, dim=-1) + EPSILON / self.__k
        return D.Categorical(weights)

    def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
        # sample: (*batchshape)
        total = 1
        for d in utils.Shaping.as_shape(dims):
            total *= samples.shape[d]
        onehot = torch.nn.functional.one_hot(samples, self.__k)  # (*batchshape, k)
        weights = torch.sum(onehot, dim=dims) / total
        weights = torch.clamp(torch.log(weights), min=-9999)
        return {'weights': weights}
    
    def default_params(self, device: torch.device, dtype: torch.dtype,
                       batchshape: ShapeLike) -> Dict[str, Tensor]:
        size = utils.Shaping.as_shape(batchshape) + (self.__k,)
        return {'weights': torch.ones(size, dtype=dtype, device=device)}

# class Beta(PType):
#     def __init__(self, size: int):
#         super().__init__(alpha=size, beta=size)
#         
#     def __call__(self, alpha: Tensor, beta: Tensor):
#         a = 1.44 * torch.nn.functional.softplus(alpha) + EPSILON
#         b = 1.44 * torch.nn.functional.softplus(beta) + EPSILON
#         return D.Beta(a, b)
# 
#     def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
#         raise NotImplementedError
# 
# 
# class Bernoulli(PType):
#     def __init__(self):
#         super().__init__(logit=1)
#         
#     def __call__(self, logit: Tensor):
#         p = torch.sigmoid(logit)
#         return D.Bernoulli(p)
# 
#     def estimate_params(self, samples: Tensor, dims: ShapeLike) -> Dict[str, Tensor]:
#         raise NotImplementedError