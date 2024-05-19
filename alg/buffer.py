from typing import Dict, Any, Final, Iterable, Tuple, List, Union

import numpy as np
import torch

from core import EnvObjClass, EnvInfo, DType
from collections import deque
from utils.typings import NamedArrays, NamedTensors, ObjectArrays, ObjectTensors
import utils


_IndexType = Union[List[int], np.ndarray, slice]


class _ClassDataCache:

    def __init__(self, size: int, c: EnvObjClass):
        self.c: Final = c
        self.cache_size: Final = 2 * size
        self.max_n_obj = 0

        self.mask: np.ndarray = np.zeros((self.cache_size, self.max_n_obj), bool)
        self.data: NamedArrays = {}
        self.data_next: NamedArrays = {}
        
        for k in c.fieldnames('all'):
            v = c.field_vtypes[k]
            default = c.field_defaults[k]
            self.data[k] = np.full(
                (self.cache_size, self.max_n_obj) + v.shape,
                default, v.dtype.numpy)
        for k in c.fieldnames('state'):
            v = c.field_vtypes[k]
            default = c.field_defaults[k]
            self.data_next[k] = np.full(
                (self.cache_size, self.max_n_obj) + v.shape,
                default, v.dtype.numpy)

    def pad_to(self, n_obj: int):
        if self.max_n_obj >= n_obj:
            return
        
        for k in self.c.fieldnames():
            v = self.c.field_vtypes[k]
            default = self.c.field_defaults[k]     
            temp = np.full((self.cache_size, n_obj) + v.shape, default, dtype=v.dtype.numpy)
            temp[:, :self.max_n_obj, ...] = self.data[k]
            self.data[k] = temp

        for k in self.c.fieldnames('state'):
            v = self.c.field_vtypes[k]
            default = self.c.field_defaults[k]
            temp = np.full((self.cache_size, n_obj) + v.shape, default, dtype=v.dtype.numpy)
            temp[:, :self.max_n_obj, ...] = self.data_next[k]
            self.data_next[k] = temp

        temp = np.zeros((self.cache_size, n_obj), dtype=bool)
        temp[:,  :self.max_n_obj] = self.mask
        self.mask = temp

        self.max_n_obj = n_obj
    
    def fetch_numpy(self, indices: _IndexType) -> Tuple[NamedArrays, NamedArrays, np.ndarray]:
        attrs = {k: self.data[k][indices, ...] for k in self.c.fieldnames()}
        attrs_next = {k: self.data_next[k][indices, ...] for k in self.c.fieldnames('state')}
        mask = self.mask[indices, ...]
        return attrs, attrs_next, mask
    
    def fetch_tensors(self, indices: _IndexType, device: torch.device):
        attrs = {
            k: self.c.field_vtypes[k].tensor(self.data[k][indices, ...], device)
            for k in self.c.fieldnames()
        }
        attrs_next = {
            k: self.c.field_vtypes[k].tensor(self.data_next[k][indices, ...], device)
            for k in self.c.fieldnames('state')
        }
        mask = torch.tensor(self.mask[indices, ...], dtype=torch.bool, device=device)
        return attrs, attrs_next, mask


class ObjectOrientedBuffer:

    def __init__(self, capacity: int, envinfo: EnvInfo):
        self.__info: Final = envinfo
        self.capacity: Final = capacity
        self.__cache_size: Final = 2 * capacity
        self.__size = 0
        self.__beg = 0
        self.__data: Final = {c.name: _ClassDataCache(capacity, c)
                              for c in self.__info.classes}
        self.__reward = np.zeros(self.__cache_size, DType.Real.numpy)

    def __len__(self):
        return self.__size
    
    @property
    def full(self):
        return self.__size >= self.capacity
    
    def max_count(self, clsname: str):
        return self.__data[clsname].max_n_obj
    
    def refresh(self):
        size = self.__size
        beg = self.__beg
        end = beg + size

        for cache in self.__data.values():
            for buf in cache.data.values():
                buf[:size] = buf[beg: end]
            for buf in cache.data_next.values():
                buf[:size] = buf[beg: end]
            cache.mask[:size] = cache.mask[beg: end]
        self.__reward[:size] = self.__reward[beg: end]

        self.__beg = 0

    def add(self, attrs: ObjectArrays, state_next: ObjectArrays, reward: float):
        if self.__beg + self.__size >= self.__cache_size:
            self.refresh()

        i = self.__beg + self.__size
        for clsname in self.__info.clsnames:
            cache = self.__data[clsname]
            cache.mask[i, :] = False

            if clsname not in attrs:  # no object
                continue
            try:
                n_obj = next(iter(attrs[clsname].values())).shape[0]
            except StopIteration:  # the class has no attribute
                continue

            cache.pad_to(n_obj)
            for k, v in attrs[clsname].items():
                cache.data[k][i, :n_obj] = v
            for k, v in state_next[clsname].items():
                cache.data_next[k][i, :n_obj] = v
            cache.mask[i, :n_obj] = True
        self.__reward[i] = reward

        if self.__size == self.capacity:
            self.__beg += 1
        else:
            self.__size += 1

    def clear(self):
        self.__beg = 0
        self.__size = 0

    def __convert_indices(self, indices: Union[int, _IndexType]) -> _IndexType:
        if isinstance(indices, int):
            return [indices + self.__beg]
        elif isinstance(indices, slice):
            beg, size = self.__beg, self.__size
            start = indices.start
            stop = indices.stop
            step = indices.step
            start = start or 0
            stop = stop or size
            if start < 0:
                start = start + size
            if stop < 0:
                stop = stop + size
            start = np.clip(start, 0, size) + beg
            stop = np.clip(stop, 0, size) + beg
            return slice(start, stop, step)
        elif isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError("indices must be a vector / sequence")
            if indices.dtype != np.int32:
                indices = indices.astype(np.int32)
            return indices + self.__beg
        elif isinstance(indices, list):
            return np.array(indices, dtype=np.int32) + self.__beg
        else:
            assert False

    def fetch_numpy(self, indices: Union[_IndexType, int]):
        indices = self.__convert_indices(indices)

        attrs: ObjectArrays = {}
        next_state: ObjectArrays = {}
        mask: NamedArrays = {}

        for clsname, cache in self.__data.items():
            attrs[clsname], next_state[clsname], mask[clsname] = cache.fetch_numpy(indices)

        reward = self.__reward[indices]
        return attrs, next_state, mask, reward

    def fetch_tensors(self, indices: Union[_IndexType, int], device: torch.device):
        indices = self.__convert_indices(indices)
    
        attrs: ObjectTensors = {}
        next_state: ObjectTensors = {}
        mask: NamedTensors = {}

        for clsname, cache in self.__data.items():
            attrs[clsname], next_state[clsname], mask[clsname] = cache.fetch_tensors(indices, device)

        reward = torch.tensor(self.__reward[indices], dtype=DType.Real.torch, device=device)
        return attrs, next_state, mask, reward

    def sample_batch(self, batchsize: int, device: torch.device, replace=True):
        indices = np.random.choice(self.__size, batchsize, replace=replace)
        return self.fetch_tensors(indices, device)
    
    def epoch(self, batchsize: int, device: torch.device, keep_tail=True):
        all_indices = np.random.permutation(self.__size)
        for i in range(0, self.__size, batchsize):
            j = i + batchsize
            if j > self.__size and not keep_tail:
                break
            indices = all_indices[i: j]
            yield self.fetch_tensors(indices, device)
