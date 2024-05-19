from __future__ import annotations

from typing import Tuple, Union
import gymnasium
from gymnasium import spaces
from argparse import ArgumentParser, Namespace
import copy
import torch
import numpy as np
import abc

from typing import Final, Any, Dict, Iterable, Optional, Set, final, Literal
from utils.typings import ObjectArrays, ObjectTensors, ObjectValues, Role
from .objcls import EnvObjClass
from .envinfo import EnvInfo


_ClsIdxAttr = Tuple[str, int, str]
_ClsAttr = Tuple[str, str]


class _Obj:
    def __init__(self, data: Union[TaskData, ParalleledTaskData],
                 clsname: str, index: int, i_parallel: Optional[int] = None) -> None:

        if clsname not in data.info:
            raise KeyError(f"unkown class: {clsname}")

        self.__clsname__: str
        self.__index__: int
        self.__i_parallel__: Optional[int]
        self.__data__: TaskData

        super().__setattr__('__clsname__', clsname)
        super().__setattr__('__index__', index)
        super().__setattr__('__i_parallel__', i_parallel)
        super().__setattr__('__data__', data)

    @property
    def c(self):
        return self.__data__.info.c(self.__clsname__)
    
    def __getattr__(self, fieldname: str) -> np.ndarray:
        if self.__i_parallel__ is None:
            return self.__data__[self.__clsname__, self.__index__, fieldname]
        else:
            assert isinstance(self.__data__, ParalleledTaskData)
            tensor = self.__data__[self.__clsname__, self.__index__, fieldname][self.__i_parallel__]
            vtype = self.__data__.info.v(self.__clsname__, fieldname)
            a: np.ndarray = tensor.detach().cpu().numpy()
            if a.dtype != vtype.dtype.numpy:
                a = a.astype(vtype.dtype.numpy)
            return a

    def __setattr__(self, fieldname: str, value: Any):
        if self.__i_parallel__ is None:
            self.__data__[self.__clsname__, self.__index__, fieldname] = value
        else:
            assert isinstance(self.__data__, ParalleledTaskData)
            self.__data__[self.__clsname__, self.__index__, fieldname][self.__i_parallel__] = value

    def print_attributes(self, indent=0, role: Role = 'state'):
        temp = '| ' * indent
        print(temp + f"object of class <{self.c.name}>:")

        names = self.c.fieldnames(role)
        for name in names:
            attr = getattr(self, name)
            print(temp + "| " + f"{name} = {attr}")


class TaskData:
    def __init__(self, envinfo: EnvInfo):
        self.info: Final = envinfo
        self._counts: Dict[str, int]
        self._attrs: Dict[str, Dict[str, np.ndarray]]
    
    def init_instances(self, **num_instances):
        self._counts = {cname: 0 for cname in self.info.clsnames}
        for clsname, n_obj in num_instances.items():
            if clsname not in self._counts:
                raise ValueError(f"<{clsname}> is not in the given classes.")
            self._counts[clsname] = n_obj

        self._attrs = {}
        for c in self.info.classes:
            n_obj = self._counts[c.name]
            self._attrs[c.name] = {}
            for fieldname in c.fieldnames():
                vtype = c.field_vtypes[fieldname]
                default = c.field_defaults[fieldname]
                self._attrs[c.name][fieldname] = np.full(
                    (n_obj,) + vtype.shape, default, vtype.dtype.numpy)

    def deepcopy(self):
        t = TaskData(self.info)
        t._counts = self._counts.copy()
        t._attrs = {c.name: {fieldname: np.copy(self._attrs[c.name][fieldname])
                            for fieldname in c.fieldnames()}
                    for c in self.info.classes}
        return t

    def __contains__(self, clsname: str):
        try:
            n = self._counts[clsname]
            return n > 0
        except KeyError:
            return False
    
    def __getitem__(self, index: Union[_ClsAttr, _ClsIdxAttr]) -> np.ndarray:
        if len(index) == 3:
            clsname, i_obj, fieldname = index

            return self._attrs[clsname][fieldname][i_obj]
        elif len(index) == 2:
            clsname, fieldname = index
            return self._attrs[clsname][fieldname]
        else:
            assert False
    
    def __setitem__(self, index: Union[_ClsAttr, _ClsIdxAttr], value: Any):
        if len(index) == 3:
            clsname, i_obj, fieldname = index
            self._attrs[clsname][fieldname][i_obj] = value
        elif len(index) == 2:
            clsname, fieldname = index
            self._attrs[clsname][fieldname][:] = value
        else:
            assert False
    
    def set_attributes(self, values: ObjectValues):        
        for clsname, cvalue in values.items():
            for fieldname, value in cvalue.items():
                self._attrs[clsname][fieldname][:] = value

    @final
    def get(self, role: Role = 'state') -> ObjectArrays:
        return {c.name: {fieldname: np.copy(self._attrs[c.name][fieldname])
                          for fieldname in c.fieldnames(role)}
                for c in self.info.classes if self._counts[c.name] > 0}
    
    def get_obj(self, clsname: str, index: int):
        return _Obj(self, clsname, index)
    
    def count(self, clsname: str):
        return self._counts[clsname]
    
    def print_objects(self, indent=0, role: Role = 'state'):
        for clsname in self.info.clsnames:
            for i in range(self._counts[clsname]):
                o = self.get_obj(clsname, i)
                o.print_attributes(indent, role)
    
    @final
    def set_action(self, action: ObjectArrays):
        for c in self.info.classes:
            for attr in c.fieldnames('action'):
                self._attrs[c.name][attr][:] = action[c.name][attr]
    
    def parallel(self, n_parallel: int, device: torch.device):
        pd = ParalleledTaskData(self.info, n_parallel, device)
        pd.init_instances(**self._counts)
        pd.set_attributes(self._attrs)
        return pd


class ParalleledTaskData:
    def __init__(self, envinfo: EnvInfo, n_parallel: int, device: torch.device):
        self.info: Final = envinfo
        self.n_parallel: Final = n_parallel
        self.device: Final = device

        self._counts: Dict[str, int]
        self._attrs: Dict[str, Dict[str, torch.Tensor]]
        
    def init_instances(self, **num_instances):
        self._counts = {cname: 0 for cname in self.info.clsnames}
        
        for clsname, n_obj in num_instances.items():
            if clsname not in self._counts:
                raise ValueError(f"<{clsname}> is not in the given classes.")
            self._counts[clsname] = n_obj
        
        self._attrs = {}
        for c in self.info.classes:
            n_obj = self._counts[c.name]
            self._attrs[c.name] = {}
            for fieldname in c.fieldnames():
                vtype = c.field_vtypes[fieldname]
                default = c.field_defaults[fieldname]
                self._attrs[c.name][fieldname] = torch.full(
                    (self.n_parallel, n_obj,) + vtype.shape, default,
                    dtype=vtype.dtype.torch, device=self.device)
    
    def __getitem__(self, index: Union[_ClsAttr, _ClsIdxAttr]) -> torch.Tensor:
        if len(index) == 3:
            clsname, i_obj, fieldname = index
            return self._attrs[clsname][fieldname][:, i_obj]
        elif len(index) == 2:
            clsname, fieldname = index
            return self._attrs[clsname][fieldname]
        else:
            assert False
    
    def __typed(self, clsname: str, fieldname: str, value: Any): 
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
            return value

        vtype = self.info.v(clsname, fieldname)
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(value, device=self.device, dtype=vtype.dtype.torch)
        else:
            tensor = value.to(device=self.device, dtype=vtype.dtype.torch)
        return tensor

    def __setitem__(self, index: Union[_ClsAttr, _ClsIdxAttr], value: Any):
        if len(index) == 3:
            clsname, i_obj, fieldname = index
            self._attrs[clsname][fieldname][:, i_obj] = self.__typed(clsname, fieldname, value)
        elif len(index) == 2:
            clsname, fieldname = index
            self._attrs[clsname][fieldname][:] = self.__typed(clsname, fieldname, value)
        else:
            assert False
    
    def get_obj(self, clsname: str, index: int, i_parallel: int):
        return _Obj(self, clsname, index, i_parallel)

    def set_attributes(self, values: ObjectValues):        
        for clsname, cvalue in values.items():
            for fieldname, value in cvalue.items():
                self._attrs[clsname][fieldname][:] = self.__typed(clsname, fieldname, value)

    def get_thread(self, i_parallel: int):
        '''returns the copy of data of one thread'''

        d = TaskData(self.info)
        d._counts = self._counts
        attrs = {}
        for cname, cdata in self._attrs.items():
            c = self.info.c(cname)
            attrs[cname] = {}
            for fieldname, attrdata in cdata.items():
                vtype = c.field_vtypes[fieldname]
                a: np.ndarray = attrdata[i_parallel].detach().cpu().numpy()
                if a.dtype != vtype.dtype.numpy:
                    a = a.astype(vtype.dtype.numpy)
                attrs[cname][fieldname] = a
        d._attrs = attrs
        return d
    
    def set_thread(self, i_parallel, thread_data: TaskData):
        for cname, cdata in self._attrs.items():
            c = self.info.c(cname)
            for fieldname, attrdata in cdata.items():
                v = c.field_vtypes[fieldname]
                thread_attrdata = torch.tensor(thread_data[cname, fieldname],
                                               dtype=v.dtype.torch, device=self.device)
                assert attrdata.shape[1:] ==  thread_attrdata.shape
                attrdata[i_parallel, ...] = thread_attrdata

    def print_objects(self, i_thread: Optional[int] = None, indent=0, role: Role = 'state'):
        if i_thread is not None:
            for clsname in self.info.clsnames:
                for i in range(self._counts[clsname]):
                    o = self.get_obj(clsname, i, i_thread)
                    o.print_attributes(indent+1, role)
        else:
            for i_parallel in range(self.n_parallel):
                print("| " * indent + f"{i_parallel}-th thread:")
                self.print_objects(i_parallel, indent+1, role)

    def set_action(self, action: ObjectTensors):
        for c in self.info.classes:
            for attr in c.fieldnames('action'):
                self._attrs[c.name][attr][:] = action[c.name][attr]

    def observe(self, role: Role = 'state') -> ObjectTensors:
        return {c.name: {fieldname: self._attrs[c.name][fieldname]
                         for fieldname in c.fieldnames(role)}
            for c in self.info.classes
            if self._counts[c.name] > 0}

    def deepcopy(self):
        t = ParalleledTaskData(self.info, self.n_parallel, self.device)
        t._counts = self._counts.copy()
        t._attrs = {c.name: {fieldname: self._attrs[c.name][fieldname].detach().clone()
                             for fieldname in c.fieldnames()}
                    for c in self.info.classes}
        return t
