from typing import Final, Optional, Set, Tuple

import torch
import numpy as np

from core import EnvInfo, EnvObjClass
from utils.typings import NamedTensors, ObjectTensors
from core.causal_graph import ObjectOrientedCausalGraph


from typing import Final, Optional, Set, Tuple

import torch
import numpy as np

from core import EnvInfo, EnvObjClass
from utils.typings import NamedTensors, ObjectTensors
from core.causal_graph import ObjectOrientedCausalGraph


class MaskGenerator:
    def __init__(self, envinfo: EnvInfo, device: torch.device):
        self.info = envinfo
        self.device: Final = device

    def global_mask(self, batchsize: int) -> torch.Tensor:
        """Generate global mask for all attributes.

        Args:
            batchsize (int)

        Returns:
            Tensor: shape can be broadcasted to `(batchsize, n_statefield, n_field)`
        """
        raise NotImplementedError

    def local_mask(self, clsname: str, batchsize: int) -> torch.Tensor:
        """Generate local mask for attributes of a given class.

        Args:
            clsname (str): the class name of the target
            batchsize (int)

        Returns:
            Tensor: shape can be broadcasted to `(batchsize, n_statefield_cls, n_field_cls)`
        """        
        raise NotImplementedError


class FullMaskGenerator(MaskGenerator):
    def __init__(self, envinfo: EnvInfo, device: torch.device) -> None:
        super().__init__(envinfo, device)

        self.__global = torch.ones(envinfo.n_field('state'), envinfo.n_field(),
                                   dtype=torch.bool, device=self.device)

        self.__local = {
            c.name: torch.ones(c.n_field('state'), c.n_field(),
                                dtype=torch.bool, device=self.device)
            for c in envinfo.classes
        }

    def global_mask(self, batchsize: int) -> torch.Tensor:
        return self.__global.expand(batchsize, -1, -1)

    def local_mask(self, clsname: str, batchsize: int) -> torch.Tensor:
        return self.__local[clsname].expand(batchsize, -1, -1)


class RandomMaskGenerator(MaskGenerator):
    def __init__(self, envinfo: EnvInfo, device: torch.device,
                 p_edge = 0.9):
        super().__init__(envinfo, device)
        self.p_edge = p_edge
    
    def make_tensor(self, *shape: int):
        return torch.rand(shape, device=self.device) <= self.p_edge
    
    def global_mask(self, batchsize: int) -> torch.Tensor:
        info = self.info
        return self.make_tensor(batchsize, info.n_field('state'), info.n_field())
    
    def local_mask(self, clsname: str, batchsize: int) -> torch.Tensor:
        c = self.info.c(clsname)
        return self.make_tensor(batchsize, c.n_field('state'), c.n_field())


class DroppedMaskGenerator(MaskGenerator):
    def __init__(self, envinfo: EnvInfo, device: torch.device):
        super().__init__(envinfo, device)
        self.__reset_tensors()
    
    def __reset_tensors(self):
        info = self.info
        self.__global = torch.ones(info.n_field('state'), info.n_field(),
                                   dtype=torch.bool, device=self.device)
        self.__local = {
            c.name: torch.ones(c.n_field('state'), c.n_field(),
                                dtype=torch.bool, device=self.device)
            for c in info.classes
        }
    
    def drop(self, src_index: int, target_index: Optional[int] = None, 
             clsname: Optional[str] = None):
        self.__reset_tensors()
        if target_index is not None:
            if clsname is None:  # global
                self.__global[target_index, src_index] = False
            else:  # local
                self.__local[clsname][target_index, src_index] = False
        else:
            if clsname is None:  # global
                self.__global[:, src_index] = False
            else:  # local
                self.__local[clsname][:, src_index] = False
  
    def global_mask(self, batchsize: int) -> torch.Tensor:
        return self.__global.expand(batchsize, -1, -1)

    def local_mask(self, clsname: str, batchsize: int) -> torch.Tensor:
        return self.__local[clsname].expand(batchsize, -1, -1)


class GraphMaskGenerator(MaskGenerator):
    def __init__(self, envinfo: EnvInfo, device: torch.device) -> None:
        super().__init__(envinfo, device)
    
    def load_graph(self, g: ObjectOrientedCausalGraph):
        info = self.info
        self.__global = torch.tensor(g.global_matrix, dtype=torch.bool,
                                     device=self.device)
        self.__local = {
            clsname: torch.tensor(g.local_matrices[clsname],
                                  dtype=torch.bool, device=self.device)
            for clsname in info.clsnames
        }

    def __create_mask_from_parents(self, c: EnvObjClass, parents: Set[str]):
        m = torch.zeros(c.n_field(), dtype=torch.bool, device=self.device)
        for i, fieldname in enumerate(c.fieldnames()):
            if fieldname in parents:
                    m[..., i] = True
        return m

    def global_mask(self, batchsize: int) -> torch.Tensor:
        return self.__global.expand(batchsize, -1, -1)

    def local_mask(self, clsname: str, batchsize: int) -> torch.Tensor:
        return self.__local[clsname].expand(batchsize, -1, -1)



# class MaskGenerator:
#     def __init__(self, envinfo: EnvInfo, device: torch.device):
#         self.clsdict: Final = envinfo.clsdict
#         self.device: Final = device
# 
#     def global_(self, clsname: str, fieldname: str, encodings: NamedTensors) -> NamedTensors:
#         """Generate global mask for attributes.
# 
#         Args:
#             clsname (str): the class name of the target
#             fieldname (str): the attribute name of the target
#             encodings (NamedTensors): the data of attributes that requires the mask. \
#                 For each `clsname`, `attrs[clsname]` is a tensor shaped as `(batchsize, n_obj, \
#                 n_field, dim_encoding)`
# 
#         Returns:
#             NamedTensors: A dictionary with the masks for all classes. For each class `cls`, \
#                 the mask is a tensor, whose shape can be broadcast to `(batchsize, n_obj_cls, \
#                 n_field_cls)`
#         """
#         raise NotImplementedError
# 
#     def local_(self, clsname: str, fieldname: str, encoding: torch.Tensor) -> torch.Tensor:
#         """Generate local mask for attributes.
# 
#         Args:
#             clsname (str): the class name of the target
#             fieldname (str): the attribute name of the target
#             encoding (Tensor): the data of attributes that requires the mask. \
#                 shaped as `(batchsize, n_obj, n_field, dim_encoding)`
# 
#         Returns:
#             Tensor: shape can be broadcast to `(batchsize, n_obj_cls, n_field_cls)`
#         """        
#         raise NotImplementedError
# 
# 
# class FullMaskGenerator(MaskGenerator):
#     def __init__(self, envinfo: EnvInfo, device: torch.device) -> None:
#         super().__init__(envinfo, device)
# 
#         self.__tensors = {clsname: torch.ones(c.n_field, device=self.device, dtype=torch.bool)
#                           for clsname, c in self.clsdict.items()}
# 
#     def global_(self, clsname: str, fieldname: str, encodings: NamedTensors) -> NamedTensors:
#         return self.__tensors
#     
#     def local_(self, clsname: str, fieldname: str, encoding: torch.Tensor) -> torch.Tensor:
#         return self.__tensors[clsname]
# 
# 
# class RandomMaskGenerator(MaskGenerator):
#     def __init__(self, envinfo: EnvInfo, device: torch.device,
#                  p_edge = 0.9):
#         super().__init__(envinfo, device)
#         self.p_edge = p_edge
#         
#     def global_(self, clsname: str, fieldname: str, encodings: NamedTensors) -> NamedTensors:
#         tensors = {
#             clsname_: (torch.rand((e.shape[0], 1, e.shape[2]), device=self.device) <= self.p_edge)
#             for clsname_, e in encodings.items()
#         }
#         return tensors
# 
#     def local_(self, clsname: str, fieldname: str, encoding: torch.Tensor) -> torch.Tensor:
#         batchsize, n_obj, n_field, dim = encoding.shape
#         return torch.rand((batchsize, 1, n_field), device=self.device) <= self.p_edge
# 
# 
# class DroppedMaskGenerator(MaskGenerator):
#     def __init__(self, envinfo: EnvInfo, device: torch.device):
#         super().__init__(envinfo, device)
# 
#         self.__dropped_clsname: str
#         self.__dropped_fieldname: str
#         self.__global = False
# 
#         self.__full_tensors = {
#             clsname: torch.ones(c.n_field, device=self.device, dtype=torch.bool)
#             for clsname, c in self.clsdict.items()}
#     
#     def __index(self, clsname: str, fieldname: str):
#         return self.clsdict[clsname].fieldnames.index(fieldname)
#     
#     def reset_drop(self, clsname: str, fieldname: str, _global = False):
#         self.__dropped_clsname = clsname
#         self.__dropped_fieldname = fieldname
#         self.__global = _global
# 
#     def global_(self, clsname: str, fieldname: str, encodings: NamedTensors) -> NamedTensors:
#         tensors = self.__full_tensors.copy()
#         if self.__global:
#             m = torch.ones_like(tensors[self.__dropped_clsname])
#             m[self.__index(self.__dropped_clsname, self.__dropped_fieldname)] = False
#             tensors[self.__dropped_clsname] = m
#         return tensors
# 
#     def local_(self, clsname: str, fieldname: str, encoding: torch.Tensor) -> torch.Tensor:
#         c = self.clsdict[clsname]
#         tensor = torch.ones(c.n_field, device=self.device, dtype=torch.bool)
#         if not self.__global and clsname == self.__dropped_clsname:
#             tensor[self.__index(clsname, self.__dropped_fieldname)] = False
#         return tensor
# 
# 
# class GraphMaskGenerator(MaskGenerator):
#     def __init__(self, envinfo: EnvInfo, device: torch.device,
#                  causal_graph: ObjectOrientedCausalGraph) -> None:
#         super().__init__(envinfo, device)
# 
#         assert causal_graph.clsdict is self.clsdict
#         self.causal_graph = causal_graph
# 
#     def __create_mask_from_parents(self, c: EnvObjClass, parents: Set[str]):
#         m = torch.zeros(c.n_field, dtype=torch.bool, device=self.device)
#         for i, fieldname in enumerate(c.fieldnames):
#             if fieldname in parents:
#                     m[..., i] = True
#         return m
# 
#     def global_(self, clsname: str, fieldname: str, encodings: NamedTensors) -> NamedTensors:
#         out: NamedTensors = {}
#         parent_dict = self.causal_graph.global_parents_of(clsname, fieldname)
#         for clsname, parents in parent_dict.items():
#             c = self.clsdict[clsname]
#             out[clsname] = self.__create_mask_from_parents(c, parents)
#         return out
# 
#     def local_(self, clsname: str, fieldname: str, encoding: torch.Tensor) -> torch.Tensor:
#         c = self.clsdict[clsname]
#         parents = self.causal_graph.local_parents_of(clsname, fieldname)
#         return self.__create_mask_from_parents(c, parents)
# 