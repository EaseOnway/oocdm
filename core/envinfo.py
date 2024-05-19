from typing import Final, Dict, Sequence, TypeVar, Literal, Union, Tuple

import torch
from torch.distributions import Distribution
from utils.typings import ObjectTensors, NamedTensors, ObjectDistributions, NamedDistributions
from .objcls import  EnvObjClass


_T = TypeVar('_T')
def _index_dict(seq: Sequence[_T]) -> Dict[_T, int]:
    return {x: i for i, x in enumerate(seq)}


Role = Literal['all', 'state', 'action']
_ROLES = ('all', 'state', 'action')


class EnvInfo:

    def __init__(self, classes: Sequence[EnvObjClass]):
        self.classes: Final = tuple(classes)
        self.clsnames: Final = tuple(c.name for c in classes)
        self.__clsdict: Final[Dict[str, EnvObjClass]] = {}
        
        for c in self.classes:
            if c.name in self.__clsdict:
                raise ValueError(f"redefinition of {c}")
            else:
                self.__clsdict[c.name] = c
        
        self.__attrs = {
            role: tuple((c.name, fieldname)
                for c in self.classes for fieldname in c.fieldnames(role))
            for role in _ROLES}

        self.__n_field = {role: len(self.__attrs[role]) for role in _ROLES}
        self.__indices = {role: _index_dict(self.__attrs[role]) for role in _ROLES}
        self.__begin_indices = {role: self.__compute_begin_indices(role) for role in _ROLES}
    
    def __compute_begin_indices(self, role: Role):
        i = 0
        out: Dict[str, int] = {}
        for c in self.classes:
            out[c.name] = i
            i += c.n_field(role)
        return out
    
    def __contains__(self, key: Union[str, Tuple[str, str], EnvObjClass]):
        if isinstance(key, str):
            return key in self.__clsdict
        elif isinstance(key, EnvObjClass):
            return key.name in self.__clsdict and key == self.__clsdict[key.name]
        elif isinstance(key, tuple):
            cname, fieldname = key
            return cname in self.__clsdict and fieldname in self.__clsdict[cname]
        else:
            return False

    def c(self, clsname: str):
        '''return the class information of `clsname`'''
        return self.__clsdict[clsname]
    
    def v(self, clsname: str, fieldname: str):
        '''returns the variable information of `clsname.fieldname`'''
        return self.__clsdict[clsname].field_vtypes[fieldname]

    def n_field(self, role: Role = 'all'):
        return len(self.__attrs[role])

    def fields(self, role: Role = 'all'):
        return self.__attrs[role]

    def field_index(self, clsname: str, fieldname: str, role: Role = 'all'):
        return self.__indices[role][clsname, fieldname]

    def field_slice(self, clsname: str, role: Role) -> slice:
        '''
        returns the index slice that points to the attributes of a class
        in global attributes.
        '''

        c = self.__clsdict[clsname]
        beg = self.__begin_indices[role][clsname]
        end = beg + c.n_field(role)
        return slice(beg, end)


class VarID:
    def __init__(self, clsname: str, obj_index: int, fieldname: str) -> None:
        self.__tuple = (clsname, obj_index, fieldname)
        self.__str = f"{clsname}__{obj_index}__{fieldname}"
        self.__repr = f"{clsname}[{obj_index}].{fieldname}"
        self.clsname: Final = clsname
        self.obj_index: Final = obj_index
        self.fieldname: Final = fieldname
    
    def __eq__(self, other):
        if isinstance(other, VarID):
            return self.__tuple == other.__tuple
        else:
            return False

    def __str__(self):
        return self.__str
    
    def __repr__(self):
        return self.__repr
    
    def __hash__(self):
        return hash(self.__tuple)
    
    def __call__(self, values: ObjectTensors, ndim_batch=1):
        i = tuple(slice(None) for _ in range(ndim_batch)) + (self.obj_index,)
        return values[self.clsname][self.fieldname][i]


class Taskinfo:
    def __init__(self, envinfo: 'EnvInfo', counts: Dict[str, int]):
        self.envinfo = envinfo
        self.counts = counts.copy()

        self.__allobj: Final = tuple(
            (c, i)
            for c in self.envinfo.classes
            for i in range(self.counts[c.name])
        )
        self.n_allobj: Final = len(self.__allobj)

        self.input_variables = tuple(
            VarID(c.name, i, fieldname)
            for c in self.envinfo.classes
            for i in range(self.counts[c.name])
            for fieldname in c.fieldnames()
        )
        self.n_input_variable = len(self.input_variables)
        self.__input_index_dict = _index_dict(self.input_variables)
        self.output_variables = tuple(
            VarID(c.name, i, fieldname)
            for c in self.envinfo.classes
            for i in range(self.counts[c.name])
            for fieldname in c.fieldnames('state')
        )
        self.__output_index_dict = _index_dict(self.output_variables)
        self.n_output_variable = len(self.output_variables)

        self.__output_variables_by_object = {
            c.name: {
                fieldname: [VarID(c.name, i, fieldname)
                           for i in range(self.counts[c.name])] 
                 for fieldname in c.fieldnames('state')}
            for c in self.envinfo.classes}

    def v(self, var: VarID):
        return self.envinfo.v(var.clsname, var.fieldname)

    def class_of_obj(self, i_obj: int):
        return self.__allobj[i_obj][0]

    def variables_of_obj(self, i_obj: int, role: Role = 'all'):
        c, i = self.__allobj[i_obj]
        return [VarID(c.name, i, fname) for fname in c.fieldnames(role)]

    def n_variable(self, role: Role = 'all'):
        if role == 'all':
            return self.n_input_variable
        elif role == 'state':
            return self.n_output_variable
        elif role == 'action':
            return self.n_input_variable - self.n_output_variable
    
    def get_var_distr(self, distr_params: Dict[VarID, NamedTensors]) -> Dict[VarID, Distribution]:
        out: Dict[VarID, Distribution] = {}
        for var, params in distr_params.items():
            out[var] = self.v(var).ptype(**params)
        return out

    def get_obj_distr(self, distr_params: Dict[VarID, NamedTensors]) -> ObjectDistributions:
        out: ObjectDistributions = {}
        for c in self.envinfo.classes:
            if self.counts[c.name] == 0:
                continue
            out_c: NamedDistributions = {}
            for fieldname in c.fieldnames('state'):
                vtype = c.v(fieldname)
                params = {}
                for paramname in vtype.ptype.param_names:
                    temp = [distr_params[i][paramname]
                            for i in self.__output_variables_by_object[c.name][fieldname]]
                    param = torch.stack(temp, dim=1)
                    params[paramname] = param
                out_c[fieldname] = vtype.ptype(**params)
            out[c.name] = out_c
        return out

    def index_input(self, clsname: str, idx_obj: int, fieldname: str):
        return self.__input_index_dict[VarID(clsname, idx_obj, fieldname)]
    
    def index_output(self, clsname: str, idx_obj: int, fieldname: str):
        return self.__output_index_dict[VarID(clsname, idx_obj, fieldname)]
