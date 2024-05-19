from typing import Final, Dict, Optional,  List, TypeVar, Sequence, Literal

import torch
from torch.distributions import Distribution
from core import TaskData
from utils.typings import ObjectTensors, NamedTensors, ObjectDistributions, NamedDistributions


_T = TypeVar('_T')
def _index_dict(seq: Sequence[_T]) -> Dict[_T, int]:
    return {x: i for i, x in enumerate(seq)}


class Variable:
    def __init__(self, clsname: str, obj_index: int, fieldname: str) -> None:
        self.__tuple = (clsname, obj_index, fieldname)
        self.__str = f"{clsname}__{obj_index}__{fieldname}"
        self.__repr = f"{clsname}[{obj_index}].{fieldname}"
        self.clsname: Final = clsname
        self.obj_index: Final = obj_index
        self.field_name: Final = fieldname
    
    def __eq__(self, other):
        if isinstance(other, Variable):
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
        return values[self.clsname][self.field_name][i]


class VariableInfo:
    def __init__(self, envdata: TaskData):
        self.envinfo = envdata.info
        self.counts = envdata._counts.copy()
        self.input_variables = tuple(
            Variable(c.name, i, fieldname)
            for c in self.envinfo.classes
            for i in range(self.counts[c.name])
            for fieldname in c.fieldnames()
        )
        self.n_input_variable = len(self.input_variables)
        self.__input_index_dict = _index_dict(self.input_variables)
        self.output_variables = tuple(
            Variable(c.name, i, fieldname)
            for c in self.envinfo.classes
            for i in range(self.counts[c.name])
            for fieldname in c.fieldnames('state')
        )
        self.__output_index_dict = _index_dict(self.output_variables)
        self.n_output_variable = len(self.output_variables)

        self.__output_variables_by_object = {
            c.name: {
                fieldname: [Variable(c.name, i, fieldname)
                           for i in range(self.counts[c.name])] 
                 for fieldname in c.fieldnames('state')}
            for c in self.envinfo.classes}

    def v(self, var: Variable):
        return self.envinfo.v(var.clsname, var.field_name)
    
    def get_var_distr(self, distr_params: Dict[Variable, NamedTensors]) -> Dict[Variable, Distribution]:
        out: Dict[Variable, Distribution] = {}
        for var, params in distr_params.items():
            out[var] = self.v(var).ptype(**params)
        return out

    def get_obj_distr(self, distr_params: Dict[Variable, NamedTensors]) -> ObjectDistributions:
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
        return self.__input_index_dict[Variable(clsname, idx_obj, fieldname)]
    
    def index_output(self, clsname: str, idx_obj: int, fieldname: str):
        return self.__output_index_dict[Variable(clsname, idx_obj, fieldname)]
