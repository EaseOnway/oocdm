from typing import Dict, Tuple, Final, Set, Iterable, Optional
import numpy as np
import json

from core import EnvInfo
from utils.typings import NamedTensors
from .varinfo import VariableInfo


class CausalGraph:
    def __init__(self, info: VariableInfo):
        self.matrix = np.zeros((info.n_output_variable, info.n_input_variable), dtype=bool)
        self.__info = info
    
    def __str__(self):
        lines = []
        for j in range(self.__info.n_output_variable):
            name_j = repr(self.__info.output_variables[j])
            parents_i = np.nonzero(self.matrix[j])[0]
            parents = ', '.join([repr(self.__info.input_variables[i]) for i in parents_i])
            lines.append(f"{name_j} <- ({parents})")
        return '\n'.join(lines)
    
    def set_edge(self, i: int, j: int, value=True):
        self.matrix[j, i] = value
    
    def state_dict(self):
        j, i = np.nonzero(self.matrix)
        return {"j": j.tolist(), "i": i.tolist()}
    
    def load_state_dict(self, state_dict: dict):
        self.matrix[:] = False
        self.matrix[state_dict['j'], state_dict['i']] = True
    
    def load_object_oriented_graph(self, g: 'ObjectOrientedCausalGraph'):
        varinfo = self.__info
        envinfo = varinfo.envinfo

        # clear
        self.matrix[:] = False

        # local edges
        for c in envinfo.classes:
            for field_i, fieldname_i in enumerate(c.fieldnames()):
                for field, fieldname_j in enumerate(c.fieldnames('state')):
                    if g.local_matrices[c.name][field, field_i]:
                        for o in range(varinfo.counts[c.name]):
                            i = varinfo.index_input(c.name, o, fieldname_i)
                            j = varinfo.index_output(c.name, o, fieldname_j)
                            self.set_edge(i, j)
        # global edges
        for field_i, (clsname_i, fieldname_i) in enumerate(envinfo.fields()):
            for field, (clsname_j, fieldname_j) in enumerate(envinfo.fields('state')):
                if g.global_matrix[field, field_i]:
                    for oi in range(varinfo.counts[clsname_i]):
                        for oj in range(varinfo.counts[clsname_j]):
                            i = varinfo.index_input(clsname_i, oi, fieldname_i)
                            j = varinfo.index_output(clsname_j, oj, fieldname_j)
                            self.set_edge(i, j)


class ObjectOrientedCausalGraph:
    def __init__(self, info: EnvInfo):
        self.__info = info
        self.global_matrix = np.zeros((info.n_field('state'), info.n_field()), dtype=bool)
        self.local_matrices = {
            c.name: np.zeros((c.n_field('state'), c.n_field()), dtype=bool)
            for c in info.classes
        }

    def local_parents_of(self, clsname: str, fieldname: str):
        c = self.__info.c(clsname)
        j = c.index(fieldname, 'state')
        mat = self.local_matrices[clsname]
        return set(name_i
            for i, name_i in enumerate(c.fieldnames())
            if mat[j, i]
        )
    
    def global_parents_of(self, clsname: str, fieldname: str):
        out: Dict[str, Set[str]] = {cname: set() for cname in self.__info.clsnames}
        j = self.__info.field_index(clsname, fieldname, 'state')
        for i, (clsname_i, fieldname_i) in enumerate(self.__info.fields()):
            if self.global_matrix[j, i]:
                out[clsname_i].add(fieldname_i)
        return out

    def set_edge(self, i: int, j: int, clsname: Optional[str], value=True):
        if clsname is None:  # global
            self.global_matrix[j, i] = value
        else:
            self.local_matrices[clsname][j, i] = value
    
    def set_local_edge_by_name(self, clsname: str, i: str, j: str, value=True):
        c = self.__info.c(clsname)
        self.set_edge(c.index(i), c.index(j, 'state'), clsname, value)
    
    def set_global_edge_by_name(self, i: Tuple[str, str], j: Tuple[str, str], value=True):
        info = self.__info
        self.set_edge(info.field_index(*i), info.field_index(*j, 'state'), None, value)

    def __str__(self):
        lines = []
        for c in self.__info.classes:
            lines.append(f"{c}:")
            for fieldname in c.fieldnames('state'):
                pa_local = list(self.local_parents_of(c.name, fieldname))
                pa_global = [
                    "%s.%s" % (clsname_j, fieldname_j) 
                    for clsname_j, fieldnames_j in self.global_parents_of(c.name, fieldname).items()
                    for fieldname_j in fieldnames_j
                ]
                pa = ', '.join(pa_local + pa_global)
                lines.append(f"- {fieldname} <- ({pa})")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return str(self)
    
    def state_dict(self):
        global_indices = [indices.tolist() for indices in np.nonzero(self.global_matrix)]
        local_indices = {
            clsname: [indices.tolist() for indices in np.nonzero(self.local_matrices[clsname])]
            for clsname in self.__info.clsnames
        }
        return {'globals': global_indices, 'locals': local_indices}

    def load_state_dict(self, d: dict):
        global_indices = tuple(d['globals'])
        self.global_matrix[:] = False
        self.global_matrix[global_indices] = True
        for clsname in self.__info.clsnames:
            self.local_matrices[clsname][:] = False
            local_indices = tuple(d['locals'][clsname])
            self.local_matrices[clsname][local_indices] = True
