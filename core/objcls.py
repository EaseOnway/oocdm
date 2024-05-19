from typing import Final, Any, Dict,  Literal, Optional
from utils.typings import Names, NamedValues, Role
from .vtype import VType


class EnvObjClass:
    def __init__(self, name: str, base: Optional['EnvObjClass'] = None):
        self.name: Final[str] = name
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a legal identifier.")

        self.field_vtypes: Dict[str, VType] = {}
        self.field_defaults: NamedValues = {}
        
        self.__fieldnames: Dict[Role, Names] = {
            'all': (), 'state': (), 'action': ()}

        self.__field_indices: Dict[Role, Dict[str, int]] = {
            'all': {}, 'state': {}, 'action': {}}

        if base is not None:
            for a in base.fieldnames('state'):
                self.declear_field(a, base.v(a), 'state', base.field_defaults[a])
            for a in base.fieldnames('action'):
                self.declear_field(a, base.v(a), 'action', base.field_defaults[a])

    def n_field(self, role: Role = 'all'):
        return len(self.__fieldnames[role])
    
    def declear_field(self, name: str, vtype: VType,
                          role: Literal['state', 'action'] = 'state',
                          default: Any = 0):

        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a legal identifier.")
        elif name in self.field_vtypes:
            raise KeyError(f"attribute {name} already exists")
        else:
            self.__field_indices['all'][name] = self.n_field('all')
            self.__fieldnames['all'] += (name,)

            self.field_defaults[name] = default
            self.field_vtypes[name] = vtype

            if role == 'action':
                self.__field_indices['action'][name] = self.n_field('action')
                self.__fieldnames['action'] += (name,)
            elif role == 'state':
                self.__field_indices['state'][name] = self.n_field('state')
                self.__fieldnames['state'] += (name,)
            else:
                assert False
    
    def __contains__(self, field_name: str):
        return field_name in self.field_vtypes
    
    @property
    def contains_action(self):
        return len(self.__fieldnames['action']) > 0
    
    def v(self, fieldname: str):
        return self.field_vtypes[fieldname]

    def fieldnames(self, role: Role = 'all'):
        return self.__fieldnames[role]
    
    def index(self, fieldname: str, role: Role = 'all'):
        return self.__field_indices[role][fieldname]

    def __str__(self):
        return f"class<{self.name}>"
    
    def __repr__(self):
        return str(self)
