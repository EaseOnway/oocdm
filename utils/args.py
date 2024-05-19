from typing import Any, Dict, Literal, Set, Tuple, final, Optional, Union, List
import torch

import json
import pathlib
import argparse


class Struct(argparse.Namespace):

    def __lines(self, parent: str = ""):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                lines.extend(v.__lines(parent + k + '.'))
            else:
                lines.append(parent + k + ': ' + str(v))
        return lines
    
    def __str__(self):
        return '\n'.join(self.__lines())

    def __repr__(self):
        return str(self)

    def valid(self, k):
        '''
        return True if attribute self.k exists and is not None.
        '''
        try:
            v = getattr(self, k)
            return v is not None
        except AttributeError:
            return False
    
    @final
    def set(self, __args: Union[dict, 'Struct', argparse.Namespace, None] = None, **kargs):
        '''set the attributes using the attributes of an object, or the key-value arguments.'''

        if __args is not None:
            if isinstance(__args, dict):
                d = __args.copy()
            else:
                d = vars(__args).copy()
        else:
            d = {}
        
        d.update(kargs)

        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Struct.from_dict(v))
            else:
                setattr(self, k, v)

    @final
    def to_dict(self):
        '''translate the arguments into a dictionary.'''

        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    @final
    def override(self, items: dict, strict=True):
        '''over-write the arguments from a dictionary'''
        for k, v in items.items():
            try:
                temp = getattr(self, k)
            except AttributeError:
                if strict:
                    raise KeyError(f"can not override non-existing attribute '{k}'.")
                continue
            if isinstance(temp, Struct):
                if isinstance(v, dict):
                    temp.override(v)
                else:
                    if strict:
                        raise KeyError(f"can not override sub-struct '{k}' with a non-dict value.")
                    continue
            else:
                setattr(self, k, v)
    
    @staticmethod
    def from_dict(d: dict):
        args = Struct()
        args.set(d)
        return args

    def save(self, path: Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: Union[str, pathlib.Path]):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        with open(path.with_suffix('.json'), 'r') as f:
            self.override(json.load(f))


class Args(Struct):
    
    @classmethod
    def set_parser(cls, parser: argparse.ArgumentParser):
        pass

    def __init__(self, args: argparse.Namespace):
        pass
