import os
from typing import Any, Optional, Final, final, List, Type, Dict, Union, Sequence, Literal
import numpy as np
import random
import torch
import json
from pathlib import Path
import abc
import argparse
import absl.app

from core import ObjectOrientedEnv
from ._get_env import get_env
import utils


class EnvArgs(utils.Struct):

    @final
    def __init__(self, env_id: str, **options):
        self.env_id = env_id
        self.options = utils.Struct(**options)

    def launch(self):
        envcls = get_env(self.env_id)
        env = envcls(**self.options.to_dict())
        print(f"Successfully launched environment: {self.env_id}")
        return env


class Experiment(abc.ABC):
    
    ROOT: Final = Path('./experiments/')
    # ROOT: Final = Path('/data1/yuzhongwei/experiments/')

    @classmethod
    def _create_parser(cls):
        return argparse.ArgumentParser(prog=cls.__name__, description=cls.__doc__)

    @property
    def path(self) -> Path:
        try:
            return self.__path
        except AttributeError:
            raise AttributeError("path is not setted")
    
    @path.setter
    def path(self, path: Path):
        self.__path = path
    
    @property
    def env(self) -> ObjectOrientedEnv:
        try:
            return self.__env
        except AttributeError:
            raise AttributeError("environment is not setted")
    
    @env.setter
    def env(self, env: ObjectOrientedEnv):
        self.__env = env
        env.reset()
    
    @property
    def envinfo(self):
        return self.env.info
    
    @property
    def taskinfo(self):
        return self.env.taskinfo
    
    @staticmethod
    def seed(x: int):
        torch.manual_seed(x)
        torch.cuda.manual_seed_all(x)
        np.random.seed(x)
        random.seed(x)

    def save_file(self, o: object, name: str, fmt: Optional[str] = None):
        path = self._file_path(name, fmt)
        if fmt == 'json':
            with path.open('w') as f:
                json.dump(o, f, indent=4)
        else:
            torch.save(o, path)

        print(f"successfully saved: {path}")

    def load_file(self, name: str, fmt: Optional[str] = None):
        path = self._file_path(name, fmt)
        if not path.exists():
            raise FileNotFoundError(f"'{path}' does not exists.")
        if fmt == 'json':
            with path.open('r') as f:
                o = json.load(f)
        else:
            o = torch.load(path)

        print(f"successfully loaded: {path}")
        return o
    
    def save_value(self, filename: str,
                   keys: Union[str, Sequence[str]], value,
                   action: Literal['set', 'append', 'add'] = 'set'):
        d = self.load_value(filename)
        if isinstance(keys, str):
            keys = [keys]
        
        if isinstance(value, utils.Struct):
            value = value.to_dict()
        
        temp = d
        for k in keys[:-1]:
            if isinstance(temp, dict):
                if k not in temp:
                    temp[k] = {}
                temp = temp[k]
            else:
                raise ValueError(f"Argument path is blocked by non-dictionary items.")
            
        k = keys[-1]
        if action == 'set':
            temp[k] = value
        elif action == 'add':
            if k not in temp:
                temp[k] = value
            else:
                temp[k] += value
        elif action == 'append':
            if k not in temp:
                temp[k] = [value]
            elif not isinstance(temp[k], list):
                raise ValueError(f"target item is not a list")
            else:
                temp[k].append(value)
        else:
            assert False

        self.save_file(d, filename, 'json')
    
    def load_value(self, filename: str, *keys: str):
        try:
            d = self.load_file(filename, 'json')
        except FileNotFoundError:
            d = {}
        
        for k in keys:
            assert isinstance(d, dict)
            d = d[k]

        return d
        
    def load_args(self, *keys: str) -> Any:
        '''load arguments from args.json'''

        return self.load_value('args', *keys)

    def save_args(self, keys: Union[str, Sequence[str]], value):
        '''save arguments to args.json'''

        self.save_value('args', keys, value)

    def save_result(self, keys: Union[str, Sequence[str]], value,
                    action: Literal['set', 'append', 'add'] = 'set'):
        '''save arguments to args.json'''

        self.save_value('results', keys, value, action)

    def new_experiment(self, env_args: EnvArgs, run_id: Optional[str]):

        print("setting up experiment")
        env = env_args.launch()
        
        env_name = env_args.env_id if env.task_family is None \
            else f"{env_args.env_id}-{env.task_family}"
        
        parent = Experiment.ROOT / env_name / self.title
        if not parent.exists():
            os.makedirs(parent)
        
        # get run_id
        if run_id is None or len(run_id) == 0:
            all_runs = os.listdir(parent)
            i = 0
            while True:
                i += 1
                run_id = "run-%d" % i
                if run_id not in all_runs:
                    break
        
        # experiment path
        path = parent / run_id
        if not path.exists():
            print("creating experiment directory at", path)
            os.makedirs(path)
        else:
            print(f"{path} already exists. Existing Arguments will be removed.")
            while True:
                temp = input("Are you sure to proceed ? [y / n]: ")
                if temp == 'n':
                    raise FileExistsError("cannot initalize experiment directory")
                elif temp == 'y':
                    break
            args_path = (path / 'args').with_suffix('.json')
            if args_path.exists():
                os.remove(args_path)
        
        self.path = path
        self.env = env

        self.save_args('env', env_args)

    def follow_experiment(self, path: Union[str, Path], env_options: dict):

        print("following experiment")
        
        if isinstance(path, str):
            path = Path(path)
        
        self.path = path
        
        d = self.load_args('env')
        env_args = EnvArgs(d['env_id'], **d['options'])

        env_args.options.set(env_options)  # override options
        env = env_args.launch()

        self.env = env


    @final
    def _file_path(self, name: str, fmt: Optional[str] = None):
        path = self.path / name
        if fmt is not None:
            path = path.with_suffix('.' + fmt)
        return path
    
    def __init__(self, *args, **kargs) -> None:
        self.title: str = self.__class__.__name__.lower()

    @abc.abstractmethod
    def main(self):
        raise NotImplementedError

    @classmethod
    def run(cls, *args, **argv):
        script = cls(*args, **argv)
        
        def main(_):
            script.main()
        
        absl.app.run(main, [cls.__name__ + '.main'])
