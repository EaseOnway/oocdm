

from __future__ import annotations

from typing import Tuple, Type
import gymnasium as gym
from gymnasium import spaces
from argparse import ArgumentParser, Namespace
import copy

import numpy as np
import torch
import abc

from typing import Final, Any, Dict, Iterable, Optional, final, Literal
from utils.typings import Role, ObjectArrays
from .objcls import EnvObjClass
from .taskdata import TaskData, ParalleledTaskData
from .envinfo import EnvInfo, Taskinfo
from .vtype import DType
from .causal_graph import CausalGraph, ObjectOrientedCausalGraph


class ObjectOrientedEnv(abc.ABC, gym.Env):
    '''
    The class that represents an Object-Oriented MDP, or a family of Object-Oriented MDPs that
    share the same class set and transition pattern.
    '''
    
    classes: Tuple[EnvObjClass, ...] = ()

    def __init__(self, truncate_step: Optional[int] = None, **options):
        self.truncate_step: Optional[int] = truncate_step
        self._options = options
        
        if len(options) > 0:
            print("Warning: unused environment options!")
            for k, v in options.items():
                print(f"- {k} = {v}")

        self.info: Final = EnvInfo(self.classes)
        self.taskinfo: Taskinfo
        self.__data: TaskData
        self.observation_space: spaces.Space
        self.action_space: spaces.Space
        self.attribute_space: spaces.Space

        self.__timestep: int
    
    @final
    @property
    def time_step(self):
        '''the step count in the episode'''
        return self.__timestep
    
    @property
    def task_family(self) -> Optional[str]:
        return None
    
    @property
    def data(self):
        return self.__data
    
    def count(self, clsname: str):
        return self.__data.count(clsname)
    
    def _unwrap_env(self) -> gym.Env:
        return self
    
    def _unwrap_action(self, action: ObjectArrays) -> Any:
        return action
    
    def _wrap_action(self, action: Any) -> ObjectArrays:
        return action

    def __set_spaces(self):
        classes = self.info.classes
        self.observation_space = spaces.Dict({
            c.name: spaces.Dict({
                fieldname: c.field_vtypes[fieldname].space(self.count(c.name))
                for fieldname in c.fieldnames('state')
            })
            for c in classes
            if self.count(c.name) > 0
        })
        self.action_space = spaces.Dict({
            c.name: spaces.Dict({
                fieldname: c.field_vtypes[fieldname].space(self.count(c.name))
                for fieldname in c.fieldnames('action')
            })
            for c in classes
            if self.count(c.name) > 0
        })
        self.attribute_space = spaces.Dict({
            c.name: spaces.Dict({
                fieldname: c.field_vtypes[fieldname].space(self.count(c.name))
                for fieldname in c.fieldnames('all')
            })
            for c in classes
            if self.count(c.name) > 0
        })

    @abc.abstractmethod
    def init_task(self, data: TaskData) -> dict:
        '''
        initialize objects of the environment and begins a new episode.

        Args:
            data (TaskData): the object data to be initialized.
            **options: optional arguments that specifies how to initialize the task.
        
        Returns: info dict
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def transit(self, data: TaskData) -> dict:
        '''Retrurns: info dict'''
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, data: TaskData, next_data: TaskData):
        raise NotImplementedError

    @abc.abstractmethod
    def terminated(self, data: TaskData) -> bool:
        raise NotImplementedError

    def paralleled_terminated(self, data: ParalleledTaskData) -> torch.Tensor:
        out = torch.zeros(data.n_parallel, dtype=torch.bool, device=data.device)
        for i in range(data.n_parallel):
            data_i = data.get_thread(i)
            out[i] = self.terminated(data_i)
        return out

    def paralleled_reward(self, data: ParalleledTaskData, next_data: ParalleledTaskData) -> torch.Tensor:
        out = torch.zeros(data.n_parallel, dtype=DType.Real.torch, device=data.device)
        for i in range(data.n_parallel):
            data_i = data.get_thread(i)
            next_data_i = next_data.get_thread(i)
            out[i] = self.reward(data_i, next_data_i)
        return out

    # def paralleled_transit(self, data: ParalleledTaskData) -> torch.Tensor:
    #     raise NotImplementedError(f"[{self.__class__.__name__}] environment does not support paralleled transition.")
    #     rewards = torch.zeros(data.n_parallel, dtype=DType.Real.torch, device=data.device)
    #     for i in range(data.n_parallel):
    #         data_i = data.get_thread(i)
    #         reward = self.transit(data_i)
    #         data.set_thread(i, data_i)
    #         rewards[i] = reward
    #     return rewards

    @final
    def reset(self):
        self.__timestep = 0
        self.__data = TaskData(self.info)

        info = self.init_task(self.__data)

        self.__set_spaces()
        self.taskinfo = Taskinfo(self.info, self.__data._counts)
        return self.observe(self.__data), info

    def truncated(self):
        return (self.__timestep == self.truncate_step)

    @final
    def step(self, action: Any):
        '''
        return:
        - obs: observation of next state
        - reward: (float)
        - terminated: (bool)
        - truncated: (bool)
        - info: dict
        '''
        
        self.__data.set_action(action)

        data = self.__data.deepcopy()  # copy current data

        info = self.transit(data)
        reward = self.reward(self.__data, data)
        obs = self.observe(data)

        self.__timestep += 1
        self.__data = data
        terminated = self.terminated(self.__data)
        truncated = self.truncated()

        return obs, reward, terminated, truncated, info

    def observe(self, data: TaskData):
        return data.get('state')

    def causal_graph(self) -> CausalGraph:
        g_ = self.object_oriented_causal_graph()
        g = CausalGraph(self.taskinfo)
        g.load_object_oriented_graph(g_)
        return g

    def object_oriented_causal_graph(self) -> ObjectOrientedCausalGraph:
        raise NotImplementedError
