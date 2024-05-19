from typing import Optional, Dict, Tuple, List, Literal


import numpy as np
import torch
import torch.nn as nn
import json
import argparse
import os
import time
import pathlib
import stable_baselines3 as sb3

from .base import Experiment
from alg.model.base import RLModel
from .train import Train
from core import DType, Taskinfo
from alg.buffer import ObjectOrientedBuffer
from alg.model import *
from core.causal_graph import CausalGraph, ObjectOrientedCausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.model.mask_generator as mg
import alg.functional as F
import utils
from utils.typings import ObjectTensors, NamedTensors, TransitionModel


_REWARD = 'reward'
_RETURN = 'return'

class Test(Experiment):
    use_existing_path = True

    class Args(utils.Struct):
        def __init__(self,
            device: str = 'cuda',
            n_timestep: int = 1000,
            eval_n_sample: int = 10000,
            eval_batchsize: int = 512,
            cem_args = CEMArgs(),
        ):
            self.device = device
            self.n_timestep = n_timestep
            self.eval_n_sample = eval_n_sample
            self.eval_batchsize = eval_batchsize
            self.cem_args = cem_args

    def __init__(self,
        path: str,
        args: Optional['Test.Args'] = None,
        env_options: dict = {},
        collector: Literal['random', 'ppo'] = 'random',
        label: str = 'test',
        **collectargs
    ) -> None:
        super().__init__()

        args = args or self.Args()
        self.device = torch.device(args.device)
        self.dtype = DType.Real.torch
        self.n_timestep = args.n_timestep
        self.eval_n_sample = args.eval_n_sample
        self.eval_batchsize = args.eval_batchsize
        self.cem_args = args.cem_args
        self.label = label

        self.follow_experiment(path, env_options)

        model_args = utils.Struct.from_dict(self.load_args('model'))
        self.model = self.load_model(model_args)
        self.model.train(False)
        
        self.__collector: Literal['random', 'ppo'] = collector
        if collector == 'ppo':
            self.__ppo_size = collectargs['collector_update_size']
            self.__ppo_interval = collectargs['collector_update_interval']
            self.ppo = sb3.PPO('MlpPolicy', self.env._unwrap_env(), batch_size=128)
            try:
                ppo_file = collectargs['collector_file']
                self.ppo.set_parameters(str(self._file_path(ppo_file, 'zip')))
                print("successfully loaded PPO parameters.")
            except KeyError:
                pass
            
        self.setup()

    def setup(self):
        pass

    def load_model(self, model_args: utils.Struct) -> RLModel:
        raise NotImplementedError
    
    def reset_env(self):
        return self.env.reset()
        
    def __get_action_cem(self, obs, info):
        return cross_entropy_method(
            self.env, self.get_transition_model(), None, self.device, self.cem_args)
    
    def __get_action_random(self, obs, info):
        return self.env.action_space.sample()
    
    def __print_log(self, log: utils.Log):
        print(f"- average episode return: {log[_RETURN].mean}")
        print(f"- average step reward: {log[_REWARD].mean}")

    def collect(self, n_sample: int, actor: Literal['random', 'cem', 'ppo'] = 'random',
                buffer: Optional[ObjectOrientedBuffer] = None):
        '''collect real-world samples and compute returns'''

        if actor == 'cem':
            get_action = self.__get_action_cem
        elif actor == 'ppo':
            get_action = lambda _, info: \
                self.env._wrap_action(self.ppo.predict(info['core_obs'])[0])
        else:
            get_action = self.__get_action_random

        log = utils.Log()
        episodic_return = 0.

        attrs, info = self.reset_env()

        _timer = time.time()
        for i_sample in range(n_sample):
            
            if self.__collector == 'ppo' and self.__ppo_size \
                    and i_sample > 0 \
                    and i_sample % self.__ppo_interval == 0:
                print("Start Training PPO...")
                self.ppo.learn(self.__ppo_size)

            # print progress every second
            _new_timer = time.time()
            if _new_timer - _timer >= 5:
                print(f"Collecting samples... ({i_sample}/{n_sample})")
                self.__print_log(log)
                _timer = _new_timer

            # interact with the environment
            a = get_action(attrs, info)

            # merge state and action
            for clsname, acls in a.items():
                attrs[clsname].update(acls)
            
            next_state, reward, terminated, truncated, info = self.env.step(a)

            # record information
            episodic_return += reward
            log[_REWARD] = reward
            
            if truncated or terminated:
                log[_RETURN] = episodic_return
                episodic_return = 0.
            
            if buffer is not None:
                buffer.add(attrs, next_state, reward)
            
            # reset if done
            if truncated or terminated:
                attrs, info = self.reset_env()
            else:
                attrs = next_state

        return log
    
    @torch.no_grad()
    def __eval_batch(self, attrs: ObjectTensors, next_state: ObjectTensors, 
                     objmask: NamedTensors, reward: torch.Tensor, model: TransitionModel,
                    ):
        s = model(attrs, objmask)
        
        label = F.raws2labels(self.envinfo, next_state)
        logprob = F.sum_logprob_by_class(F.logprob(s, label), objmask)

        return float(logprob)

    def eval_model(self):
        '''
        train network with fixed causal graph.
        '''
        buffer = ObjectOrientedBuffer(self.eval_n_sample, self.envinfo)

        actor = self.__collector

        print("collecting samples for model evaluation")
        self.collect(self.eval_n_sample, actor, buffer)

        log = utils.Log()
        batch_size = self.eval_batchsize
        model = self.get_transition_model()
        for batch in buffer.epoch(batch_size, self.device):
            logp = self.__eval_batch(*batch, model)
            log['logp'] = logp
        
        print("model evaluation:")
        print(f"- loglikelihood: {log['logp'].mean}")

        return log

    def get_transition_model(self) -> TransitionModel:
        raise NotImplementedError

    def main(self):
        label = self.label

        eval_log = self.eval_model()
        self.save_result((label, 'loglikelihood'), eval_log['logp'].mean)

        collect_log = self.collect(self.n_timestep, 'cem')
        print("finished:")
        self.__print_log(collect_log)
        self.save_result((label, 'reward'), collect_log[_REWARD].mean)
        self.save_result((label, 'return'), collect_log[_RETURN].mean)


class TestOOC(Test):
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = OOCModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def load_causal_graph(self):
        causal_graph = ObjectOrientedCausalGraph(self.envinfo)
        with open(self.path / 'causal-graph.json', 'r') as f:
            d = json.load(f)
        causal_graph.load_state_dict(d)
        return causal_graph

    def setup(self):
        super().setup()
        # mask generators
        self.__maskgen_graph = mg.GraphMaskGenerator(self.envinfo, self.device)
        self.__maskgen_full = mg.FullMaskGenerator(self.envinfo, self.device)

        # causal graph
        causal_graph = self.load_causal_graph()
        self.__maskgen_graph.load_graph(causal_graph)

    def get_transition_model(self):
        self.model: OOCModel
        return self.model.make_transition_model(self.__maskgen_graph)


class TestFull(Test):
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = OOCModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def setup(self):
        super().setup()
        # mask generators
        self.__maskgen_full = mg.FullMaskGenerator(self.envinfo, self.device)

    def get_transition_model(self):
        self.model: OOCModel
        return self.model.make_transition_model(self.__maskgen_full)


class TestMLP(Test):
    
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = MLPModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def get_transition_model(self):
        assert isinstance(self.model, MLPModel)
        return self.model.make_transition_model()


class TestCDL(Test):
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = CDLModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def load_causal_graph(self):
        causal_graph = CausalGraph(self.taskinfo)
        with open(self.path / 'causal-graph.json', 'r') as f:
            d = json.load(f)
        causal_graph.load_state_dict(d)
        return causal_graph

        # causal graph
    def setup(self):
        self.causal_graph = self.load_causal_graph()

    def get_transition_model(self):
        self.model: CDLModel
        return self.model.make_transition_model('graph', causal_graph=self.causal_graph)


class TestGRU(Test):
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = GRUModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def load_causal_graph(self):
        causal_graph = CausalGraph(self.taskinfo)
        with open(self.path / 'causal-graph.json', 'r') as f:
            d = json.load(f)
        causal_graph.load_state_dict(d)
        return causal_graph

    def setup(self):
        super().setup()
        self.causal_graph = self.load_causal_graph()

    def get_transition_model(self):
        self.model: GRUModel
        return self.model.make_transition_model(causal_graph=self.causal_graph)
    

class TestTICSA(Test):
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = TISCAModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def get_transition_model(self):
        self.model: TISCAModel
        return self.model.make_transition_model(tau=0)


class TestGNN(Test):
    
    def load_model(self, model_args: utils.Struct) -> RLModel:
        m = GNNModel(self.env, model_args, self.device, self.dtype)
        state_dict = torch.load(self._file_path('model', 'nn'))
        m.load_state_dict(state_dict, strict=False)
        return m

    def get_transition_model(self):
        assert isinstance(self.model, GNNModel)
        return self.model.make_transition_model()