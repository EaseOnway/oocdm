from typing import Optional, Dict, Tuple, List, Literal, Type

import numpy as np
import torch
import torch.nn as nn
import json
import argparse
import tensorboardX
import os
import time
import abc
import stable_baselines3 as sb3

from .base import Experiment, EnvArgs
from alg.model.base import RLModel
from core import DType, CausalGraph
from alg.buffer import ObjectOrientedBuffer
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors, EnvModel


_REWARD = 'reward'
_RETURN = 'return'


class OptimArgs(utils.Struct):

    def __init__(self,
                 optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam',
                 lr=1e-3,
                 max_grad_norm: Optional[float] = 0.5,
                 batch_size=512,
                 args: dict = {}):
        self.optimizer: str = optimizer
        self.lr: float = lr
        self.max_grad_norm: Optional[float] = max_grad_norm
        self.batch_size: int = batch_size
        self.args = utils.Struct.from_dict(args)

    def get_optimizer(self, network: nn.Module, maximize=False
                      ) -> torch.optim.Optimizer:
        optimizer = self.optimizer
        if optimizer == "adam" or optimizer == "adamw":
            return torch.optim.Adam(
                network.parameters(), self.lr, **vars(self.args), maximize=maximize)  # type: ignore
        elif optimizer == "adamw":
            return torch.optim.AdamW(
                network.parameters(), self.lr, **vars(self.args), maximize=maximize)  # type: ignore
        elif optimizer == "sgd":
            return torch.optim.SGD(
                network.parameters(), self.lr, **vars(self.args), maximize=maximize)  # type: ignore
        else:
            raise ValueError(f"unsupported algorithm: {optimizer}")


def _optim_step(args: OptimArgs, network: torch.nn.Module,
                opt: torch.optim.Optimizer):
    if args.max_grad_norm is not None:
        torch.nn.utils.clip_grad.clip_grad_norm_(
            network.parameters(), args.max_grad_norm)
    # for name, pa in network.named_parameters():
    #     if torch.any(torch.isnan(pa.grad)):
    #         assert False
    opt.step()
    opt.zero_grad()


class Train(Experiment):
    use_existing_path = False
    
    class Args(utils.Struct):
         def __init__(self, 
            n_iter=100,
            device: str = 'cuda',
            buffer_size=10000,
            n_batch=100,
            n_timestep=500,
            n_timestep_warmup: Optional[int] = None,
            n_batch_warmup: Optional[int] = None,
            n_batch_eval: Optional[int] = None,
            n_batch_quick_eval: Optional[int] = 8,
            collector: Literal['random', 'ppo'] = 'random',
            collector_update_size = 2048,
            collector_update_interval = 2048,
            use_noised_data: bool = False,
            use_incomplete_data: bool = False,
            optim_args=OptimArgs(),
        ):
            self.n_iter: int = n_iter
            self.device = device
            self.buffer_size: int = buffer_size
            self.n_batch: int = n_batch
            self.n_timestep: int = n_timestep
            self.n_timestep_warmup: Optional[int] = n_timestep_warmup
            self.n_batch_warmup: Optional[int] = n_batch_warmup
            self.n_batch_eval: Optional[int] = n_batch_eval
            self.n_batch_quick_eval: Optional[int] = n_batch_quick_eval
            self.collector = collector
            self.collector_update_size = collector_update_size
            self.collector_update_interval = collector_update_interval
            self.use_noised_data = use_noised_data
            self.use_incomplete_data = use_incomplete_data
            self.optim = optim_args

    def __init__(self, env_id: str,
                 title: Optional[str] = None,
                 run_id: Optional[str] = None,
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 env_options: dict = {},
                 _continue=False):
        super().__init__()
        
        if title is not None:
            self.title = title

        env_args = EnvArgs(env_id, **env_options)
        self.new_experiment(env_args, run_id)

        train_args = train_args or self.Args()
        self.save_args('train', train_args)

        self.n_iter: int = train_args.n_iter
        self.device = torch.device(train_args.device)
        self.buffer_size: int = train_args.buffer_size
        self.n_batch: int = train_args.n_batch
        self.n_timestep: int = train_args.n_timestep
        self.n_timestep_warmup: Optional[int] = train_args.n_timestep_warmup
        self.n_batch_warmup: Optional[int] = train_args.n_batch_warmup
        self.n_batch_eval: Optional[int] = train_args.n_batch_eval
        self.n_batch_quick_eval: Optional[int] = train_args.n_batch_quick_eval
        self.__continue = _continue
        self.optim = train_args.optim

        self.__networks: Dict[str, nn.Module] = {}
        self.__optimizers: Dict[str, torch.optim.Optimizer] = {}

        # config
        self.dtype = DType.Real.torch

        # buffer
        self.buffer = ObjectOrientedBuffer(
            self.buffer_size, self.envinfo)
        self.global_step: int = 0

        # log
        self.log_path = self.path / 'logs'
        self.writer = tensorboardX.SummaryWriter(str(self.log_path))

        # ppo
        self.__collector = train_args.collector
        if self.__collector == 'ppo':
            gymenv = self.env._unwrap_env()
            self.ppo = sb3.PPO('MlpPolicy', gymenv, batch_size=128, verbose=1)
            self.__ppo_size = train_args.collector_update_size
            self.__ppo_interval = train_args.collector_update_interval

        # i_step
        self.i_iter = -1

    def make_model(self, modelcls: Type[RLModel], args: utils.Struct,
                   key='model', maximize=False):
        self.save_args(key, args)
        model = modelcls(self.env, args, self.device, self.dtype)
        self.add_network(key, model, maximize)
        return model

    def add_network(self, key: str, network: nn.Module, maximize=False):
        if key in self.__networks:
            raise ValueError(f"{key} is already used")
        F.init_params(network)
        self.__networks[key] = network
        self.__optimizers[key] = self.optim.get_optimizer(
            network, maximize)

    def get_network(self, key: str):
        return self.__networks[key]

    def get_optimizer(self, key: str):
        return self.__networks[key]

    def _set_network_status(self, train=False):
        for network in self.__networks.values():
            network.train(train)

    def optim_step(self, key: str):
        _optim_step(self.optim,
                    self.__networks[key], self.__optimizers[key])

    def __get_action_random(self, obs, info: dict) -> ObjectArrays:
        return self.env.action_space.sample()

    def __get_action_ppo(self, obs, info: dict) -> ObjectArrays:
        action, _ = self.ppo.predict(info['core_obs'])
        return self.env._wrap_action(action)

    def collect(self, buffer: ObjectOrientedBuffer, n_sample: int):
        '''collect real-world samples into the buffer, and compute returns'''
        self._set_network_status(train=False)

        if self.__collector == 'ppo':
            get_action = self.__get_action_ppo
        else:
            get_action = self.__get_action_random
        
        log = utils.Log()
        episodic_return = 0.

        attrs, info = self.env.reset()

        _timer = time.time()
        for i_sample in range(n_sample):
          
            _global_step = self.global_step + i_sample
            if self.__collector == 'ppo' and _global_step > 0 and \
                    _global_step % self.__ppo_interval == 0:
                print("Start Training PPO...")
                self.ppo.learn(self.__ppo_size)

            # print progress every second
            _new_timer = time.time()
            if _new_timer - _timer >= 1:
                print(f"Collecting samples... ({i_sample}/{n_sample})")
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

            # write buffer
            buffer.add(attrs, next_state, reward)

            # reset if done
            if truncated or terminated:
                attrs, info = self.env.reset()
            else:
                attrs = next_state

            # truncate the last sample
            if i_sample == n_sample - 1:
                truncated = True

        return log

    def warmup(self, n_sample: int):
        return self.collect(self.buffer, n_sample)

    # @utils.decorators.timed("batch")
    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors,
                  obj_mask: NamedTensors, reward: torch.Tensor, eval=False) -> Tuple[float, ...]:
        raise NotImplementedError

    def log_batch(self, log: utils.Log, *scalars: float):
        raise NotImplementedError

    def print_batch(self, i_batch: int, n_batch: int, *scalars: float):
        raise NotImplementedError

    def fit_epoch(self, buffer: ObjectOrientedBuffer, log: utils.Log, eval=False):
        '''
        train network with fixed causal graph.
        '''
        batch_size = self.optim.batch_size
        for batch in buffer.epoch(batch_size, self.device):
            batch_info = self.fit_batch(*batch, eval)
            self.log_batch(log, *batch_info)

    def fit(self, n_batch: int):
        # fit causal equation
        train_log = utils.Log()
        batch_size = self.optim.batch_size
        interval = n_batch // 20

        # print(f"setting up normalizer")
        # self.update_variable_normalizer()

        print(f"start fitting...")
        self._set_network_status(train=True)
        for i_batch in range(n_batch):
            batch = self.buffer.sample_batch(batch_size, self.device)
            batchinfo = self.fit_batch(*batch, eval=False)
            self.log_batch(train_log, *batchinfo)

            if interval == 0 or i_batch % interval == 0:
                self.print_batch(i_batch, n_batch, *batchinfo)

        return train_log

    def eval(self, n_batch: Optional[int]):
        self._set_network_status(train=False)
        eval_log = utils.Log()
        if n_batch is None:  # use the whole buffer
            self.fit_epoch(self.buffer, eval_log, eval=True)
        else:
            batch_size = self.optim.batch_size
            for i_batch in range(n_batch):
                batch = self.buffer.sample_batch(batch_size, self.device)
                batchinfo = self.fit_batch(*batch, eval=True)
                self.log_batch(eval_log, *batchinfo)
        return eval_log

    def save(self):
        for key, network in self.__networks.items():
            self.save_file(network.state_dict(), key, "nn")
        
        if self.__collector == 'ppo':
            self.ppo.save(self._file_path('ppo'))

    def load(self):
        for key, network in self.__networks.items():
            try:
                network.load_state_dict(self.load_file(key, "nn"))
            except Exception as e:
                print(f"Warning: error occurs in loading network {key}.")
                print(f"Error Message: {e}")
        
        # if self.__collector == 'ppo':
        #     self.ppo.set_parameters(self._file_path('ppo'))

    def __restore(self):
        values, steps, wall_time = utils.torchutils.read_tensorboard_logfile(
            str(self.log_path), 'loss'
        )
        step = max(steps)
        self.global_step = step
        self.load()
        print("successfully loaded the existing model.")

    def __collect_log(self, log: utils.Log):
        # show info
        print(f"- reward: {log[_REWARD].mean}")
        print(f"- return: {log[_RETURN].mean}")
        self.writer.add_scalar('reward', log[_REWARD].mean, self.global_step)
        self.writer.add_scalar('return', log[_RETURN].mean, self.global_step)

    def record(self, train_log: utils.Log, eval_log: utils.Log):
        # show info
        pass

    def update(self):
        pass

    def __train_until_better(self, target_loss: float, train_log: utils.Log):
        assert self.n_batch_quick_eval is not None
        print(f"Do extra training to make sure that the loss < {target_loss}")

        n = 0
        bunch = 8
        batch_size = self.optim.batch_size

        max_n = self.n_batch

        while True:
            quick_eval_log = self.eval(self.n_batch_quick_eval)
            if np.min(quick_eval_log.data) <= target_loss:
                print("Done!")
                break
            elif n >= max_n:
                print("Reached batch limit.")
                break
            for i in range(bunch):
                self._set_network_status(train=True)
                batch = self.buffer.sample_batch(batch_size, self.device)
                batchinfo = self.fit_batch(*batch, eval=False)
                self.log_batch(train_log, *batchinfo)
                n += 1
                if i == bunch - 1:
                    self.print_batch(n + self.n_batch,
                                     self.n_batch, *batchinfo)

    def end_iter(self) -> bool:
        return self.i_iter >= self.n_iter

    def main(self):
        if self.__continue:
            self.__restore()

        if self.n_timestep_warmup is not None:
            print(f"---------------------warm up---------------------")
            collect_log = self.warmup(self.n_timestep_warmup)
            self.__collect_log(collect_log)
            self.global_step += self.n_timestep_warmup
            if self.n_batch_warmup is not None and not self.__continue:
                self.update()
                train_log = self.fit(self.n_batch_warmup)
                eval_log = self.eval(self.n_batch_eval)
                self.record(train_log, eval_log)
            self.save()

        self.i_iter = 0
        while not self.end_iter():
            print(f"---------------------iter {self.i_iter}---------------------")

            # collect samples
            collect_log = self.collect(self.buffer, self.n_timestep)
            self.__collect_log(collect_log)
            self.global_step += self.n_timestep

            self.update()

            # before training, do a quick evalutaion
            if self.n_batch_quick_eval is not None:
                quick_eval_log_1 = self.eval(
                    self.n_batch_quick_eval)
            else:
                quick_eval_log_1 = None

            train_log = self.fit(self.n_batch)

            # after training, make sure loss decreases
            if self.n_batch_quick_eval is not None:
                assert quick_eval_log_1 is not None
                self.__train_until_better(quick_eval_log_1.mean, train_log)

            eval_log = self.eval(self.n_batch_eval)
            self.record(train_log, eval_log)
            self.save()

            self.i_iter += 1


class TrainCausal(Train):

    @abc.abstractproperty
    def causal_graph(self) -> CausalGraph:
        raise NotImplementedError

    def compute_graph_accuracy(self):
        try:
            g_ = self.env.causal_graph()
        except NotImplementedError:
            return None

        g = self.causal_graph
        temp: np.ndarray = (g.matrix == g_.matrix)
        acc = np.count_nonzero(temp) / temp.size

        return acc

    @abc.abstractmethod
    def update_causal_graph(self, timer: utils.Timer):
        raise NotImplementedError

    def update(self):
        timer = utils.Timer('causal discovery')
        self.update_causal_graph(timer)
        t = timer.last_record
        acc = self.compute_graph_accuracy()
        if acc is None:
            print("[causal discovery] time = %.4f" % t)
        else:
            print("[causal discovery] accuracy = %.4f%%, time = %.4f" %
                  (acc*100, t))
            self.save_result("causal-graph-accuracy", acc)
            self.writer.add_scalar(
                'causal_graph_accuracy', acc, self.global_step)
        self.save_result('causal_discovery_time', t)
