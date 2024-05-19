from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import torch
import torch.nn as nn

from .train import Train
from core import DType
from alg.buffer import ObjectOrientedBuffer
from alg.model.gnn_model import GNNModel
from alg.cem import cross_entropy_method, CEMArgs
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors


class TrainGNN(Train):

    ModelArgs = GNNModel.Args

    def __init__(self, env_id: str,
                 suffix='', run_id: Optional[str] = None, 
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 env_options={},
                 _continue=False):
        super().__init__(env_id, title='gnn'+suffix, run_id=run_id, train_args=train_args,
            model_args=model_args, env_options=env_options, _continue=_continue)

        if model_args is None:
            model_args = self.ModelArgs()
        model = self.make_model(GNNModel, model_args)
        assert isinstance(model, GNNModel)
        self.model = model
    
    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors, 
                    obj_mask: NamedTensors, reward: torch.Tensor, eval=False
                    ) -> Tuple[float, ...]:
        label = F.raws2labels(self.envinfo, next_state)
        
        variables = self.model.forward(attrs)
        state = self.taskinfo.get_obj_distr(variables)
        logprob = F.logprob(state, label)
        ll = F.sum_logprob_by_class(logprob, obj_mask)
        loss = -ll
        
        if not eval:
            loss.backward()
            self.optim_step('model')

        return float(loss), float(ll)

    def log_batch(self, log: utils.Log, *scalars: float):
        loss, ll = scalars
        log(loss)
        log['ll'] = ll
    
    def print_batch(self, i_batch: int, n_batch: int, *scalars: float):
        loss, ll = scalars
        print(f"loss of batch {i_batch}/{n_batch}: {loss}")
        print(f"- loglikelihood: {ll}")

    def record(self, train_log, eval_log: utils.Log):
        # show info
        print(f"evaluation loss: {eval_log.mean}")
        print(f"- loglikelihood: {eval_log['ll'].mean}")

        self.writer.add_scalar('loss', eval_log.mean, self.global_step)
        self.writer.add_scalar('loglikelihood', eval_log['ll'].mean, self.global_step)

        self.save_result(('train', 'loglikelihood'), eval_log['ll'].mean)
