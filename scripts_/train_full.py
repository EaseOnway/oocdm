from typing import Optional, Dict, Tuple, List, Literal


import numpy as np
import torch
import torch.nn as nn

from .train import Train
from alg.model import OOCModel
from core.causal_graph import ObjectOrientedCausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.model.mask_generator as mg
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors, EnvModel


class TrainFull(Train):
    
    ModelArgs = OOCModel.Args

    def __init__(self, env_id: str,
                 suffix='', run_id: Optional[str] = None, 
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 env_options={},
                 _continue=False):
        super().__init__(env_id, title='full'+suffix, run_id=run_id, train_args=train_args,
            model_args=model_args, env_options=env_options, _continue=_continue)
        
        model_args = model_args or self.ModelArgs()

        if model_args is None:
            model_args = self.ModelArgs()
        model = self.make_model(OOCModel, model_args)
        assert isinstance(model, OOCModel)
        self.model = model

        envinfo = self.envinfo
        device = self.device
        self.__maskgen_full = mg.FullMaskGenerator(envinfo, device)

    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors, 
                  obj_mask: NamedTensors, reward: torch.Tensor, eval=False
                    ) -> Tuple[float, ...]:
        model = self.model

        label = F.raws2labels(self.envinfo, next_state)
        dist = model.forward(attrs, self.__maskgen_full, obj_mask)
        ll = F.sum_logprob_by_class(F.logprob(dist, label), obj_mask)
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
    
    def update(self):
        super().update()

    def record(self, train_log, eval_log: utils.Log):
        # show info
        print(f"evaluation loss: {eval_log.mean}")
        print(f"- loglikelihood: {eval_log['ll'].mean}")

        self.writer.add_scalar('loss', eval_log.mean, self.global_step)
        self.writer.add_scalar('loglikelihood', eval_log['ll'].mean, self.global_step)

        self.save_result(('train', 'loglikelihood'), eval_log['ll'].mean)
