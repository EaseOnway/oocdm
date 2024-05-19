from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from .base import Experiment
from .train import TrainCausal, Train
from core import DType
from alg.buffer import ObjectOrientedBuffer
from alg.model.gru_model import GRUModel
from core.causal_graph import CausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors
from utils.fcit import test_rl


class TrainGRU(TrainCausal):
    
    ModelArgs = GRUModel.Args

    def __init__(self, env_id: str,
                 suffix='', run_id: Optional[str] = None, 
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 causal_threshold = 0.1,
                 n_job_fcit = 4,
                 max_size_fcit: Optional[int] = None,
                 env_options={},
                 _continue=False):
        super().__init__(env_id, title='gru'+suffix, run_id=run_id, train_args=train_args,
            model_args=model_args, env_options=env_options, _continue=_continue)

        if model_args is None:
            model_args = self.ModelArgs()
        model = self.make_model(GRUModel, model_args)
        assert isinstance(model, GRUModel)
        self.model = model
    
        self.causal_thres: float = causal_threshold
        self.n_job_fcit: int = n_job_fcit
        self.max_size_fcit = max_size_fcit
        
        # causal graph
        self.__causal_graph = CausalGraph(self.taskinfo)

    @property
    def causal_graph(self):
        return self.__causal_graph
    
    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors, 
                    obj_mask: NamedTensors, reward: torch.Tensor, eval=False
                    ) -> Tuple[float, ...]:
        label = F.raws2labels(self.envinfo, next_state)
        
        distr = self.model.forward(attrs, self.causal_graph)
        distr = self.taskinfo.get_obj_distr(distr)

        ll = F.sum_logprob_by_class(F.logprob(distr, label), obj_mask)
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

    def update_causal_graph(self, timer: utils.Timer):
        g = self.get_causal_graph()
        timer.record()
        print(g)
        self.__causal_graph = g

    @torch.no_grad()
    def get_causal_graph(self):
        g = CausalGraph(self.taskinfo)
        
        if self.max_size_fcit is not None and self.max_size_fcit < len(self.buffer):
            attrs, next_state, _, _ = self.buffer.sample_batch(
                self.max_size_fcit, device=torch.device('cpu'), replace=False)
        else:
            attrs, next_state, _, _ = self.buffer.fetch_tensors(slice(None), device=torch.device('cpu'))

        envinfo = self.envinfo
        attrs_: ObjectArrays = {
            clsname: {
                fieldname: envinfo.v(clsname, fieldname).raw2input(value).numpy()
                for fieldname, value in c_data.items()
            } 
            for clsname, c_data in attrs.items()
        }
        next_state_: ObjectArrays = {
            clsname: {
                fieldname: envinfo.v(clsname, fieldname).raw2input(value).numpy()
                for fieldname, value in c_data.items()
            } 
            for clsname, c_data in next_state.items()
        }

        xs = [attrs_[var.clsname][var.fieldname][:, var.obj_index]
              for var in self.taskinfo.input_variables]
        ys = [next_state_[var.clsname][var.fieldname][:, var.obj_index]
              for var in self.taskinfo.output_variables]

        p_values = test_rl(xs, ys, self.n_job_fcit)

        for (i, j), p in p_values.items():
            vi = self.taskinfo.input_variables[i]
            vj = self.taskinfo.output_variables[j]
            assert p is not None
            if p < self.causal_thres:
                print(repr(vi), '->', repr(vj), ': assureance =', 1-p)
                g.set_edge(i, j)
        
        return g
    
    def update(self):
        if self.i_iter < 0: 
            super().update()
        else:
            pass

    def save(self):
        super().save()
        self.save_file(self.__causal_graph.state_dict(), "causal-graph", 'json')
    
    def load(self):
        super().load()

        g = CausalGraph(self.taskinfo)
        g.load_state_dict(self.load_file("causal-graph", 'json'))
        self.__causal_graph = g
