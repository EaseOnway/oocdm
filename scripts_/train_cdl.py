from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from .base import Experiment
from .train import TrainCausal, Train
from core import DType
from alg.buffer import ObjectOrientedBuffer
from alg.model.cdl_model import CDLModel
from core.causal_graph import CausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors


class TrainCDL(TrainCausal):
    
    ModelArgs = CDLModel.Args

    def __init__(self, env_id: str, suffix='',
                 run_id: Optional[str] = None, 
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 causal_threshold = 0.1,
                 env_options={},
                 _continue=False):
        
        model_args = model_args or self.ModelArgs()
        if model_args.kernel == 'attn':
            title = 'cdla'
        else:
            title = 'cdl'
        title = title + suffix
        
        super().__init__(env_id, title=title, run_id=run_id, train_args=train_args,
            model_args=model_args, env_options=env_options, _continue=_continue)
        
        model = self.make_model(CDLModel, model_args)
        assert isinstance(model, CDLModel)
        self.model = model
    
        self.causal_thres: float = causal_threshold
        
        # causal graph
        self.__causal_graph = CausalGraph(self.taskinfo)

    @property
    def causal_graph(self):
        return self.__causal_graph

    def __loglikeli(self, model: CDLModel, enc: torch.Tensor,
            mask: Literal['full', 'graph', 'random'],
            label: ObjectTensors,
            obj_mask: NamedTensors):
        g = self.__causal_graph if mask == 'graph' else None
        dist = self.taskinfo.get_obj_distr(model.infer(enc, mask, None, g))
        logprob = F.logprob(dist, label)
        return F.sum_logprob_by_class(logprob, obj_mask)

    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors, 
                  obj_mask: NamedTensors, reward: torch.Tensor, eval=False
                  ) -> Tuple[float, ...]:
        model = self.model

        label = F.raws2labels(self.envinfo, next_state)
        encodings = model.variable_encoder.forward(attrs)

        ll_graph = self.__loglikeli(model, encodings, 'graph', label, obj_mask)
        ll_full = self.__loglikeli(model, encodings, 'full', label, obj_mask)
        ll_random = self.__loglikeli(model, encodings, 'random', label, obj_mask)

        loss = -(ll_graph + ll_full + ll_random) / 3.

        if not eval:
            loss.backward()
            self.optim_step('model')

        return float(loss), float(ll_graph), float(ll_full), float(ll_random)

    def log_batch(self, log: utils.Log, *scalars: float):
        loss, ll_graph, ll_full, ll_random = scalars
        log(loss)
        log['graph'] = ll_graph
        log['full'] = ll_full
        log['random'] = ll_random         
    
    def print_batch(self, i_batch: int, n_batch: int, *scalars: float):
        loss, ll_graph, ll_full, ll_random = scalars
        print(f"loss of batch {i_batch}/{n_batch}: {loss}")
        print(f"- loglikelihood (graph-mask): {ll_graph}")
        print(f"- loglikelihood (full-mask): {ll_full}")
        print(f"- loglikelihood (random-mask): {ll_random}")

    def record(self, train_log, eval_log):
        super().record(train_log, eval_log)

        print(f"evaluation loss: {eval_log.mean}")
        print(f"- - loglikelihood (graph-mask): {eval_log['graph'].mean}")
        print(f"- - loglikelihood (full-mask): {eval_log['full'].mean}")
        print(f"- - loglikelihood (random-mask): {eval_log['random'].mean}")

        self.writer.add_scalar('loss', eval_log.mean, self.global_step)
        self.writer.add_scalars(
            'loglikelihood',
            {
                'graph': eval_log['graph'].mean,
                'full': eval_log['full'].mean,
                'random': eval_log['random'].mean
            },
            global_step = self.global_step
        )

        self.save_result(('train', 'loglikelihood'), eval_log['graph'].mean)
    
    def update_causal_graph(self, timer: utils.Timer):
        g = self.get_causal_graph()
        timer.record()
        print(g)
        self.__causal_graph = g

    @torch.no_grad()
    def get_causal_graph(self):
        self.model.train(False)
        taskinfo = self.taskinfo

        g = CausalGraph(taskinfo)
        
        def logp(d: torch.distributions.Distribution, x: torch.Tensor) -> np.ndarray:
            # d: batchsize * size
            # x: batchsize * size
            # mask: batchsize * n_obj

            temp = torch.sum(d.log_prob(x), dim=1)  # batchsize * n_obj
            return temp.detach().cpu().numpy()

        logp_full= {j: [] for j in range(taskinfo.n_output_variable)}
        logp_drop= {(j, i): [] for j in range(taskinfo.n_output_variable)
                    for i in range(taskinfo.n_input_variable)}

        for batch in self.buffer.epoch(self.optim.batch_size, self.device):
            attrs, next_state, objmasks, _ = batch
            encodings = self.model.variable_encoder.forward(attrs)
            labels = F.raws2labels(self.envinfo, next_state)

            # full
            out = taskinfo.get_var_distr(self.model.infer(encodings, 'full'))
            for j, idx in enumerate(taskinfo.output_variables):
                d = out[idx]
                label = idx(labels)
                logp_full[j].append(logp(d, label))

            # drop global edges
            for i in range(taskinfo.n_input_variable):
                out = taskinfo.get_var_distr(self.model.infer(encodings, 'drop', i_drop=i))
                for j, idx in enumerate(taskinfo.output_variables):
                    d = out[idx]
                    label = idx(labels)
                    logp_drop[j, i].append(logp(d, label))

        # determine edges
        threshold = self.causal_thres

        # determine global edges
        for j in range(taskinfo.n_output_variable):
            for i in range(taskinfo.n_input_variable):
                cmi = np.mean(np.concatenate(logp_full[j])) - \
                    np.mean(np.concatenate(logp_drop[j, i]))  # type: ignore
                #  print("(%s) -> (%s): %f" % 
                #     (taskinfo.input_variables[i], taskinfo.output_variables[j], cmi))
                if cmi > threshold:
                    g.set_edge(i, j)

        return g

    def save(self):
        super().save()
        self.save_file(self.__causal_graph.state_dict(), "causal-graph", 'json')
    
    def load(self):
        super().load()

        g = CausalGraph(self.taskinfo)
        g.load_state_dict(self.load_file("causal-graph", 'json'))
        self.__causal_graph = g
