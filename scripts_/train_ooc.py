from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import torch
import torch.nn as nn

from .train import TrainCausal, Train
from core import DType, CausalGraph
from alg.model import OOCModel
from core.causal_graph import ObjectOrientedCausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.model.mask_generator as mg
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors, EnvModel


class TrainOOC(TrainCausal):

    ModelArgs = OOCModel.Args
    
    def __init__(self, env_id: str,
                 suffix='',
                 run_id: Optional[str] = None, 
                 train_args: Optional['Train.Args'] = None,
                 model_args: Optional[utils.Struct] = None,
                 p_edge=0.9,
                 causal_threshold=0.1,
                 n_iter_stable: Optional[int] = None,
                 env_options={},
                 _continue=False):
        super().__init__(env_id, title='ooc'+suffix, run_id=run_id, train_args=train_args,
            model_args=model_args, env_options=env_options, _continue=_continue)

        self.p_edge: float = p_edge
        self.causal_thres: float = causal_threshold
        self.n_iter_stable = n_iter_stable

        self.__last_graph_change = -1

        # mask generators
        envinfo, device = self.envinfo, self.device
        self.__maskgen_graph = mg.GraphMaskGenerator(envinfo, device)
        self.__maskgen_full = mg.FullMaskGenerator(envinfo, device)
        self.__maskgen_random = mg.RandomMaskGenerator(
            envinfo, device, self.p_edge)
        self.__maskgen_dropped = mg.DroppedMaskGenerator(envinfo, device)

        # causal graph
        self.__set_causal_graph(ObjectOrientedCausalGraph(envinfo))

        if model_args is None:
            model_args = self.ModelArgs()
        model = self.make_model(OOCModel, model_args)
        assert isinstance(model, OOCModel)
        self.model = model

    @property
    def causal_graph(self):
        g = CausalGraph(self.env.taskinfo)
        g.load_object_oriented_graph(self.__causal_graph)
        return g

    def __set_causal_graph(self, g: ObjectOrientedCausalGraph):
        self.__causal_graph = g
        self.__maskgen_graph.load_graph(g)

    def __loglikeli(self, model: OOCModel, encodings: NamedTensors,
                    label: ObjectTensors, maskgen: mg.MaskGenerator, obj_mask: NamedTensors):
        dist = model.inferer.forward(encodings, maskgen, obj_mask)
        logprob = F.logprob(dist, label)

        return F.sum_logprob_by_class(logprob, obj_mask)

    def fit_batch(self, attrs: ObjectTensors, next_state: ObjectTensors,
                  obj_mask: NamedTensors, reward: torch.Tensor, eval=False
                  ) -> Tuple[float, ...]:
        model = self.model
        label = F.raws2labels(self.envinfo, next_state)
        encodings = model.variable_encoder.forward(attrs)

        ll_graph = self.__loglikeli(
            model, encodings, label, self.__maskgen_graph, obj_mask)
        ll_full = self.__loglikeli(
            model, encodings, label, self.__maskgen_full, obj_mask)
        ll_random = self.__loglikeli(
            model, encodings, label, self.__maskgen_random, obj_mask)

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
            global_step=self.global_step
        )

        self.save_result(('train', 'loglikelihood'), eval_log['graph'].mean)

    def update_causal_graph(self, timer: utils.Timer):
        g = self.get_causal_graph()
        timer.record()
        print(g)
        self.__set_causal_graph(g)
    
    def update(self):
        old_g = self.__causal_graph

        super().update()

        new_g = self.__causal_graph
        if not new_g == old_g:
            print("Causal graph changes.")
            self.__last_graph_change = self.i_iter
        else:
            print(f"Causal graph has been stable for {self.i_iter - self.__last_graph_change} iterations.")
    
    def end_iter(self) -> bool:
        return super().end_iter() or (
            self.n_iter_stable is not None and self.buffer.full and
            self.i_iter - self.__last_graph_change >= self.n_iter_stable
        )

    @torch.no_grad()
    def get_causal_graph(self):
        self.model.train(False)
        envinfo = self.envinfo

        g = ObjectOrientedCausalGraph(envinfo)

        def logp(d: torch.distributions.Distribution, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
            # d: batchsize * n_obj * size
            # x: batchsize * n_obj * size
            # mask: batchsize * n_obj

            temp = torch.sum(d.log_prob(x), dim=2)  # batchsize * n_obj
            temp = torch.masked_select(temp, mask)
            return temp.detach().cpu().numpy()

        logp_full = {j: [] for j in range(envinfo.n_field('state'))}
        logp_global = {(j, i): []
                       for j in range(envinfo.n_field('state'))
                       for i in range(envinfo.n_field())}
        logp_local = {(c.name, j, i): []
                      for c in envinfo.classes
                      for j in range(c.n_field('state'))
                      for i in range(c.n_field())}

        for batch in self.buffer.epoch(self.optim.batch_size, self.device):
            attrs, next_state, objmasks, _ = batch

            encodings = self.model.variable_encoder.forward(attrs)
            labels = F.raws2labels(envinfo, next_state)

            # full
            out = self.model.inferer.forward(
                encodings, self.__maskgen_full, objmasks)
            for j, (clsname, fieldname) in enumerate(envinfo.fields('state')):
                d = out[clsname][fieldname]
                label = labels[clsname][fieldname]
                m = objmasks[clsname]
                logp_full[j].append(logp(d, label, m))

            # drop global edges
            for i in range(envinfo.n_field()):
                self.__maskgen_dropped.drop(i)
                out = self.model.inferer.forward(
                    encodings, self.__maskgen_dropped, objmasks)
                for j, (clsname, fieldname) in enumerate(envinfo.fields('state')):
                    d = out[clsname][fieldname]
                    label = labels[clsname][fieldname]
                    m = objmasks[clsname]

                    logp_global[j, i].append(logp(d, label, m))

            # drop local edges
            global_encs = self.model.inferer.get_global_encodings(
                encodings, self.__maskgen_full)
            attnmasks = self.model.inferer.get_attn_masks(encodings, objmasks)
            for c in envinfo.classes:
                attnmask = attnmasks[c.name]
                objmask = objmasks[c.name]
                for i in range(c.n_field()):
                    self.__maskgen_dropped.drop(i, clsname=c.name)
                    x = encodings[c.name]
                    out = self.model.inferer.infer_one_class(
                        c.name, global_encs, x, self.__maskgen_dropped, attnmask)
                    for j, fieldname in enumerate(c.fieldnames('state')):
                        d = out[fieldname]
                        label = labels[c.name][fieldname]
                        logp_local[c.name, j, i].append(
                            logp(d, label, objmask))

        # determine edges
        threshold = self.causal_thres

        # determine global edges
        for j in range(envinfo.n_field('state')):
            for i in range(envinfo.n_field()):
                cmi = np.mean(np.concatenate(logp_full[j])) - \
                    np.mean(np.concatenate(logp_global[j, i]))  # type: ignore
                print("(%s.%s) -> (%s.%s): %f" %
                      (*envinfo.fields()[i], *envinfo.fields('state')[j], cmi))
                if cmi > threshold:
                    g.set_edge(i, j, None)

        # determine local edges
        for c in envinfo.classes:
            for j, name_j in enumerate(c.fieldnames('state')):
                j_ = envinfo.field_index(c.name, name_j, 'state')
                for i, name_i in enumerate(c.fieldnames()):
                    cmi = np.mean(np.concatenate(logp_full[j_])) - \
                        np.mean(np.concatenate(logp_local[c.name, j, i]))  # type: ignore
                    print("%s.(%s -> %s): %f" % (c.name, name_i, name_j, cmi))
                    if cmi > threshold:
                        g.set_edge(i, j, c.name)

        return g

    def save(self):
        super().save()
        self.save_file(self.__causal_graph.state_dict(),
                        "causal-graph", 'json')

    def load(self):
        super().load()
        g = ObjectOrientedCausalGraph(self.envinfo)
        g.load_state_dict(self.load_file("causal-graph", 'json'))
        self.__set_causal_graph(g)
