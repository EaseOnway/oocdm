from typing import Optional, Dict, Tuple, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from .base import Experiment
from .train import TrainCausal, Train
from core import DType
from alg.buffer import ObjectOrientedBuffer
from alg.model.ticsa_model import TISCAModel
from core.causal_graph import CausalGraph
from alg.cem import cross_entropy_method, CEMArgs
import alg.functional as F
import utils
from utils.typings import ObjectArrays, ObjectTensors, NamedTensors


class TrainTICSA(TrainCausal):

    ModelArgs = TISCAModel.Args

    def __init__(
        self,
        env_id: str, suffix="",
        run_id: Optional[str] = None,
        train_args: Optional["Train.Args"] = None,
        model_args: Optional[utils.Struct] = None,
        tau: Union[float, Tuple[float, float]] = (1.0, 0.1),
        norm_penalty: float = 1.0,
        env_options={},
        _continue=False,
    ):
        super().__init__(
            env_id,
            title="ticsa" + suffix,
            run_id=run_id,
            train_args=train_args,
            model_args=model_args,
            env_options=env_options,
            _continue=_continue,
        )

        model_args = model_args or self.ModelArgs()
        model = self.make_model(TISCAModel, model_args)
        assert isinstance(model, TISCAModel)
        self.model = model

        if isinstance(tau, (float, int)):
            pass
        else:
            self.__tau = tau[0]
            self.__tau_step = (tau[1] - tau[0]) / self.n_iter
            self.__tau_min = max(0, tau[0], tau[1])

        self.norm_penalty = norm_penalty

    @property
    def tau(self):
        return self.__tau

    @property
    def causal_graph(self):
        return self.model.extract_causal_graph("max")

    def update(self):
        super().update()
        self.__tau = max(self.__tau_min, self.__tau + self.__tau_step)

    def fit_batch(
        self,
        attrs: ObjectTensors,
        next_state: ObjectTensors,
        obj_mask: NamedTensors,
        reward: torch.Tensor,
        eval=False,
    ) -> Tuple[float, ...]:
        label = F.raws2labels(self.envinfo, next_state)

        variables, m = self.model.forward(attrs, self.tau)
        state = self.taskinfo.get_obj_distr(variables)
        logprob = F.logprob(state, label)

        ll = F.sum_logprob_by_class(logprob, obj_mask)
        norm = m.abs().sum(dim=1).mean()
        loss = -ll + self.norm_penalty * norm

        if not eval:
            loss.backward()
            self.optim_step("model")

        return float(loss), float(ll), float(norm)

    def log_batch(self, log: utils.Log, *scalars: float):
        loss, ll, norm = scalars
        log(loss)
        log["ll"] = ll
        log["L1-norm"] = norm

    def print_batch(self, i_batch: int, n_batch: int, *scalars: float):
        loss, ll, norm = scalars
        print(f"loss of batch {i_batch}/{n_batch}: {loss}")
        print(f"- loglikelihood: {ll}")
        print(f"- L1-norm of causal graph: {norm}")

    def record(self, train_log, eval_log: utils.Log):
        # show info
        print(f"evaluation loss: {eval_log.mean}")
        print(f"- loglikelihood: {eval_log['ll'].mean}")
        print(f"- L1-norm of causal graph: {eval_log['L1-norm'].mean}")

        self.writer.add_scalar("loss", eval_log.mean, self.global_step)
        self.writer.add_scalar("loglikelihood", eval_log["ll"].mean, self.global_step)
        self.writer.add_scalar("L1-norm", eval_log["L1-norm"].mean, self.global_step)

        self.save_result(("train", "loglikelihood"), eval_log["ll"].mean)

    def update_causal_graph(self, timer: utils.Timer):
        timer.record()
