from typing import Dict, Tuple, List, Optional, Callable
import torch

from core import ObjectOrientedEnv, EnvObjClass, TaskData, \
    ParalleledTaskData, DType, EnvInfo
from utils.typings import TransitionModel, ObjectTensors, ObjectDistributions, RewardModel
import utils
import alg.functional as F


ActionParameters = Dict[str, Dict[str, Dict[str, torch.Tensor]]]


class CEMArgs(utils.Args):
    def __init__(self, n_iter=5, n_sample=500, n_elite=100,
                 horizon=20, discount=0.95):
        self.n_iter: int = n_iter
        self.n_sample: int = n_sample
        self.n_elite: int = n_elite
        self.horizon: int = horizon
        self.gamma: float = discount

def new_action_params(data: TaskData, device: torch.device) -> ActionParameters:
    return {
        c.name: {
            fieldname: c.field_vtypes[fieldname].ptype.default_params(
                device, DType.Real.torch, data.count(c.name))
            for fieldname in c.fieldnames('action')
        }
        for c in data.info.classes
        if c.contains_action
    }


def get_distribution(envinfo: EnvInfo, params: ActionParameters) -> ObjectDistributions:
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).ptype(**attrparams)
            for fieldname, attrparams in clsparams.items()
        }
        for clsname, clsparams in params.items()
    }


def estimate_params(envinfo: EnvInfo, actions: ObjectTensors,
                    topk_indices: torch.Tensor) -> ActionParameters:
    params: ActionParameters = {
        clsname: {} for clsname, c_action in actions.items()}

    for clsname, clsactions in actions.items():
        for fieldname, attr in clsactions.items():
            v = envinfo.v(clsname, fieldname)
            label = v.raw2label(attr[topk_indices])
            params[clsname][fieldname] = v.ptype.estimate_params(label, 0)
    
    return params


def _show_history(history: List[ParalleledTaskData], i_thread):
    for i, step in enumerate(history):
        print(f"--------------step_{i}-------------")
        step.print_objects(i_thread, role='all')

def simulate(env: ObjectOrientedEnv,
             model: Callable[[ObjectTensors], Tuple[ObjectTensors, Optional[torch.Tensor]]],
             paramlist: List[ActionParameters], n_sample: int,
             gamma: float, device: torch.device):

    distrlist = [get_distribution(env.info, params) for params in paramlist]
    actionlist = [F.sample_raw(env.info, d, n_sample) for d in distrlist]
    
    data = env.data.parallel(n_sample, device)
    rewards = torch.zeros(n_sample, dtype=DType.Real.torch, device=device)
    terminated = torch.zeros(n_sample, dtype=torch.bool, device=device)

    _history = []
    
    discount = 1.
    with torch.no_grad():
        for action in actionlist:
            data.set_action(action)
            _history.append(data.deepcopy())
            state, r = model(data.observe('all'))
            data.set_attributes(state)
            if r is None:  # use ground truth
                r = env.paralleled_reward(_history[-1], data)
            r[terminated] = 0.
            rewards += r * discount
            discount *= gamma
            terminated = torch.logical_or(terminated, env.paralleled_terminated(data))

    return actionlist, rewards

 
def cross_entropy_method(env:ObjectOrientedEnv, dynamics: TransitionModel,
                         reward_model: Optional[RewardModel],
                         device: torch.device, args: CEMArgs):
    
    paramlist = [new_action_params(env.data, device)
                 for h in range(args.horizon)]
    
    def model(attrs: ObjectTensors):
        state_distr = dynamics(attrs, None)
        s = F.sample_raw(env.info, state_distr)
        if reward_model is None:
            r = None
        else:
            r_distr = reward_model(attrs, None)
            r = r_distr.sample()
        return s, r

    actionlist = None
    rewards = None
    for i_iter in range(args.n_iter):
        actionlist, rewards = simulate(env, model, paramlist, args.n_sample, args.gamma, device)
        indices = torch.topk(rewards, args.n_elite, sorted=False).indices
        paramlist = [estimate_params(env.info, actions, indices) for actions in actionlist]

    assert actionlist is not None
    assert rewards is not None

    i_action = torch.argmax(rewards)
    action = F.apply(actionlist[0], lambda clsname, fieldname, x: x[i_action])

    # params = paramlist[0]
    # distr = get_distribution(env.info, params)
    # action = F.mode_raw(env.info, distr)

    return F.as_arrays(action)
