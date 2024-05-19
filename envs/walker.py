from gymnasium.core import Env
from core import *
from envs.cores.walker import Walker2dEnv as _WalkerEnv
import numpy as np
import torch

from typing import Any, Optional, List, Tuple

_normal = Normal()

c_rotor = EnvObjClass('Rotor')
c_rotor.declear_field('angle', vtype=Normal())
c_rotor.declear_field('v_angle', vtype=Normal())
c_rotor.declear_field('torque', vtype=Normal(low=-1, high=1), role='action')

c_top = EnvObjClass('Top')
c_top.declear_field('z', vtype=Normal())
c_top.declear_field('angle', vtype=Normal())
c_top.declear_field('v_x', vtype=Normal())
c_top.declear_field('v_z', vtype=Normal())
c_top.declear_field('v_angle', vtype=Normal())

_STATEFIELD_INDICES = {
    ('Rotor', 'angle'): slice(2, 8),
    ('Rotor', 'v_angle'): slice(11, 17),
    ('Top', 'z'): 0,
    ('Top', 'angle'): 1,
    ('Top', 'v_x'): 8,
    ('Top', 'v_z'): 9,
    ('Top', 'v_angle'): 10,
}


class WalkerEnv(ObjectOrientedEnv):

    classes = (c_rotor, c_top)
    
    def __init__(self, truncate_step: Optional[int] = 500,
                 check_healthy=False, ctrl_cost=0.001,
                 render=False, **options):
        super().__init__(truncate_step, **options)

        self._check_healthy = check_healthy
        self._healthy_z_range = (0.8, 2)
        self._healthy_angle_range = (-1, 1)
        self._ctrl_cost = ctrl_cost
        self.__render = render
        self.gymcore: _WalkerEnv = _WalkerEnv(terminate_when_unhealthy=False,
                                              render_mode='human' if render else None)

    def init_task(self, data: TaskData):
        data.init_instances(Rotor=6, Top=1)
        core_obs, info = self.gymcore.reset()
        self._assign_variables(data, core_obs)

        info: dict
        info['core_obs'] = core_obs

        return info
    
    def _assign_variables(self, data: TaskData, core_obs):
        for (clsname, fieldname), idx in _STATEFIELD_INDICES.items():
            data[clsname, fieldname] = core_obs[idx]

    def transit(self, data):
        action = data['Rotor', 'torque']

        core_obs, _, terminated, truncated, info = self.gymcore.step(action)  # type: ignore
        self._assign_variables(data, core_obs)

        info: dict
        info['core_obs'] = core_obs

        return info
    
    def _is_healthy(self, data: TaskData):
        z, angle = data['Top', 0, 'z'], data['Top', 0, 'angle']
        z = float(z)
        angle = float(angle)
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle
        return is_healthy

    def _is_healthy_parallel(self, data: ParalleledTaskData):
        z, angle = data['Top', 0, 'z'], data['Top', 0, 'angle']
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        healthy_z = (min_z < z) & (z < max_z)
        healthy_angle = (min_angle < angle) & (angle < max_angle)
        is_healthy = healthy_z & healthy_angle
        return is_healthy
    
    def reward(self, data: TaskData, next_data: TaskData):
        is_healthy = self._is_healthy(data)
        healthy_reward = 1 if is_healthy else 0
        
        vx = data['Top', 0, 'v_x']
        forward_reward = vx

        torque = data['Rotor', 'torque']
        ctrl_cost = self._ctrl_cost * np.sum(np.square(torque))
        
        return float(healthy_reward + forward_reward - ctrl_cost)

    def paralleled_reward(self, data: ParalleledTaskData, next_data: ParalleledTaskData) -> torch.Tensor:
        is_healthy = self._is_healthy_parallel(data)
        healthy_reward = torch.where(is_healthy, 1., 0.)
        
        vx = data['Top', 0, 'v_x']
        forward_reward = vx

        torque = data['Rotor', 'torque']
        ctrl_cost = self._ctrl_cost * torch.sum(torch.square(torque), dim=1)
        
        return healthy_reward + forward_reward - ctrl_cost

    def terminated(self, data: TaskData) -> bool:
        return self._check_healthy and not self._is_healthy(data)

    def paralleled_terminated(self, data: ParalleledTaskData):
        if not self._check_healthy:
            return torch.zeros(data.n_parallel, device=data.device, dtype=torch.bool)
        else:
            return self._is_healthy_parallel(data)
    
    def _unwrap_env(self) -> Env:
        return _WalkerEnv(terminate_when_unhealthy=self._check_healthy,
                          truncate_step=self.truncate_step)

    def _unwrap_action(self, action: Any):
        return action['Rotor', 'torque']
    
    def _wrap_action(self, action: Any):
        return {'Rotor': {'torque': action}, 'Top': {}}
