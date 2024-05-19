from core import *
import numpy as np
import torch

from typing import Optional

from core.taskdata import TaskData


c_block = EnvObjClass('Block')
c_block.declear_field('x1', Normal())
c_block.declear_field('x2', Normal())
c_block.declear_field('x3', Normal())
c_block.declear_field('a', Normal(), role='action')

c_total = EnvObjClass('Total')
c_total.declear_field('x1', Normal())
c_total.declear_field('x2', Normal())
c_total.declear_field('x3', Normal())
c_total.declear_field('t', Normal())


MATRIX_TRANSIT = np.array([
    [1, 0., 0., -0.2,],
    [0.5, 1.0, 0., 0.,],
    [0., 0.25, 0.75, 1.0]
])

INIT_DISTRUBUTION = {  # (loc, scale)
    'x1': (1.0, 1.0),
    'x2': (0., 1.0),
    'x3': (0., 1.0),
}

INIT_DISTRUBUTION_OOD = {
    'x1': (0., 1.0),
    'x2': (1., 2.0),
    'x3': (0., 2.0),
}


class BlockEnv(ObjectOrientedEnv):

    classes = (c_block, c_total)

    def __init__(self,
                 truncate_step = 50,
                 n_block: Optional[int] = None,
                 ood=False,
                 noised=False,
                 incomplete=False,
                 **options):
        super().__init__(truncate_step, **options)

        self.n_block: Optional[int] = n_block
        self.ood: bool = ood
        self.transition_noise: float = 0.01

        self.init_distribution = INIT_DISTRUBUTION_OOD if self.ood \
            else INIT_DISTRUBUTION

        self._noised = noised
        self._incomplete = incomplete
        self._observation_noise: float = 0.01
        self._incomplete_prob: float= 0.01

    @property
    def task_family(self) -> Optional[str]:
        if self.n_block is None:
            return None
        else:
            return str(self.n_block)

    def init_task(self, data: TaskData):
        if self.n_block is None:
            n_block = np.random.randint(low=1, high=9)
        else:
            n_block = self.n_block

        init_distribution = self.init_distribution

        data.init_instances(Block=n_block, Total=1)
        
        for fieldname in (self.info.c('Block').fieldnames('state')):
            data['Block', fieldname] = np.random.normal(
                *init_distribution[fieldname], size=data.count('Block'))
        
        return {}

    def transit(self, data):
        x1 = data['Block', 'x1']
        x2 = data['Block', 'x2']
        x3 = data['Block', 'x3']
        a = np.tanh(data['Block', 'a'])
        
        temp = np.stack((x1, x2, x3, a), axis=0)  # 4 * n_obj
        x = MATRIX_TRANSIT @ temp
        x = x + self.transition_noise * np.random.randn(*x.shape)

        for attr in c_block.fieldnames('state'):
            data['Total', attr] = 0.5 * data['Total', attr] + \
                0.5 * np.max(data['Block', attr], axis=0)
        
        data['Total', 't'] += np.random.normal(1.0, scale=self.transition_noise)

        for i, fieldname in enumerate(c_block.fieldnames('state')):
            data['Block', fieldname] = x[i]
        
        return {}
    
    def observe(self, data: TaskData):
        obs = super().observe(data)

        if self._noised:
            noise_scale = self._observation_noise
            for c in self.classes:
                n =  data.count(c.name)
                for fieldname in c.fieldnames('state'):
                    obs[c.name][fieldname] += np.random.normal(0., noise_scale, n)

        if self._incomplete:
            mask = np.random.rand(3, data.count('Block')) < self._incomplete_prob
            obs['Block']['x1'] = np.where(mask[0], -100., obs['Block']['x1'])
            obs['Block']['x2'] = np.where(mask[1], -100., obs['Block']['x2'])
            obs['Block']['x3'] = np.where(mask[2], -100., obs['Block']['x3'])
        
        return obs

    def reward(self, data: TaskData, next_data: TaskData):
        return 0.
    
    def paralleled_reward(self, data: ParalleledTaskData,
                          next_data: ParalleledTaskData) -> torch.Tensor:
        return torch.zeros(data.n_parallel, dtype=DType.Real.torch, device=data.device)

    def terminated(self, data: TaskData) -> bool:
        temp = np.stack(
            [data['Block', fieldname] for fieldname in c_block.fieldnames()],
            axis=1)  # (obj, 4)
        return bool(np.any(np.abs(temp) > 1000))

    def paralleled_terminated(self, data: ParalleledTaskData) -> torch.Tensor:
        # (thread, obj, 4)
        temp = torch.stack(
            [data['Block', fieldname] for fieldname in c_block.fieldnames()],
            dim=2)
        return torch.any(temp.flatten(1), dim=1)

    def object_oriented_causal_graph(self) -> ObjectOrientedCausalGraph:
        g = ObjectOrientedCausalGraph(self.info)
        
        g.local_matrices['Block'][:] = (MATRIX_TRANSIT != 0.)
        for name in ('x1', 'x2', 'x3'):
            g.set_local_edge_by_name('Total', name, name)
            g.set_global_edge_by_name(('Block', name), ('Total', name))
        g.set_local_edge_by_name('Total', 't', 't')
        return g
