from core import *
import numpy as np
import torch

from typing import Optional


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


class AsymBlockEnv(ObjectOrientedEnv):

    classes = (c_block, c_total)

    def __init__(self,
                 truncate_step = 50,
                 n_block: Optional[int] = None,
                 ood=False,
                 **options):
        super().__init__(truncate_step, **options)

        self.n_block: Optional[int] = n_block
        self.ood: bool = ood
        self.noise: float = 0.01

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

        init_distribution = INIT_DISTRUBUTION_OOD if self.ood else INIT_DISTRUBUTION

        data.init_instances(Block=n_block, Total=n_block)
        
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
        x = x + self.noise * np.random.randn(*x.shape)

        _, n_block = x.shape

        for attr in c_block.fieldnames('state'):
            for i in range(n_block):
                data['Total', i, attr] = 0.5 * data['Total', i, attr] + \
                    0.5 * np.max(data['Block', attr][:i+1], axis=0)
        
        data['Total', 't'] += np.random.normal(1.0, scale=self.noise, size=n_block)

        for i, fieldname in enumerate(c_block.fieldnames('state')):
            data['Block', fieldname] = x[i]
        
        return {}

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
