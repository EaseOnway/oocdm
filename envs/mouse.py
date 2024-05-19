from core import *
import numpy as np
import torch

from typing import Optional, List, Tuple


HEIGHT = 8
WIDTH = 8
DIM_NOISE = 3

_position_vtype = Normal(2, low=0, high=(WIDTH-1, HEIGHT-1),
                         dtype=DType.Integar, transform=False)

c_trap = EnvObjClass('Trap')
c_trap.declear_field('position', vtype=_position_vtype)
c_trap.declear_field('duration', vtype=Normal(low=0, dtype=DType.Integar))

c_monster = EnvObjClass('Monster')
c_monster.declear_field('position', vtype=_position_vtype)
c_monster.declear_field('noise', vtype=Normal(DIM_NOISE))

c_food = EnvObjClass('Food')
c_food.declear_field('position', vtype=_position_vtype)
c_food.declear_field('amount', vtype=Normal(low=0))

c_mouse = EnvObjClass('Mouse')
c_mouse.declear_field('position', vtype=_position_vtype)
c_mouse.declear_field('health', vtype=Normal())
c_mouse.declear_field('hunger', vtype=Normal(low=0, high=100))
c_mouse.declear_field('move', vtype=Categorical(5), role='action')


class MouseEnv(ObjectOrientedEnv):

    classes = (c_mouse, c_food, c_monster, c_trap)

    MOVEMENTS = np.array([
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ])
    
    def __init__(self, 
                 truncate_step: Optional[int] = 300,
                 ood = False,
                 task: Optional[str] = None,
                 **options):
        super().__init__(truncate_step, **options)

        self.ood: bool = ood
        
        self.__task_family: Optional[str] = task
        self.tasks = self.__generate_task_pool(task)

    @property
    def task_family(self) -> Optional[str]:
        return self.__task_family
    
    def __generate_task_pool(self, task: Optional[str]):
        tasks: List[Tuple[int, int, int]] = []

        if task is None:
            for n_food in range(3, 7):
                for n_monster in range(1, 6):
                    for n_trap in range(1, 6):
                        tasks.append((n_food, n_monster, n_trap))
        elif task in ('seen', 'unseen'):
            seed = 12345
            for n_food in range(3, 7):
                for n_monster in range(1, 6):
                    for n_trap in range(1, 6):
                        seed = (seed * 22695477 + 1) % (1 << 32)
                        x = seed / (1 << 32)
                        if (task == 'seen' and x < 0.5) or (task == 'unseen' and x >= 0.5):
                            tasks.append((n_food, n_monster, n_trap))
        elif len(task) == 3:
            try:
                n_food = int(task[0])
                n_monster = int(task[1])
                n_trap = int(task[2])
                tasks.append((n_food, n_monster, n_trap))
            except ValueError:
                pass
        if len(tasks) == 0:
            raise ValueError(
                f"Unsupported task: '{task}'. "
                "supported: 'seen', 'unseen', or any 3-number string, e.g, '444', '234'")
        return tasks
        

    def init_task(self, data: TaskData):
        i_task = np.random.randint(len(self.tasks))
        n_food, n_monster, n_trap = self.tasks[i_task]

        data.init_instances(Mouse=1, Food=n_food, Monster=n_monster, Trap=n_trap)
        self.init_objects(data)

        return {}
    
    @staticmethod
    def __restrict_position(x: np.ndarray):
        return np.clip(x, (0, 0), (WIDTH - 1, HEIGHT - 1))
    
    @staticmethod
    def __random_position(n: int):  
        return np.random.randint(low=(0, 0), high=(WIDTH, HEIGHT), size=(n, 2))  # type: ignore

    def init_objects(self, data: TaskData):
        ood: bool = self.ood

        for clsname in self.info.clsnames:
            data[clsname, 'position'] = self.__random_position(data.count(clsname))
       
        data['Mouse', 'health'] = 10
        data['Mouse', 'hunger'] = np.random.uniform(low=40., high=60., size=data.count('Mouse'))
        data['Trap', 'duration'] = np.random.randint(1, 5, size=data.count('Trap'))
        data['Monster', 'noise'] = np.random.normal(
            size=(data.count('Monster'), DIM_NOISE), scale=(3.0 if ood else 1.0))
        data['Food', 'amount'] = np.random.uniform(low=0., high=30., size=data.count('Food')) + (
            np.random.uniform(2., 4.) * data['Food', 'position'][:, 0] if not ood
            else np.random.uniform(2., 4.) * data['Food', 'position'][:, 1]
        )

    def transit(self, data):
        mouse = data.get_obj('Mouse', 0)

        food_eaten = np.all(data['Food', 'position'] == mouse.position, axis=1)
        n_food_eaten = np.sum(data['Food', 'amount'][food_eaten])
        trap_met = np.all(data['Trap', 'position'] == mouse.position, axis=1)
        monster_met = np.all(data['Monster', 'position'] == mouse.position, axis=1)
        n_monster_met = np.count_nonzero(monster_met)
        hungry = mouse.hunger < 25
        full = mouse.hunger >= 75

        # update mouse
        # - mouse position
        if np.any(data['Trap', 'duration'][trap_met] > 0):
            pass  # trapped
        else:
            mouse.position = self.__restrict_position(
                mouse.position + self.MOVEMENTS[mouse.move])
        
        # - mouse hunger
        mouse.hunger = max(0, mouse.hunger - 2.0)  # type: ignore
        mouse.hunger = min(mouse.hunger + n_food_eaten, 100)
        
        # - mouse health
        delta_health = - n_monster_met * 5.0
        if hungry:
            delta_health += -1
        if full:
            delta_health += 1

        mouse.health = min(10, mouse.health + delta_health)  # type: ignore

        # update traps
        data['Trap', 'duration'][trap_met] = np.fmax(
            data['Trap', 'duration'][trap_met] - 1, 0)

        # update monsters
        delta_position = self.MOVEMENTS[np.random.choice(5, size=data.count('Monster'))]
        data['Monster', 'position'] = self.__restrict_position(
            data['Monster', 'position'] + delta_position)
        data['Monster', 'noise'] = data['Monster', 'noise'] + \
            np.random.normal(size=(data.count('Monster'), DIM_NOISE), scale=0.1)
        
        # update food
        data['Food', 'amount'] += np.random.normal(1.0, scale=0.1, size=data.count('Food'))
        data['Food', 'amount'][food_eaten] = 0.
        data['Food', 'amount'] = np.clip(data['Food', 'amount'], 0., 50.0)

        return {}
    
    def object_oriented_causal_graph(self) -> ObjectOrientedCausalGraph:
        try:
            return self.__graph
        except AttributeError:
            g = ObjectOrientedCausalGraph(self.info)
            g.set_local_edge_by_name('Mouse', 'position', 'position')
            g.set_global_edge_by_name(('Trap', 'position'), ('Mouse', 'position'))
            g.set_global_edge_by_name(('Trap', 'duration'), ('Mouse', 'position'))
            g.set_local_edge_by_name('Mouse', 'move', 'position')
            g.set_local_edge_by_name('Mouse', 'hunger', 'hunger')
            g.set_global_edge_by_name(('Food', 'position'), ("Mouse", 'hunger'))
            g.set_global_edge_by_name(('Food', 'amount'), ("Mouse", 'hunger'))
            g.set_local_edge_by_name('Mouse', 'position', 'hunger')
            g.set_global_edge_by_name(('Monster', 'position'), ('Mouse', 'health'))
            g.set_local_edge_by_name('Mouse', 'position', 'health')
            g.set_local_edge_by_name('Mouse', 'health', 'health')
            g.set_local_edge_by_name('Mouse', 'hunger', 'health')
            g.set_local_edge_by_name('Trap', 'position', 'position')
            g.set_local_edge_by_name('Trap', 'duration', 'duration')
            g.set_local_edge_by_name('Trap', 'position', 'duration')
            g.set_global_edge_by_name(('Mouse', 'position'), ('Trap', 'duration'))
            g.set_local_edge_by_name('Monster', 'position', 'position')
            g.set_local_edge_by_name('Monster', 'noise', 'noise')
            g.set_local_edge_by_name('Food', 'position', 'position')
            g.set_local_edge_by_name('Food', 'amount', 'amount')
            g.set_local_edge_by_name('Food', 'position', 'amount')
            g.set_global_edge_by_name(('Mouse', 'position'), ('Food', 'amount'))
            self.__graph = g
            return g
    
    def reward(self, data: TaskData, next_data: TaskData):
        delta_hunger = next_data['Mouse', 0, 'hunger'] - data['Mouse', 0, 'hunger']  # type: ignore
        delta_health = next_data['Mouse', 0, 'health'] - data['Mouse', 0, 'health']  # type: ignore
        hunger = data['Mouse', 0, 'hunger']
        r = delta_hunger * 0.05 + delta_health + hunger * 0.01
        return float(r)
    
    def paralleled_reward(self, data: ParalleledTaskData, next_data: ParalleledTaskData) -> torch.Tensor:
        delta_hunger = next_data['Mouse', 0, 'hunger'] - data['Mouse', 0, 'hunger']
        delta_health = next_data['Mouse', 0, 'health'] - data['Mouse', 0, 'health']
        hunger = data['Mouse', 0, 'hunger']
        r = delta_hunger * 0.05 + delta_health + hunger * 0.01
        return r

    def terminated(self, data: TaskData) -> bool:
        return bool(data['Mouse', 0, 'health'] <= 0)

    def paralleled_terminated(self, data: ParalleledTaskData):
        return data['Mouse', 0, 'health'] <= 0
