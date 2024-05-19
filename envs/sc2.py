from typing import List, Tuple, Optional, Dict, Union, Iterable


from pysc2.agents.base_agent import BaseAgent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.named_array import NamedNumpyArray
import numpy as np
import abc
import enum
import torch
from core import *
from utils.typings import NamedValues, ObjectArrays


class OtherUnits(enum.IntEnum):
    MineralShard = 1680


UnitType = Union[units.Neutral, units.Protoss, units.Terran, units.Zerg, OtherUnits]


UNIT_TYPES: Dict[int, UnitType] = {}
for c in (units.Neutral, units.Protoss, units.Terran, units.Zerg, OtherUnits):
    for utype in c:
        UNIT_TYPES[utype.value] = utype


class Agent(BaseAgent):
    def step(self, obs):
        super().step(obs)
        return actions.FUNCTIONS.no_op()


class Unit:

    def __init__(self, array: NamedNumpyArray):
        self.unit_type = UNIT_TYPES[int(array.unit_type)]
        self.health = int(array.health)
        self.alliance = int(array.alliance)
        self.unit_id = int(array.tag)
        self.x = int(array.x)
        self.y = int(array.y)

    def __str__(self):
        return "Unit<%x (%s) at (%d,%d)>" % (self.unit_id, self.unit_type, self.x, self.y)
    
    def __repr__(self):
        return str(self)


class Observation:
    def __init__(self, pysc2_timestep):
        __units_by_id: Dict[int, Unit] = {}
        for array in pysc2_timestep.observation.raw_units:
            unit = Unit(array)
            __units_by_id[unit.unit_id] = unit
        
        self.__units_by_id = __units_by_id
        self.done: bool = pysc2_timestep.last()

        self.__units_of_type: Dict[UnitType, List[Unit]] = {}
        for u in self.__units_by_id.values():
            try:
                self.__units_of_type[u.unit_type].append(u)
            except KeyError:
                self.__units_of_type[u.unit_type] = [u]
    
    def unit(self, uid: int):
        return self.__units_by_id[uid]
    
    def units(self, utype: Optional[UnitType] = None) -> Iterable[Unit]:
        if utype is None:
            return self.__units_by_id.values()
        else:
            try:
                return self.__units_of_type[utype]
            except KeyError:
                return ()

    def count_unit(self, utype: Optional[UnitType] = None) -> int:
        if utype is None:
            return len(self.__units_by_id)
        else:
            try:
                return len(self.__units_of_type[utype])
            except KeyError:
                return 0


class SC2Env(ObjectOrientedEnv):
    map_name: str
    player_index: int = 0

    VTYPE_POSITION = Normal(2, low=-99, high=99, transform=False, dtype=DType.Integar)
    VTYPE_HEALTH = Normal(low=-1, high=999, transform=False, dtype=DType.Integar)

    c_unit = EnvObjClass('Unit')
    c_unit.declear_field('position', VTYPE_POSITION)
    c_unit.declear_field('health', VTYPE_HEALTH)
    c_unit.declear_field('alive', Boolean(), default=True)

    def __init__(self, truncate_step: Optional[int] = 200, **options):
        super().__init__(truncate_step=truncate_step, **options)
        self.core = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self._create_players(),
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                # use_feature_units=True,
                use_raw_actions=True, use_raw_units=True),
        )

    @classmethod
    @abc.abstractmethod
    def _create_players(cls) -> list:
        raise NotImplementedError
    
    @abc.abstractmethod
    def _clsmap(self, utype: UnitType) -> Optional[str]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def _assign_varaibles(self, obs: Observation, data: TaskData):
        raise NotImplementedError

    def _get_obs(self, core_timesteps):
        return Observation(core_timesteps[self.player_index])

    def init_task(self, data: TaskData):
        core_timesteps = self.core.reset()
        self._core_obs = self._get_obs(core_timesteps)
        self._unit_ids: Dict[str, List[int]] = {c.name: [] for c in self.classes}
        
        for u in self._core_obs.units():
            cname = self._clsmap(u.unit_type)
            if cname is not None:
                self._unit_ids[cname].append(u.unit_id)
        counts = {cname: len(uids) for cname, uids in self._unit_ids.items()}
        data.init_instances(**counts)
        self._assign_varaibles(self._core_obs, data)

        return {}

    def _uid(self, clsname: str, obj_index: int):
        return self._unit_ids[clsname][obj_index]
    
    def _u(self, clsname: str, obj_index: int, obs: Observation):
        try:
            return obs.unit(self._uid(clsname, obj_index))
        except KeyError:
            return None
    
    @abc.abstractmethod
    def translate_action(self, data: TaskData) -> List:
        raise NotImplementedError
    
    def transit(self, data: TaskData):
        action = self.translate_action(data)

        core_timesteps = self.core.step([action])
        self._core_obs = self._get_obs(core_timesteps)
        
        self._assign_varaibles(self._core_obs, data)

        return {}

    def skip(self):
        core_timesteps = self.core.step([])
        core_obs = self._get_obs(core_timesteps)
        data = self.data.deepcopy()  # copy current data
        self._assign_varaibles(core_obs, data)
        terminated = self.terminated(data)
        return terminated
    
    def truncated(self):
        return super().truncated() or self._core_obs.done

    def close(self):
        super().close()
        self.core.close()


class DefeatZerglingBaneling(SC2Env):
    map_name = "DefeatZerglingsAndBanelings"
    players = [sc2_env.Agent(sc2_env.Race.terran)],

    c_zergling = EnvObjClass('Zergling', SC2Env.c_unit)
    c_baneling = EnvObjClass('Baneling', SC2Env.c_unit)
    
    c_marine = EnvObjClass('Marine', SC2Env.c_unit)
    c_marine.declear_field('action', Categorical(5), role='action')
    classes = (c_zergling, c_baneling, c_marine)

    @classmethod
    def _create_players(cls) -> list:
        return [sc2_env.Agent(sc2_env.Race.terran)]
    
    def __move_command(self, i_marine: int, move: int):
        u = self._u('Marine', i_marine, self._core_obs)
        if u is None:
            return None
        elif move == 0:
            return None
        else:
            x = u.x + [0,1,-1,0,0][move]
            y = u.y + [0,0,0,1,-1][move]
            # x = np.clip(x, 0, 84)
            # y = np.clip(y, 0, 84)

        return actions.RAW_FUNCTIONS.Move_pt('now', u.unit_id, (x, y))
    
    __clsmap: Dict[UnitType, str] = {
        units.Terran.Marine: c_marine.name,
        units.Zerg.Zergling: c_zergling.name,
        units.Zerg.Baneling: c_baneling.name}

    def _clsmap(self, utype: UnitType) -> str:
        return  self.__clsmap[utype]
    
    def _assign_varaibles(self, obs: Observation, data: TaskData):
        for cname in ('Marine', 'Zergling', 'Baneling'):
            for i in range(data.count(cname)):
                u = self._u(cname, i, obs)
                if u is None:
                    data[cname, i, 'alive'] = False
                    data[cname, i, 'health'] = 0
                else:
                    data[cname, i, 'position'] = (u.x, u.y)
                    data[cname, i, 'health'] = u.health
                    data[cname, i, 'alive'] = True

    def translate_action(self, data: TaskData) -> List:
        out = []
        for i_marine in range(data.count('Marine')):
            a = self.__move_command(i_marine, int(data['Marine', i_marine, 'action']))
            if a is not None:
                out.append(a)
        return out

    def reward(self, data: TaskData, next_data: TaskData) -> float:
        h0 = np.sum(data['Zergling', 'health']) + np.sum(data['Baneling', 'health'])
        h1 = np.sum(next_data['Zergling', 'health']) + np.sum(next_data['Baneling', 'health'])
        return h0 - h1
    
    def paralleled_reward(self, data: ParalleledTaskData, next_data: ParalleledTaskData):
        h0 = torch.sum(data['Zergling', 'health'], dim=1) + \
            torch.sum(data['Baneling', 'health'], dim=1)
        h1 = torch.sum(next_data['Zergling', 'health'], dim=1) + \
            torch.sum(next_data['Baneling', 'health'], dim=1)
        return h0 - h1

    def terminated(self, data: TaskData) -> bool:
        return False
    
    def paralleled_terminated(self, data: ParalleledTaskData) -> torch.Tensor:
        return torch.zeros(data.n_parallel, dtype=torch.bool, device=data.device)


class CollectMineralShards(SC2Env):
    map_name = 'CollectMineralShards'

    c_marine = EnvObjClass('Marine')
    c_marine.declear_field('position', SC2Env.VTYPE_POSITION)
    c_marine.declear_field('move', Categorical(5), role='action')

    c_mineral = EnvObjClass('MineralShard')
    c_mineral.declear_field('position', SC2Env.VTYPE_POSITION)
    c_mineral.declear_field('collected', Boolean(), default=False)

    classes = (c_marine, c_mineral)

    @classmethod
    def _create_players(cls) -> list:
        return [sc2_env.Agent(sc2_env.Race.terran)]
    
    __clsmap: Dict[UnitType, str] = {
        units.Terran.Marine: c_marine.name,
        OtherUnits.MineralShard: c_mineral.name,
    }

    def _clsmap(self, utype: UnitType) -> str:
        return self.__clsmap[utype]
    
    def _assign_varaibles(self, obs: Observation, data: TaskData):
        cname = 'Marine'
        for i in range(data.count(cname)):
            u = self._u(cname, i, obs)
            if u is None:
                assert False
            else:
                data[cname, i, 'position'] = (u.x, u.y)

        cname = 'MineralShard'
        for i in range(data.count(cname)):
            u = self._u(cname, i, obs)
            if u is None:
                data[cname, i, 'collected'] = True
            else:
                data[cname, i, 'position'] = (u.x, u.y)
                data[cname, i, 'collected'] = False
        

    def __move_command(self, i_marine: int, move: int):
        u = self._u('Marine', i_marine, self._core_obs)
        if u is None:
            assert False
        elif move == 0:
            return None
        else:
            x = u.x + [0,1,-1,0,0][move]
            y = u.y + [0,0,0,1,-1][move]
            # x = np.clip(x, 0, 84)
            # y = np.clip(y, 0, 84)

        return actions.RAW_FUNCTIONS.Move_pt('now', u.unit_id, (x, y))
    
    def translate_action(self, data: TaskData) -> List:
        out = []
        for i_marine in range(data.count('Marine')):
            a = self.__move_command(i_marine, int(data['Marine', i_marine, 'move']))
            if a is not None:
                out.append(a)
        return out

    def reward(self, data: TaskData, next_data: TaskData) -> float:
        temp = np.logical_xor(data['MineralShard', 'collected'], next_data['MineralShard', 'collected'])
        return float(np.count_nonzero(temp)) - 0.1
    
    def paralleled_reward(self, data: ParalleledTaskData, next_data: ParalleledTaskData) -> torch.Tensor:
        temp = torch.logical_xor(data['MineralShard', 'collected'], next_data['MineralShard', 'collected'])
        return torch.count_nonzero(temp, dim=1).to(dtype=DType.Real.torch) - 0.1

    def terminated(self, data: TaskData) -> bool:
        return bool(np.all(data['MineralShard', 'collected']))

    def paralleled_terminated(self, data: ParalleledTaskData):
        return data['MineralShard', 'collected'].all(dim=1)
