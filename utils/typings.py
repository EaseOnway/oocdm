from typing import Dict, Optional, Union, Sequence, Tuple, Callable, List, TypeVar, Literal
from torch import Tensor
from torch.distributions import Distribution
from numpy import ndarray


_T = TypeVar('_T')


NamedTensors = Dict[str, Tensor]
NamedArrays = Dict[str, ndarray]
NamedValues = Dict[str, _T]
NamedDistributions = Dict[str, Distribution]
Names = Tuple[str, ...]
ShapeLike = Union[int, List[int], Tuple[int, ...]]
Shape = Tuple[int, ...]
ObjectValues = Dict[str, Dict[str, _T]]
ObjectArrays = Dict[str, Dict[str, ndarray]]
ObjectTensors = Dict[str, Dict[str, Tensor]]
ObjectDistributions = Dict[str, Dict[str, Distribution]]
Role = Literal['all', 'state', 'action']
ObjectMask = Dict[str, Tensor]
EnvModel = Callable[[ObjectTensors, Optional[ObjectMask]],
                    Tuple[ObjectDistributions, Distribution]]
TransitionModel = Callable[[ObjectTensors, Optional[ObjectMask]],
                           ObjectDistributions]
RewardModel = Callable[[ObjectTensors, Optional[ObjectMask]],
                        Distribution]
