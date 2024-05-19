from typing import Any, Dict, List, Optional, Set, Tuple, final
import numpy as np
import abc
import torch
from utils import Shaping
from utils.typings import Shape, ShapeLike
import core.ptype as ptype
import enum

import gymnasium.spaces as spaces


class DType(enum.Enum):
    Real = 0  # real number
    Bool = 1  # boolean number
    Integar = 2  # integar number

    @property
    def numpy(self):
        return NP_DTYPE_MAP[self]
    
    @property
    def torch(self):
        return TORCH_DTYPE_MAP[self]


NP_DTYPE_MAP = {
    DType.Bool: np.bool_,
    DType.Real: np.float64,
    DType.Integar: np.int64
}

TORCH_DTYPE_MAP = {
    DType.Bool: torch.bool,
    DType.Real: torch.float,
    DType.Integar: torch.long
}
    

EPSILON = 1e-5
DECIMAL = 4 


class VType(abc.ABC):
    """the class describing a variable in the environment. How the
    buffers and neural networks deal with the data relative to this
    variable will be determined by this class.
    """
    
    @abc.abstractproperty
    def shape(self) -> Shape:
        """the shape of the **raw data** of the variable. Used by the
        environment and the buffer."""
        raise NotImplementedError

    @abc.abstractproperty
    def size(self) -> int:
        """the size of each sample of **input data** the variable. Used by
           the neural networks."""
        raise NotImplementedError
    
    @abc.abstractproperty
    def ptype(self) -> ptype.PType:
        """the posterior probability pattern. This is required since the
        neural network infers not deterministic values, but the distribution
        of the variables."""
        raise NotImplementedError

    def tensor(self, value, device):
        dtype = self.dtype.torch
        if not isinstance(value, torch.Tensor):
            tensor = torch.tensor(value, device=device, dtype=dtype)
        else:
            tensor = value.to(dtype=dtype, device=device)

        ndim = len(self.shape)
        if ndim != 0 and tensor.shape[-ndim:] != self.shape:
            raise ValueError("inconsistent shape")
    
        return tensor
    
    @abc.abstractmethod
    def space(self, n: int) -> spaces.Space:
        raise NotImplementedError
    
    @abc.abstractproperty
    def dtype(self) -> DType:
        """the data type of "raw data" and "label" of the variable. This is
        required by the buffer and network respectively."""
        raise NotImplementedError
    
    def _raw_batch_shape(self, batch: torch.Tensor):
        return batch.shape[0: batch.ndim - len(self.shape)]
    
    def _flatten_features(self, batch: torch.Tensor):
        return batch.view(*self._raw_batch_shape(batch), -1)

    @abc.abstractmethod
    def raw2input(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the raw data into the input of the network.

        Args:
            batch (torch.Tensor): the batched raw data like (*batch_shape, *shape)

        Returns:
            torch.Tensor: the encoded data like (*batch_shape, size_input)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def raw2label(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the raw data into the label of the network.

        Args:
            batch (torch.Tensor): the batched raw data like (*batch_shape, *shape)

        Returns:
            torch.Tensor: the encoded data like (*batch_shape, size_label)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def label2raw(self, batch: torch.Tensor) -> torch.Tensor:
        """encode the label inferred by networks into raw data like
        
        Args:
            torch.Tensor: the encoded data like (*batch_shape, size_label)

        Returns:
            batch (torch.Tensor): the batched raw data like (*batch_shape, *shape)
        """

        raise NotImplementedError


class ArrayBase(VType):
    """Base Class for Continuous Variables"""

    def __init__(self, shape: ShapeLike):
        super().__init__()
        self.__shape = Shaping.as_shape(shape)
        self.__size = Shaping.get_size(shape)
    
    @property
    def shape(self):
        return self.__shape

    @property
    def size(self) -> int:
        """the size of each sample of the variable as torch.Tensor."""
        return self.__size

    def raw2input(self, batch: torch.Tensor):
        return self._flatten_features(batch)
    
    def raw2label(self, batch: torch.Tensor):
        return self._flatten_features(batch)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.view(*batch.shape[:-1], *self.shape)


class Normal(ArrayBase):

    def __init__(self, shape: ShapeLike = (), 
                 low=None, high=None, scale: Optional[float] = None,
                 dtype: DType = DType.Real, transform=True):
        """_summary_

        Args:
            shape (ShapeLike, optional): the shape of the variable
            low (_type_, optional): the lower bound of the variable. None for (-inf).
            high (_type_, optional): the higher bound of the variable. None for (inf).
            scale (Optional[float], optional): the standard deviance of the distribution. None
                means the deviance is learned.
            dtype (DType, optional): the data type of the variable
            transform (bool, optional): whether to apply transforms on the predicted (mean, std)
                in order to better fit the bound [low, high]. It can help stablizing
                RL algorithms like PPO.
        """        

        super().__init__(shape)

        self.__l = None if low is None else torch.tensor(low, dtype=DType.Real.torch)
        self.__h = None if high is None else torch.tensor(high, dtype=DType.Real.torch)

        if transform:
            if low is None and high is None:  # unbounded
                self.__ptype = ptype.Normal(self.size, scale)
            elif low is None:  # (-inf, high]
                self.__ptype = ptype.SoftplusNormal(self.size, scale, start=high, negative=True)
            elif high is None:  # [low, +inf)
                self.__ptype = ptype.SoftplusNormal(self.size, scale, start=low, negative=False)
            else: # [low, high]
                self.__ptype = ptype.TanhNormal(self.size, scale, (low, high))
        else:
            self.__ptype = ptype.Normal(self.size, scale)
        
        self.__dtype = dtype
    
    @property
    def dtype(self):
        return self.__dtype

    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype
    
    def space(self, n: int) -> spaces.Space:
        shape = (n,) + self.shape
        low = np.array(-np.inf) if self.__l is None else self.__l.cpu().numpy()
        high = np.array(np.inf) if self.__h is None else self.__h.cpu().numpy()
        low = np.broadcast_to(low, shape)
        high = np.broadcast_to(high, shape)
        
        return spaces.Box(low, high, shape, self.__dtype.numpy)

    def __get_low_high(self, device: torch.device):
        if self.__l is not None:
            self.__l = self.__l.to(device)
        if self.__h is not None:
            self.__h = self.__h.to(device)
        return self.__l, self.__h
    
    def raw2input(self, batch: torch.Tensor):
        return self.__real(self._flatten_features(batch))
    
    def raw2label(self, batch: torch.Tensor):
        return self.__real(self._flatten_features(batch))
    
    def label2raw(self, batch: torch.Tensor):
        low, high = self.__get_low_high(batch.device)
        batch = batch.view(*batch.shape[:-1], *self.shape)
        if low is not None or high is not None:
            batch = torch.clamp(batch, min=low, max=high)
        batch = self.__convert_dtype(batch)
        return batch

    def __convert_dtype(self, batch: torch.Tensor):
        dtype = self.__dtype
        if dtype == DType.Integar:
            batch = batch.round()
        if dtype == DType.Bool:
            batch = (batch >= 0)
        batch = batch.to(dtype.torch)
        return batch
    
    def __real(self, batch: torch.Tensor):
        dtype = self.__dtype
        if dtype == DType.Bool:
            batch = torch.where(batch, 1., -1.)
        batch = batch.to(DType.Real.torch)
        return batch


class Categorical(VType):
    """Class for Categorical Variables"""

    def __init__(self, k: int):
        """
        Args:
            k (int): the number of categories
        """        

        super().__init__()
        self.__k = k
        self.__ptype = ptype.Categorical(k)
    
    @property
    def shape(self):
        return ()

    @property
    def size(self) -> int:
        return self.__k
    
    @property
    def dtype(self):
        return DType.Integar
    
    def space(self, n: int) -> spaces.Space:
        return spaces.MultiDiscrete([self.__k for _ in range(n)], DType.Integar.numpy)

    def raw2input(self, batch: torch.Tensor):
        return torch.nn.functional.one_hot(batch, self.__k).\
            to(DType.Real.torch)
    
    def raw2label(self, batch: torch.Tensor):
        return batch
    
    def label2raw(self, batch: torch.Tensor):
        return batch
    
    @property
    def ptype(self) -> ptype.PType:
        return self.__ptype


class Boolean(Normal):
    """Class for Categorical Variables"""

    def __init__(self, shape: ShapeLike = (), scale: Optional[float] = None):
        super().__init__(shape, scale, dtype=DType.Bool)
    
    def space(self, n: int) -> spaces.Space:
        return spaces.MultiBinary((n,) + self.shape)


class Binary(Categorical):
    """Class for Binary Variables"""

    def __init__(self):
        super().__init__(2)

    @property
    def shape(self):
        return ()
    
    @property
    def dtype(self):
        return DType.Bool
    
    def space(self, n: int) -> spaces.Space:
        return spaces.MultiBinary(n)
    
    def raw2input(self, batch: torch.Tensor):
        return super().raw2input(batch.to(DType.Integar.torch))
    
    def raw2label(self, batch: torch.Tensor):
        return batch.to(DType.Integar.torch)
    
    def label2raw(self, batch: torch.Tensor):
        return batch.bool()
