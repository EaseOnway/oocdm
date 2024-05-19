from typing import Final, Dict, Optional, Iterable

import torch
import torch.nn as nn

from core import VType, DType, EnvObjClass, EnvInfo
from utils.typings import NamedTensors, ObjectTensors
import utils


class AttributeEncoder(nn.Module):

    def __init__(self, vtype: VType,
                 dim_hidden: int, dim_out: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.vtype: Final = vtype
        self.dim_out: Final = dim_out

        # self.mean: torch.Tensor
        # self.std: torch.Tensor
        # self.register_buffer('mean', torch.zeros(vtype.size, device=device, dtype=dtype))
        # self.register_buffer('std', torch.ones(vtype.size, device=device, dtype=dtype))

        self.input_norm = nn.BatchNorm1d(vtype.size, device=device, dtype=dtype,
                                         momentum=norm_momentum)
        self.f = nn.Sequential(
            nn.Linear(vtype.size, dim_hidden, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_out, device=device, dtype=dtype),
            nn.ReLU()
            # nn.Linear(dim_hidden, dim_out, device=device, dtype=dtype),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        data = self.vtype.raw2input(raw)  # (n_sample, n_obj, feature)
        data: torch.Tensor = self.input_norm(data.transpose(1, 2))
        data = data.transpose(1, 2)
        return self.f(data)

    # def load_std_mean(self, std: torch.Tensor, mean: torch.Tensor):
    #     self.mean[:] = mean.to(self.mean.device, self.mean.dtype)
    #     self.std[:] = std.to(self.std.device, self.std.dtype)


class ClassAttributeEncoder(nn.Module):
    def __init__(self, c: EnvObjClass,
                 dim_hidden: int, dim_out: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.c: Final = c
        self.fieldnames: Final = c.fieldnames()

        self.encoders = {
            fieldname: AttributeEncoder(c.field_vtypes[fieldname],
                                        dim_hidden, dim_out, norm_momentum,
                                        device, dtype)
            for fieldname in c.fieldnames()
        }
        for fieldname, module in self.encoders.items():
            self.add_module('%s_encoder' % fieldname, module)

    def forward(self, raws: NamedTensors) -> torch.Tensor:
        """_summary_

        Args:
            raw_tensors (NamedTensors): A dictionary that consists of (attribute_name, attribute_data),
                where attribute_data is a tensor as (batch_size, n_object, *shape)
        Returns:
            torch.Tensor: A tensor as (batch_size, n_object, n_fieldibute, dim_out).
        """

        if len(raws) == 0:
            raise ValueError("The input dict inclues no attribute. "
                             f"Please check the definition of {self.c}")
        temp = [self.encoders[fieldname].forward(raws[fieldname])
                for fieldname in self.fieldnames]
        out = torch.stack(temp, dim=2)

        return out


class VariableEncoder(nn.Module):

    def __init__(self, envinfo: EnvInfo,
                 dim_hidden: int, dim_out: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.clsdict: Final = envinfo.__clsdict

        self.encoders = {
            cls_name: ClassAttributeEncoder(
                c, dim_hidden, dim_out, norm_momentum,
                device, dtype)
            for cls_name, c in self.clsdict.items()
        }
        for cls_name, module in self.encoders.items():
            self.add_module('%s_encoder' % cls_name, module)

    def forward(self, raw_attributes: ObjectTensors) -> NamedTensors:
        return {cls_name: self.encoders[cls_name].forward(x)
                for cls_name, x in raw_attributes.items()}
