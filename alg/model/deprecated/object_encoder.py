from typing import Final, Dict, Optional, Iterable

import torch
import torch.nn as nn

from core import VType, DType, EnvObjClass, EnvInfo
from utils.typings import NamedTensors, ObjectTensors


class ObjectEncoder(nn.Module):
    def __init__(self, envinfo: EnvInfo,
                 dim_variable_encoding: int, dim_hidden: int, dim_out: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.classes: Final = envinfo.__clsdict

        self.transforms = {
            clsname: nn.Sequential(
                nn.Linear(c.n_field() * dim_variable_encoding, dim_hidden,
                          device=device, dtype=dtype),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden, dim_out,
                          device=device, dtype=dtype),
                nn.LeakyReLU(),
            )
            for clsname, c in self.classes.items()
        }
        for clsname, module in self.transforms.items():
            self.add_module("%s_transform" % clsname, module)

    def __apply_mask(self, attr_encoding: torch.Tensor, attr_mask: torch.Tensor):
        """
        Args:
            attr_encoding (torch.Tensor): tensor as (batch_size, n_obj, n_field, dim_variable_encoding)
            attr_mask (torch.Tensor):  tensor as (batch_size, n_obj, n_field) or (n_obj, n_field) or (n_field)

         Returns:
            torch.Tensor: tensor as (batchsize, n_obj, n_field * dim_variable_encoding)
        """

        mask = attr_mask.unsqueeze(-1).logical_not()
        x = torch.masked_fill(attr_encoding, mask, 0.)
        x = torch.flatten(x, start_dim=2)

        return x

    def forward(self, attr_encodings: NamedTensors, attr_masks: NamedTensors):
        """
        Args:
            attr_encodings (NamedTensors): each tensor as (batch_size, n_obj, n_field, dim_variable_encoding)
            attr_mask (NamedTensors): each tensor that can be broadcasted to (batch_size, n_obj, n_field)
        Returns:
            NamedTensors: each tensor as (batchsize, n_obj, n_field * dim_variable_encoding)
        """
        out: NamedTensors = {}

        for clsname, x in attr_encodings.items():
            mask = attr_masks[clsname]
            # (batchsize, n_obj, n_field * dim_variable_encoding)
            x = self.__apply_mask(x, mask)
            x: torch.Tensor = self.transforms[clsname](x)  # (batchsize, n_obj, dim_out)
            out[clsname] = x

        return out

    def forward_one(self, clsname: str, attr_encoding: torch.Tensor, attr_mask: torch.Tensor):
        x = self.__apply_mask(attr_encoding, attr_mask)
        out: torch.Tensor = self.transforms[clsname](x)
        return out
    