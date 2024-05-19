from typing import Type, Optional
from core import ObjectOrientedEnv
from alg.model import *

from envs.mouse import MouseEnv
from envs.block import BlockEnv
from envs.asymblock import AsymBlockEnv
from envs.sc2 import DefeatZerglingBaneling
from envs.sc2 import CollectMineralShards
from envs.walker import WalkerEnv

import torch
import numpy as np


_ENVS = {
    "mouse": MouseEnv,
    "block": BlockEnv,
    'asymblock': AsymBlockEnv,
    "dzb": DefeatZerglingBaneling,
    "cms": CollectMineralShards,
    'walker': WalkerEnv,
}


def get_env(envid: str) -> Type[ObjectOrientedEnv]:
    try:
        return _ENVS[envid]
    except KeyError:
        print("Supported environments are:")
        for envid_ in _ENVS.keys():
            print(f"- {envid_}")
        raise ValueError(f"Environment '{envid}' is not supported.")
