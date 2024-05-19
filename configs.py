from scripts_ import *
import utils
import numpy as np

from typing import Type, Dict, Any, Optional, Literal


def clip(x, min_, max_):
    return int(np.clip(x, min_, max_))


class Config:
    def __init__(self,
                 env_id: str,
                 trainargs: Train.Args,
                 testargs: Test.Args,
                 ooc_args=TrainOOC.ModelArgs(),
                 mlp_args=TrainMLP.ModelArgs(),
                 cdl_args=TrainCDL.ModelArgs(),
                 gru_args=TrainGRU.ModelArgs(),
                 tisca_args = TrainTICSA.ModelArgs(),
                 gnn_args = TrainGNN.ModelArgs(),
                 p_edge=0.9,
                 cmi_thres: float = 0.1,
                 fcit_thres: float = 0.1,
                 fcit_size: Optional[int] = None,
                 njob_fcit: int = 4,
                 env_option: dict = {},
                 ooc_n_iter_stable: Optional[int] = None,
                 suffix='',
                 ):
        self.env_id = env_id
        self.env_option = env_option
        self.trainargs = trainargs
        self.cmi_thres = cmi_thres
        self.fcit_thres = fcit_thres
        self.fcit_size = fcit_size
        self.njob_fcit = njob_fcit
        self.testargs = testargs
        self.ooc_args = ooc_args
        self.mlp_args = mlp_args
        self.__cdl_args = cdl_args
        self.gru_args = gru_args
        self.gnn_args = gnn_args
        self.tisca_args = tisca_args
        self.p_edge = p_edge
        self.ooc_n_iter_stable = ooc_n_iter_stable
        self.suffix=suffix

    def cdl_args(self, kernel: Literal['max', 'attn']):
        self.__cdl_args.kernel = kernel
        return self.__cdl_args

def config_block(n_block: Optional[int] = None, noised=False, incomplete=False):
    n = n_block or 5
    suffix=''
    if noised:
        suffix += '-n'
    if incomplete:
        suffix += '-inc'
    sample_size = 10000

    return Config(
        'block',
        Train.Args(
            n_iter=5,
            buffer_size=sample_size,
            n_batch=100,
            n_batch_warmup=0,
            n_timestep=0,
            n_timestep_warmup=sample_size,
        ),
        Test.Args(
            n_timestep=0,
            eval_n_sample=100000,
            eval_batchsize=1000),
        env_option=dict(
            truncate_step=30,
            n_block=n_block,
            noised=noised,
            incomplete=incomplete,
        ),
        cmi_thres=0.3,
        mlp_args=TrainMLP.ModelArgs(
            dim_h1=clip(64 + 16*n, 64, 256),
            dim_h2=clip(64 + 16*n, 64, 256),
            dim_h3=clip(64 + 16*n, 64, 256),
        ),
        cdl_args=TrainCDL.ModelArgs(
            dim_variable_encoding=clip(32 + 16*n, 64, 256),
            attn_dim_k=clip(64 + 16*n, 32, 256),
        ),
        gru_args=TrainGRU.ModelArgs(
            dim_variable_encoding=clip(32 + 16*n, 64, 256),
        ),
        tisca_args=TrainTICSA.ModelArgs(
            dim_v=4, dim_h1=clip(24 + 4*n, 32, 64)
        ),
        gnn_args = TrainGNN.ModelArgs(
            dim_z=32, dim_h_z=32, dim_a=8, dim_h_a=8,
            dim_e=32, dim_h_edge=40, dim_h_node=32
        ),
        ooc_n_iter_stable = None,
        suffix=suffix
    )


def config_asymblock(n_block: Optional[int] = None, use_hidden_encoding=True):
    n = n_block or 5

    return Config(
        'asymblock',
        Train.Args(
            n_iter=75,
            buffer_size=10000,
            n_batch=1000,
            n_batch_warmup=0,
            n_timestep=0,
            n_timestep_warmup=10000,
        ),
        Test.Args(
            n_timestep=0,
            eval_n_sample=100000,
            eval_batchsize=1000),
        env_option=dict(
            truncate_step=30,
            n_block=n_block,
        ),
        cmi_thres=0.3,
        ooc_args=TrainOOC.ModelArgs(
            dim_id=64 if use_hidden_encoding else 0
        ),
        mlp_args=TrainMLP.ModelArgs(
            dim_h1=clip(64 + 16*n, 64, 256),
            dim_h2=clip(64 + 16*n, 64, 256),
            dim_h3=clip(64 + 16*n, 64, 256),
        ),
        cdl_args=TrainCDL.ModelArgs(
            dim_variable_encoding=clip(32 + 16*n, 64, 256),
            attn_dim_k=clip(64 + 16*n, 32, 256),
        ),
        gru_args=TrainGRU.ModelArgs(
            dim_variable_encoding=clip(32 + 16*n, 64, 256),
        ),
        tisca_args=TrainTICSA.ModelArgs(
            dim_v=4, dim_h1=clip(24 + 4*n, 32, 64)
        ),
        gnn_args = TrainGNN.ModelArgs(
            dim_z=32, dim_h_z=32, dim_a=8, dim_h_a=8,
            dim_e=32, dim_h_edge=40, dim_h_node=32
        ),
        ooc_n_iter_stable = None,
    )


def config_mouse(task: Optional[str] = None):
    if task is None:
        n = 10
    else:
        try:
            n = int(task[0]) + int(task[1]) + int(task[2])
        except ValueError:
            n = 10

    return Config(
        'mouse',
        Train.Args(
            n_iter=200,
            buffer_size=50000,
            n_batch=1000,
            n_batch_warmup=0,
            n_timestep=0,
            n_timestep_warmup=50000,
        ),
        Test.Args(
            n_timestep=3000,
            eval_n_sample=100000,
            eval_batchsize=1000),
        env_option=dict(
            truncate_step=300,
            task=task,
        ),
        mlp_args=TrainMLP.ModelArgs(
            dim_h1=clip(64 + 16*n, 64, 320),
            dim_h2=clip(64 + 16*n, 64, 320),
            dim_h3=clip(64 + 16*n, 64, 320),
        ),
        cdl_args=TrainCDL.ModelArgs(
            dim_variable_encoding=clip(64 + 16*n, 32, 256),
            attn_dim_k=clip(64 + 16*n, 32, 256),
        ),
        gru_args=TrainGRU.ModelArgs(
            dim_variable_encoding=clip(64 + 16*n, 32, 256),
        ),
        tisca_args=TrainTICSA.ModelArgs(dim_h1=32),
        gnn_args = TrainGNN.ModelArgs(
            dim_z=64, dim_h_z=64, dim_a=16, dim_h_a=16,
            dim_e=64, dim_h_edge=80, dim_h_node=64
        ),
        ooc_n_iter_stable = 15,
    )


CONFIG_CMS = Config(
    'cms',
    Train.Args(
        n_iter=80,
        buffer_size=100000,
        n_batch=1000,
        n_batch_warmup=0,
        n_timestep=1000,
        n_timestep_warmup=20000,
    ),
    Test.Args(
        n_timestep=2400,
        eval_n_sample=100000,
        eval_batchsize=1000),
    env_option=dict(
        truncate_step=200,
    ),
    cmi_thres=0.2,
    mlp_args=TrainMLP.ModelArgs(
        dim_h1=320,
        dim_h2=320,
        dim_h3=320,
    ),
    cdl_args=TrainCDL.ModelArgs(
        dim_variable_encoding=256,
        attn_dim_k=256,
    ),
    gru_args=TrainGRU.ModelArgs(
        dim_variable_encoding=256,
    ),
    tisca_args=TrainTICSA.ModelArgs(dim_h1=32),
    gnn_args = TrainGNN.ModelArgs(
        dim_z=32, dim_h_z=32, dim_a=8, dim_h_a=8,
        dim_e=32, dim_h_edge=40, dim_h_node=32
    ),
    ooc_n_iter_stable = 8,
)


CONFIG_DZB = Config(
    'dzb',
    Train.Args(
        n_iter=200,
        buffer_size=200000,
        n_batch=1000,
        n_batch_warmup=0,
        n_timestep=0,
        n_timestep_warmup=200000,
    ),
    Test.Args(
        n_timestep=2400,
        eval_n_sample=100000,
        eval_batchsize=1000),
    env_option=dict(
        truncate_step=200,
    ),
    cmi_thres=0.03,
    mlp_args=TrainMLP.ModelArgs(
        dim_h1=400,
        dim_h2=400,
        dim_h3=400,
    ),
    cdl_args=TrainCDL.ModelArgs(
        dim_variable_encoding=320,
        attn_dim_k=320,
    ),
    gru_args=TrainGRU.ModelArgs(
        dim_variable_encoding=320,
    ),
    tisca_args=TrainTICSA.ModelArgs(dim_h1=32),
    gnn_args = TrainGNN.ModelArgs(
        dim_z=64, dim_h_z=64, dim_a=16, dim_h_a=16,
        dim_e=64, dim_h_edge=80, dim_h_node=64
    ),
    ooc_n_iter_stable = 4,
)


def config_walker(use_hidden_encoding=True):
    return Config(
        'walker',
        Train.Args(
            n_iter=60,
            buffer_size=100000,
            n_batch=500,
            n_batch_warmup=0,
            n_timestep=0,
            n_timestep_warmup=100000,
            collector='ppo',
            collector_update_interval=20000,
            collector_update_size=2048,
        ),
        Test.Args(
            n_timestep=0,
            eval_n_sample=100000,
            eval_batchsize=1000,
        ),
        env_option=dict(check_healthy=True),
        cmi_thres=0.1,
        ooc_args=TrainOOC.ModelArgs(
            dim_id=(64 if use_hidden_encoding else 0),
            dim_local=32, dim_global=32, dim_k=32, dim_v=32),
        mlp_args=TrainMLP.ModelArgs(
            dim_h1=256,
            dim_h2=256,
            dim_h3=256,
        ),
        cdl_args=TrainCDL.ModelArgs(
            dim_variable_encoding=200,
            attn_dim_k=200,
        ),
        gru_args=TrainGRU.ModelArgs(
            dim_variable_encoding=200,
        ),
        tisca_args=TrainTICSA.ModelArgs(dim_h1=32),
        gnn_args = TrainGNN.ModelArgs(
            dim_z=128, dim_h_z=128, dim_a=32, dim_h_a=32,
            dim_e=128, dim_h_edge=128, dim_h_node=128
        ),
        fcit_size=10000,
        ooc_n_iter_stable = 8,
    )
