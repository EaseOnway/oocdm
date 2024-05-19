from scripts_ import *
import argparse
import utils
import absl.app as app

import configs

from typing import Type, Dict, Any, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()


SEED = args.seed

CONTINUE_N_ITER: Optional[int] = None  # for debugging only
CONTINUE = CONTINUE_N_ITER is not None

Train.seed(SEED)
SCRIPTS = {
    "gru": (TrainGRU, TestGRU),
    "cdl": (TrainCDL, TestCDL),
    "cdl-a": (TrainCDL, TestCDL),
    "mlp": (TrainMLP, TestMLP),
    "ooc": (TrainOOC, TestOOC),
    "full": (TrainFull, TestFull),
    "ticsa": (TrainTICSA, TestTICSA),
    "gnn": (TrainGNN, TestGNN),
}


RUN_ID = f"seed-{SEED}"


def get_trainer(approach: str, config: configs.Config) -> Train:
    if "ooc" == approach:
        train = TrainOOC(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.ooc_args,
            p_edge=config.p_edge,
            causal_threshold=config.cmi_thres,
            n_iter_stable=config.ooc_n_iter_stable,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif "full" == approach:
        train = TrainFull(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.ooc_args,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif "mlp" == approach:
        train = TrainMLP(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.mlp_args,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif approach == "cdl":
        train = TrainCDL(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.cdl_args("max"),
            causal_threshold=config.cmi_thres,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif approach == "cdl-a":
        train = TrainCDL(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.cdl_args("attn"),
            causal_threshold=config.cmi_thres,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif "gru" == approach:
        train = TrainGRU(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.gru_args,
            causal_threshold=config.fcit_thres,
            n_job_fcit=config.njob_fcit,
            max_size_fcit=config.fcit_size,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif "ticsa" == approach:
        train = TrainTICSA(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.tisca_args,
            norm_penalty=1.0,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    elif "gnn" == approach:
        train = TrainGNN(
            config.env_id,
            config.suffix,
            RUN_ID,
            config.trainargs,
            config.gnn_args,
            env_options=config.env_option,
            _continue=CONTINUE,
        )
    else:
        assert False
    return train


def run_test(
    approach: str,
    path: str,
    config: configs.Config,
    label="test",
    collector="random",
    env_options={},
    **collector_args,
):
    test_type: Type[Test] = SCRIPTS[approach][1]
    test_type(
        path, config.testargs, env_options, collector, label, **collector_args  # type: ignore
    ).main()


def experiment_1(_):
    # ----------< Selct the environment here! >----------- #
    CONFIG = configs.config_block(2)
    # CONFIG = configs.config_block(5)
    # CONFIG = configs.config_block(10)
    # CONFIG = configs.config_asymblock(2)
    # CONFIG = configs.config_asymblock(5)
    # CONFIG = configs.config_asymblock(10)
    # CONFIG = configs.config_mouse('444')
    # CONFIG = configs.CONFIG_CMS
    # CONFIG = configs.CONFIG_DZB
    # CONFIG = configs.config_walker(True)
    # CONFIG = configs.config_block(5, noised=True)
    # ----------< Selct the environment here! >----------- #
    
    # CONFIG.env_option['render'] = True

    if CONTINUE:
        assert CONTINUE_N_ITER is not None
        CONFIG.trainargs.n_iter = CONTINUE_N_ITER

    # ----- list the concerned approaches here -----
    for approach in [
        "ooc",  # OOCDM
        "full",  # OOFULL
        "cdl",  # CDL
        "cdl-a",  # CDL + attention
        "mlp",  # MLP
        "ticsa",  # TICSA
        "gnn",  # GNN
        "gru",  # FCIT+GRU
    ]:
        trainer = get_trainer(approach, CONFIG)
        trainer.main()  # annotate this line if you only want to evaluate the already-trained models.
        path = str(trainer.path)
        del trainer

        test_envoptions = {}
    
        if CONFIG.env_id == "walker":
            run_test(
                approach,
                path,
                CONFIG,
                "test",
                collector="ppo",
                collector_update_size=2048,
                collector_update_interval=20000,
            )
        else:
            run_test(approach, path, CONFIG, "test",
                     env_options=test_envoptions)

        if CONFIG.env_id in ("block", "mouse", "asymblock"):
            test_envoptions['ood'] = True
            run_test(
                approach,
                path,
                CONFIG,
                "test-new",
                env_options=test_envoptions,
            )
        elif CONFIG.env_id == "walker":
            run_test(
                approach,
                path,
                CONFIG,
                "test-new",
                collector="ppo",
                collector_update_size=2048 * 4,
                collector_update_interval=10000,
                env_options=test_envoptions
            )


def experiment_2(_):
    for approach in [
        "ooc",
        "full",
    ]:
        CONFIG = configs.config_mouse("seen")
        if CONTINUE:
            assert CONTINUE_N_ITER is not None
            CONFIG.trainargs.n_iter = CONTINUE_N_ITER

        trainer = get_trainer(approach, CONFIG)
        trainer.main()
        path = str(trainer.path)
        del trainer
        run_test(approach, path, CONFIG, "test")
        run_test(approach, path, CONFIG, "test-unseen", env_options=dict(task="unseen"))

        CONFIG.env_option["task"] = "unseen"
        trainer = get_trainer(approach, CONFIG)
        trainer.main()
        path = str(trainer.path)
        del trainer
        run_test(approach, path, CONFIG, "test")


if __name__ == "__main__":
    # ------ Select the experiment here ------
    app.run(experiment_1, ["_"])
    # app.run(experiment_2, ['_'])
