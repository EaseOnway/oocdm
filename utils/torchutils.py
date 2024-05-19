import torch
import tensorboard
from tensorboard.backend.event_processing import event_accumulator


def inv_softplus(x: torch.Tensor):
    return torch.log(torch.exp(x) - 1)


def read_tensorboard_logfile(path: str, key: str):
    """read the log files saved by tensorboard

    Args:
        path (str): path of the log files.
        key (str): key of the data

    Returns:
        list: values
        list: steps
        list: wall times
    """    

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    
    if key not in ea.scalars.Keys():
        values = []
        steps = []
        times = []

    items = ea.scalars.Items(key)
    values = [item.value for item in items]
    steps = [item.step for item in items]
    times = [item.wall_time for item in items]
    
    return values, steps, times
