from typing import Sequence
import numpy as np
import joblib
from .fcit import test


def _test(i: int, xs: Sequence[np.ndarray], y: np.ndarray):
    x = xs[i]
    z = np.concatenate([xs[j] for j in range(len(xs)) if j != i], axis=1)
    p = test(x, y, z)
    print('.', end='')
    return p


def test_rl(xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], n_jobs=1):
    edges = [(i, j) for i in range(len(xs)) for j in range(len(ys))]
    print("Performing FCIT tests: ", end='')
    p_values = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_test)(i, xs, ys[j]) for i, j in edges
    )
    print(' Done!')
    assert p_values is not None
    return {
        (i, j): p
        for (i, j), p in zip (edges, p_values)
    }
