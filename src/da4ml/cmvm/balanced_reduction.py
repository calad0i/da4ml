import numpy as np

from .fixed_variable import FixedVariable
from .scoring import py_scorer


def _balanced_reduction(vars: list[FixedVariable]):
    vars = vars.copy()

    n = len(vars)
    if n == 0:
        return FixedVariable.from_const(0.0)
    score_mat = np.full((len(vars), len(vars)), -np.inf, dtype=np.float32)
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            score_mat[i, j] = py_scorer(vars[i], vars[j]) - 1000 * (vars[i]._depth + vars[j]._depth)

    while n > 1:
        idx = np.argmax(score_mat)
        i, j = np.unravel_index(idx, score_mat.shape)
        vars[i] = vars[i] + vars[j]
        vars.pop(j)
        score_mat[j : n - 1] = score_mat[j + 1 : n]
        score_mat[:, j : n - 1] = score_mat[:, j + 1 : n]
        score_mat = score_mat[: n - 1, : n - 1]
        n -= 1
        for k in range(n):
            if k == i:
                continue
            if k < i:
                score_mat[k, i] = py_scorer(vars[k], vars[i]) - 1000 * (vars[k]._depth + vars[i]._depth)
            else:
                score_mat[i, k] = py_scorer(vars[i], vars[k]) - 1000 * (vars[i]._depth + vars[k]._depth)

    return vars[0]


def balanced_reduction(vars: list[FixedVariable], signed=True):
    if not signed:
        return _balanced_reduction(vars)
    pos_vars = [v for v in vars if v._factor > 0]
    neg_vars = [v for v in vars if v._factor < 0]
    for v in neg_vars:
        v._factor = -v._factor

    return _balanced_reduction(pos_vars) - _balanced_reduction(neg_vars)
