from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ..fixed_variable_array import FixedVariable, FixedVariableArray
from .conv_utils import conv
from .einsum_utils import einsum

T = TypeVar('T', FixedVariableArray, NDArray[np.floating], list[FixedVariable])


def relu(x: T, i: NDArray[np.integer] | None = None, f: NDArray[np.integer] | None = None, round_mode: str = 'TRN') -> T:
    if isinstance(x, FixedVariableArray):
        return x.relu(i=i, f=f, round_mode=round_mode)
    elif isinstance(x, list):
        return [xx.relu(i=ii, f=ff, round_mode=round_mode) for xx, ii, ff in zip(x, i, f)]  # type: ignore
    else:
        x = np.maximum(x, 0)
        if f is not None:
            if round_mode.upper() == 'RND':
                x += 2.0 ** (-f - 1)
            sf = 2.0**f
            x = np.floor(x * sf) / sf
        if i is not None:
            x = x % 2.0**i
        return x


def quantize(
    x: T,
    k: NDArray[np.integer],
    i: NDArray[np.integer],
    f: NDArray[np.integer],
    overflow_mode: str = 'WRAP',
    round_mode: str = 'TRN',
) -> T:
    assert overflow_mode.upper() == 'WRAP', 'Only WRAP overflow mode is supported'
    if isinstance(x, FixedVariableArray):
        return x.quantize(k=k, i=i, f=f, overflow_mode=overflow_mode, round_mode=round_mode)
    else:
        if round_mode.upper() == 'RND':
            x += 2.0 ** (-f - 1)
        b = k + i + f
        bias = 2.0 ** (b - 1) * k
        eps = 2.0**-f
        return eps * ((np.floor(x / eps) + bias) % 2.0**b - bias)  # type: ignore


__all__ = [
    'conv',
    'einsum',
    'relu',
    'quantize',
]
