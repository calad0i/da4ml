from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..fixed_variable_array import FixedVariable
from .conv_utils import conv, pool
from .einsum_utils import einsum
from .reduce_utils import reduce

if TYPE_CHECKING:
    from ..fixed_variable_array import FixedVariableArray

T = TypeVar('T', 'FixedVariableArray', NDArray[np.floating], list[FixedVariable])


def relu(x: T, i: NDArray[np.integer] | None = None, f: NDArray[np.integer] | None = None, round_mode: str = 'TRN') -> T:
    from ..fixed_variable_array import FixedVariableArray

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
    k: NDArray[np.integer] | np.integer | int,
    i: NDArray[np.integer] | np.integer | int,
    f: NDArray[np.integer] | np.integer | int,
    overflow_mode: str = 'WRAP',
    round_mode: str = 'TRN',
) -> T:
    from ..fixed_variable_array import FixedVariableArray

    if isinstance(x, FixedVariableArray):
        return x.quantize(k=k, i=i, f=f, overflow_mode=overflow_mode, round_mode=round_mode)
    else:
        x = x.copy()
        if overflow_mode in ('SAT', 'SAT_SYM'):
            step = 2.0**-f
            _high = 2.0**i
            high = _high - step
            low = -_high * k if overflow_mode == 'SAT' else -high * k
            x = np.clip(x, low, high)  # type: ignore
        if round_mode.upper() == 'RND':
            x += 2.0 ** (-f - 1)  # type: ignore
        b = k + i + f
        bias = 2.0 ** (b - 1) * k
        eps = 2.0**-f
        return eps * ((np.floor(x / eps) + bias) % 2.0**b - bias)  # type: ignore


__all__ = [
    'conv',
    'einsum',
    'relu',
    'quantize',
    'pool',
    'reduce',
]
