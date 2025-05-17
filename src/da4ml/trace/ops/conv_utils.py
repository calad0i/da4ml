from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ..fixed_variable_array import FixedVariableArray


def r_im2col(kernel_size: Sequence[int], arr: np.ndarray, buffer: np.ndarray, axis: int):
    w = kernel_size[0]
    if len(kernel_size) == 3:  # 1D
        for i in range(arr.shape[axis] - w + 1):
            patch = np.take(arr, range(i, i + w), axis=axis)
            buffer[i] = patch.flatten()
    else:  # 2D+
        for i in range(arr.shape[axis] - w + 1):
            patch = arr[i : i + w]
            r_im2col(kernel_size[1:], patch, buffer[i], axis + 1)


def _im2col(kernel_size: Sequence[int], arr: np.ndarray):
    if len(kernel_size) < 3:
        return arr
    shape = [inp_d - ker_d + 1 for inp_d, ker_d in zip(arr.shape, kernel_size[:-2])]
    shape.append(np.prod(kernel_size[:-1]))  # type: ignore
    buf = np.empty(shape, dtype=arr.dtype)
    r_im2col(kernel_size, arr, buf, 0)
    return buf


def stride_arr(stride: int | tuple[int, ...], arr: np.ndarray):
    ndim = arr.ndim
    if isinstance(stride, int):
        stride = (stride,) * (ndim - 1)
    assert len(stride) == ndim - 1, f'Invalid stride {stride} for array with {ndim} dimensions'

    _idx = tuple(slice(None, None, st) for st in stride)
    return arr[*_idx]


T = TypeVar('T', FixedVariableArray, NDArray[np.integer | np.floating])


def conv(
    x: T,
    kernel: NDArray[np.integer | np.floating],
    bias: NDArray[np.integer | np.floating] | None = None,
    strides: int | tuple[int, ...] = 1,
    padding: tuple[tuple[int, int], ...] | str = 'VALID',
    format: str = 'channels_last',
):
    if isinstance(x, FixedVariableArray):
        solver_options = x.solver_options
        data = x._vars
        is_symbolic = True
    else:
        solver_options = None
        data = x
        is_symbolic = False

    ndim = data.ndim
    ch_in, ch_out = kernel.shape[-2:]
    _ch_in = data.shape[-1]
    assert ch_in == _ch_in, f'Invalid input shape {data.shape} for kernel {kernel.shape}'
    assert kernel.ndim == ndim + 1

    assert format in ('channels_last', 'channels_first'), f'Invalid format {format}'

    if isinstance(strides, int):
        strides = (strides,) * (ndim - 1)
    assert len(strides) == ndim - 1, f'Invalid stride {strides} for array with {ndim} dimensions'

    if isinstance(padding, str):
        padding = padding.upper()
        if padding == 'VALID':
            padding = ((0, 0),) * (ndim - 1)
        elif padding == 'SAME':
            _padding = []
            for i in range(ndim - 1):
                pad0 = kernel.shape[i] // 2
                pad1 = kernel.shape[i] - pad0 - 1
                _padding.append((pad1, pad0))
            padding = tuple(_padding)
        else:
            raise ValueError(f'Invalid padding {padding}')
    assert len(padding) == ndim - 1, f'Invalid padding {padding} for array with {ndim} dimensions'
    assert all(len(p) == 2 for p in padding), f'Invalid padding {padding} for array with {ndim} dimensions'

    data = np.pad(data, padding + ((0, 0),), mode='constant', constant_values=0.0)
    data = _im2col(kernel.shape, data)
    if is_symbolic:
        _data = FixedVariableArray(data, solver_options) @ kernel.reshape(-1, ch_out)
        data = _data._vars
    else:
        data = data @ kernel.reshape(-1, ch_out)
    data = stride_arr(strides, data)
    if bias is not None:
        data = data + bias
    if format == 'channels_first':
        data = np.moveaxis(data, -1, 1)
    if solver_options is not None:
        return FixedVariableArray(data, solver_options)
    return data
