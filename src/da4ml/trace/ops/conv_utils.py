from collections.abc import Sequence
from math import prod
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


def _conv(
    x: T,
    kernel: NDArray[np.integer | np.floating],
    bias: NDArray[np.integer | np.floating] | None = None,
    strides: int | tuple[int, ...] = 1,
    padding: tuple[tuple[int, int], ...] | str = 'VALID',
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
    if kernel.ndim != ndim + 1:
        if kernel.ndim == ndim:
            raise ValueError('Inputs should not contain batch dimension')
        raise ValueError(f'Invalid kernel shape {kernel.shape} for input with {ndim} dimensions')
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
    if solver_options is not None:
        return FixedVariableArray(data, solver_options)
    return data


def conv(
    x: T,
    kernel: NDArray[np.integer | np.floating],
    bias: NDArray[np.integer | np.floating] | None = None,
    strides: int | tuple[int, ...] = 1,
    padding: tuple[tuple[int, int], ...] | str = 'VALID',
    format: str = 'channels_last',
    groups: int | None = None,
) -> T:
    assert format in ('channels_last', 'channels_first'), f'Invalid format {format}'
    if format == 'channels_first':
        if isinstance(x, FixedVariableArray):
            x = FixedVariableArray(np.moveaxis(x._vars, 0, -1), x.solver_options)  # type: ignore
        else:
            x = np.moveaxis(x, 0, -1)

    *_, _ch_in, ch_out = kernel.shape
    ch_in = x.shape[-1]
    assert ch_in % _ch_in == 0, f'groups is not integer (total_ch_in={ch_in}, kernel_ch_in={_ch_in})'
    if groups is None:
        groups = ch_in // _ch_in
    else:
        assert (
            groups == ch_in // _ch_in
        ), f'groups {groups} does not match input channels {ch_in} and kernel input channels {_ch_in}'
    assert ch_out % groups == 0, f'groups is not integer (total_ch_out={ch_out}, groups={groups})'
    _ch_out = ch_out // groups

    buf: list[T] = []
    for gp in range(groups):
        _kernel = kernel[..., gp * _ch_out : (gp + 1) * _ch_out]
        _x = x[..., gp * _ch_in : (gp + 1) * _ch_in]
        _buf = _conv(
            _x,
            _kernel,
            strides=strides,
            padding=padding,
        )
        buf.append(_buf)  # type: ignore

    if isinstance(x, FixedVariableArray):
        data = np.concatenate([b._vars for b in buf], axis=-1)  # type: ignore
    else:
        data = np.concatenate(buf, axis=-1)  # type: ignore

    data = data + bias if bias is not None else data

    if format == 'channels_first':
        if isinstance(x, FixedVariableArray):
            return FixedVariableArray(np.moveaxis(data, -1, 0), x.solver_options)
        return np.moveaxis(data, -1, 0)

    if isinstance(x, FixedVariableArray):
        return FixedVariableArray(data, x.solver_options)
    return data


def pool(
    x: T,
    pool_size: Sequence[int],
    strides: int | Sequence[int] | None = None,
    padding: tuple[tuple[int, int], ...] | str = 'VALID',
    pool_type: str = 'avg',
) -> T:
    if isinstance(x, FixedVariableArray):
        solver_options = x.solver_options
        data = x._vars
    else:
        solver_options = None
        data = x

    strides = strides or pool_size

    assert pool_type in ('avg', 'max'), f'Invalid pool type {pool_type}'
    ndim = data.ndim
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
                pad0 = pool_size[i] // 2
                pad1 = pool_size[i] - pad0 - 1
                _padding.append((pad1, pad0))
            padding = tuple(_padding)
        else:
            raise ValueError(f'Invalid padding {padding}')
    assert len(padding) == ndim - 1, f'Invalid padding {padding} for array with {ndim} dimensions'
    assert all(len(p) == 2 for p in padding), f'Invalid padding {padding} for array with {ndim} dimensions'

    data = np.pad(data, padding + ((0, 0),), mode='constant', constant_values=-np.inf)
    ch_in = data.shape[-1]
    fake_kernel_shape = tuple(pool_size) + (ch_in, ch_in)
    data = _im2col(fake_kernel_shape, data)
    data = data.reshape(*data.shape[:-1], ch_in, prod(pool_size))
    if pool_type == 'avg':
        div = np.sum(data != -np.inf, axis=-2)
        data = np.nan_to_num(data, neginf=0)
        data = np.sum(data, axis=-2) * (1 / div)
    else:
        raise NotImplementedError(f'Pool type {pool_type} is not implemented')

    data = stride_arr(tuple(strides), data)
    if isinstance(x, FixedVariableArray):
        return FixedVariableArray(data, solver_options)
    return data
