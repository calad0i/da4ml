import typing
from collections.abc import Sequence
from math import ceil, log2
from typing import overload

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable import FixedVariable
    from ..fixed_variable_array import FixedVariableArray


def cmp_swap(a: 'Sequence[FixedVariable]|NDArray', b: 'Sequence[FixedVariable]|NDArray', ascending: bool):
    ka, kb = a[0], b[0]
    k = ka <= kb
    a, b = zip(*[(k.msb_mux(va, vb, zt_sensitive=False), k.msb_mux(vb, va, zt_sensitive=False)) for va, vb in zip(a, b)])

    if not ascending:
        return b, a
    return a, b


def _bitonic_merge(a: 'NDArray', ascending: bool):
    if len(a) <= 1:
        return
    for i in range(len(a) // 2):
        _a, _b = a[i], a[i + len(a) // 2]
        _a, _b = cmp_swap(_a, _b, ascending)
        a[i], a[i + len(a) // 2] = _a, _b

    _bitonic_merge(a[: len(a) // 2], ascending)
    _bitonic_merge(a[len(a) // 2 :], ascending)


def _bitonic_sort(a: 'NDArray', ascending: bool):
    if len(a) <= 1:
        return

    _bitonic_sort(a[: len(a) // 2], True)
    _bitonic_sort(a[len(a) // 2 :], False)
    _bitonic_merge(a, ascending)


def _pad_to_pow2(a: 'FixedVariableArray') -> 'tuple[FixedVariableArray, int, int]':
    assert a.ndim == 3
    from da4ml.trace import FixedVariable

    size = a.shape[-2]
    n_pad = 2 ** ceil(log2(size)) - size
    n_pad_low, n_pad_high = n_pad // 2, n_pad - n_pad // 2
    _low, _high, _ = a.lhs
    low_pad = FixedVariable.from_const(np.min(_low) - 1, hwconf=a.hwconf)
    high_pad = FixedVariable.from_const(np.max(_high) + 1, hwconf=a.hwconf)
    low_pad = np.full((a.shape[0], n_pad_low, a.shape[-1]), low_pad)
    high_pad = np.full((a.shape[0], n_pad_high, a.shape[-1]), high_pad)
    return np.concatenate([low_pad, a, high_pad], axis=-2), n_pad_low, n_pad_high  # type: ignore


@overload
def sort(
    a: np.ndarray,
    axis: int | None = None,
    kind: str = 'bitonic',
    aux_value: None = None,
) -> np.ndarray: ...


@overload
def sort(  # type: ignore
    a: FixedVariableArray,
    axis: int | None = None,
    kind: str = 'bitonic',
    aux_value: None = None,
) -> FixedVariableArray: ...


@overload
def sort(
    a: FixedVariableArray,
    axis: int | None = None,
    kind: str = 'bitonic',
    aux_value: FixedVariableArray = ...,
) -> tuple[FixedVariableArray, FixedVariableArray]: ...


def sort(  # type: ignore
    a: 'FixedVariableArray | np.ndarray',
    axis: int | None = None,
    kind: str = 'bitonic',
    aux_value: 'FixedVariableArray | None' = None,
):

    if isinstance(a, np.ndarray):
        return np.sort(a, axis=axis)
    if axis is None:
        axis = -1

    axis = axis % a.ndim

    if aux_value is not None:
        assert a.ndim == 1, f'When using aux_value, only 1D index is supported. Got a.ndim={a.ndim}'
        assert a.shape[0] == aux_value.shape[0], (
            f'Length of the arrays must match. Got a.shape={a.shape}, aux_value.shape={aux_value.shape}'
        )

    if aux_value is not None:
        if aux_value.shape == a.shape:
            aux_value = aux_value[..., None]
        assert aux_value.ndim - a.ndim == 1 and aux_value.shape[:-1] == a.shape, (
            f'aux_value must be of shape a.shape (+ (k,)). Got a.shape={a.shape}, aux_value.shape={aux_value.shape}'
        )
        a = np.concatenate([a[..., None], aux_value], axis=-1)  # type: ignore
    else:
        a = a[..., None]

    sort_dim = a.shape[axis]

    r = np.moveaxis(a, axis, -2).copy()  # type: ignore
    _shape = r.shape
    r = r.reshape(-1, sort_dim, r.shape[-1])

    r, n_pad_low, n_pad_high = _pad_to_pow2(r)  # type: ignore

    for i in range(len(r)):
        _bitonic_sort(r._vars[i], ascending=True)  # type: ignore

    r = r[:, n_pad_low : r.shape[1] - n_pad_high, :].reshape(_shape)
    r = np.moveaxis(r, -2, axis)  # type: ignore
    if aux_value is not None:
        return r[..., 0], r[..., 1:]
    assert r.shape[-1] == 1
    return r[..., 0]
