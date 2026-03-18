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
    a, b = zip(*[(k.msb_mux(va, vb), k.msb_mux(vb, va)) for va, vb in zip(a, b)])

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


def batcher_odd_even_merge_sort(a: 'NDArray', ascending: bool):
    """##copy-paste from wikipedia https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort; apparently it works

    note: the input sequence is indexed from 0 to (n-1)
    for p = 1, 2, 4, 8, ... # as long as p < n
      for k = p, p/2, p/4, p/8, ... # as long as k >= 1
        for j = mod(k,p) to (n-1-k) with a step size of 2k
          for i = 0 to min(k-1, n-j-k-1) with a step size of 1
            if floor((i+j) / (p*2)) == floor((i+j+k) / (p*2))
              compare and sort elements (i+j) and (i+j+k)"""

    for _p in range(ceil(log2(a.shape[0]))):
        p = 2**_p
        for _k in range(_p, -1, -1):
            k = 2**_k
            for j in range(k % p, a.shape[0] - k, 2 * k):
                for i in range(min(k, a.shape[0] - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        a[i + j], a[i + j + k] = cmp_swap(a[i + j], a[i + j + k], ascending)


@overload
def sort(  # type: ignore
    a: 'np.ndarray',
    axis: int | None = None,
    kind: str = 'batcher',
    aux_value: None = None,
) -> 'FixedVariableArray': ...


@overload
def sort(
    a: 'np.ndarray',
    axis: int | None = None,
    kind: str = 'batcher',
    aux_value: 'FixedVariableArray' = ...,
) -> 'tuple[FixedVariableArray, FixedVariableArray]': ...


def sort(  # type: ignore
    a: 'np.ndarray',
    axis: int | None = None,
    kind: str = 'batcher',
    aux_value: 'FixedVariableArray | None' = None,
):
    from ..fixed_variable_array import FixedVariableArray

    if not isinstance(a, FixedVariableArray):
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
            aux_value = aux_value[..., None]  # type: ignore
        assert aux_value.ndim - a.ndim == 1 and aux_value.shape[:-1] == a.shape, (  # type: ignore
            f'aux_value must be of shape a.shape (+ (k,)). Got a.shape={a.shape}, aux_value.shape={aux_value.shape}'  # type: ignore
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
        kind = kind.lower()
        if kind == 'bitonic':
            _bitonic_sort(r[i], ascending=True)
        elif kind == 'batcher':
            batcher_odd_even_merge_sort(r[i], ascending=True)  # type: ignore
        else:
            raise ValueError(f'Unsupported sorting algorithm: {kind}')

    r = r[:, n_pad_low : r.shape[1] - n_pad_high, :].reshape(_shape)
    r = np.moveaxis(r, -2, axis)  # type: ignore
    if aux_value is not None:
        return r[..., 0], r[..., 1:]
    assert r.shape[-1] == 1  # type: ignore
    return r[..., 0]
