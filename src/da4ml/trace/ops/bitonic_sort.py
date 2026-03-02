import typing
from math import log2
from typing import overload
from warnings import warn

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable import FixedVariable
    from ..fixed_variable_array import FixedVariableArray


def _sort_2(ka: 'FixedVariable', kb: 'FixedVariable', vavb: 'tuple[NDArray, NDArray] | None', ascending: bool):
    diff = ka <= kb
    low = diff.msb_mux(ka, kb, zt_sensitive=False)
    high = diff.msb_mux(kb, ka, zt_sensitive=False)

    if vavb is not None:
        va, vb = vavb
        value_low = [diff.msb_mux(va[i], vb[i]) for i in range(len(va))]
        value_high = [diff.msb_mux(vb[i], va[i]) for i in range(len(va))]
    else:
        value_low = value_high = None
    if not ascending:
        return high, low, value_high, value_low
    return low, high, value_low, value_high


def _bitonic_merge(key: 'NDArray', value: 'list[NDArray] | None', ascending: bool):
    if len(key) <= 1:
        return
    for i in range(len(key) // 2):
        ka, kb = key[i], key[i + len(key) // 2]
        if value is not None:
            vavb = value[i], value[i + len(key) // 2]
        else:
            vavb = None
        kl, kh, vl, vh = _sort_2(ka, kb, vavb, ascending)
        key[i], key[i + len(key) // 2] = kl, kh
        if value is not None:
            value[i], value[i + len(key) // 2] = vl, vh  # type: ignore

    if value is not None:
        _bitonic_merge(key[: len(key) // 2], value[: len(value) // 2], ascending)
        _bitonic_merge(key[len(key) // 2 :], value[len(value) // 2 :], ascending)
    else:
        _bitonic_merge(key[: len(key) // 2], None, ascending)
        _bitonic_merge(key[len(key) // 2 :], None, ascending)


def _bitonic_sort(key: 'NDArray', value: 'list[NDArray] | None', ascending: bool):
    if len(key) <= 1:
        return
    if value is not None:
        _bitonic_sort(key[: len(key) // 2], value[: len(value) // 2], True)
        _bitonic_sort(key[len(key) // 2 :], value[len(key) // 2 :], False)
    else:
        _bitonic_sort(key[: len(key) // 2], None, True)
        _bitonic_sort(key[len(key) // 2 :], None, False)
    _bitonic_merge(key, value, ascending)


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


def sort(
    a: 'FixedVariableArray | np.ndarray',
    axis: int | None = None,
    kind: str = 'bitonic',
    aux_value: 'FixedVariableArray | None' = None,
):
    from ..fixed_variable_array import FixedVariableArray

    if isinstance(a, np.ndarray):
        return np.sort(a, axis=axis)
    if axis is None:
        axis = -1

    if aux_value is not None:
        assert a.ndim == 1, f'When using aux_value, only 1D index is supported. Got a.ndim={a.ndim}'
        assert a.shape[0] == aux_value.shape[0], (
            f'Length of the arrays must match. Got a.shape={a.shape}, aux_value.shape={aux_value.shape}'
        )

    if kind != 'bitonic':
        warn(f'Only bitonic sort is supported for now, discarding kind={kind}', stacklevel=2)
    if aux_value is not None:
        assert a.ndim == 1 and a.shape[0] == aux_value.shape[0], (
            f'When using aux_value, only 1D index is supported and the length of the arrays must match. '
            f'Got a.shape={a.shape}, aux_value.shape={aux_value.shape}'
        )
    sort_dim = a.shape[axis]
    assert log2(sort_dim).is_integer(), f'Bitonic sort requires the sorting dimension to be a power of 2. Got {sort_dim}'
    r = np.moveaxis(a._vars, axis, -1).copy()
    _shape = r.shape
    r = r.reshape(-1, sort_dim)
    if aux_value is not None:
        _aux_r = aux_value._vars.copy()
        _aux_shape = _aux_r.shape
        _aux_r = _aux_r.reshape(1, sort_dim, -1)
    else:
        _aux_r = None
    for i in range(len(r)):
        _bitonic_sort(r[i], _aux_r[i] if _aux_r is not None else None, ascending=True)
    r = np.array(r).reshape(_shape)
    r = np.moveaxis(r, -1, axis)
    if _aux_r is not None:
        _aux_r = _aux_r.reshape(_aux_shape)  # type: ignore
        # _aux_r = np.moveaxis(_aux_r, -1, 0)
        return FixedVariableArray(r, hwconf=a.hwconf), FixedVariableArray(_aux_r, hwconf=a.hwconf)
    return FixedVariableArray(r, hwconf=a.hwconf)
