import heapq
import typing
from collections.abc import Callable, Sequence
from math import prod
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable import FixedVariable
    from ..fixed_variable_array import FixedVariableArray


T = typing.TypeVar('T', 'FixedVariable', float, np.floating)
TA = TypeVar('TA', 'FixedVariableArray', NDArray[np.integer | np.floating])


class Packet:
    def __init__(self, v):
        self.value = v

    def __gt__(self, other: 'Packet') -> bool:  # type: ignore
        from ..fixed_variable_array import FixedVariable

        a, b = self.value, other.value

        if isinstance(a, FixedVariable):
            if isinstance(b, FixedVariable):
                if b.latency > a.latency:
                    return False
                if b.latency < a.latency:
                    return True
                if b._factor > 0 and a._factor < 0:
                    return False
                if b._factor < 0 and a._factor > 0:
                    return True
                return sum(a.kif[:2]) > sum(b.kif[:2])
            return True

        return False

    def __lt__(self, other: 'Packet') -> bool:  # type: ignore
        return not self.__gt__(other)


def _reduce(operator: Callable[[T, T], T], arr: Sequence[T]) -> T:
    from ..fixed_variable_array import FixedVariable

    if isinstance(arr, np.ndarray):
        arr = list(arr.ravel())
    assert len(arr) > 0, 'Array must not be empty'
    if len(arr) == 1:
        return arr[0]
    dtype = arr[0].__class__
    if not issubclass(dtype, FixedVariable):
        r = operator(arr[0], arr[1])
        for i in range(2, len(arr)):
            r = operator(r, arr[i])
        return r

    heap = [Packet(v) for v in arr]  # type: ignore
    heapq.heapify(heap)
    while len(heap) > 1:
        v1 = heapq.heappop(heap).value
        v2 = heapq.heappop(heap).value
        v = operator(v1, v2)
        heapq.heappush(heap, Packet(v))  # type: ignore
    return heap[0].value


def reduce(operator: Callable[[T, T], T], x: TA, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> TA:
    """
    Reduce the array by summing over the specified axis.
    """
    from ..fixed_variable_array import FixedVariableArray

    if isinstance(x, FixedVariableArray):
        solver_config = x.solver_options
        arr = x._vars
    else:
        solver_config = None
        arr = x
    all_axis = tuple(range(arr.ndim))
    axis = axis if axis is not None else all_axis
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    axis = tuple(a if a >= 0 else a + arr.ndim for a in axis)

    xpose_axis = sorted(all_axis, key=lambda a: (a in axis) * 1000 + a)
    if keepdims:
        target_shape = tuple(d if ax not in axis else 1 for ax, d in enumerate(arr.shape))
    else:
        target_shape = tuple(d for ax, d in enumerate(arr.shape) if ax not in axis)

    dim_contract = prod(arr.shape[a] for a in axis)
    arr = np.transpose(arr, xpose_axis)  # type: ignore
    _arr = arr.reshape(-1, dim_contract)
    _arr = np.array([_reduce(operator, _arr[i]) for i in range(_arr.shape[0])])
    r = _arr.reshape(target_shape)  # type: ignore

    if isinstance(x, FixedVariableArray):
        r = FixedVariableArray(r, solver_config)
        if r.shape == ():
            return r._vars.item()  # type: ignore
    return r if r.shape != () or keepdims else r.item()  # type: ignore
