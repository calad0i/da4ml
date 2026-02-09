from math import log2
from typing import TypeVar, overload

from ...cmvm.types import QInterval
from ..fixed_variable import FixedVariable, _binary_bit_op, _unary_bit_op

T = TypeVar('T', FixedVariable, float)


@overload
def binary_bit_op(a: FixedVariable, b: FixedVariable, op: int, *args, **kwargs) -> FixedVariable: ...


@overload
def binary_bit_op(a: float, b: float, op: int, qint0: QInterval, qint1: QInterval) -> float: ...


def binary_bit_op(
    a: T, b: T, op: int, qint0: QInterval | None = None, qint1: QInterval | None = None, qint: QInterval | None = None
) -> T:
    _fn = {0: lambda x, y: x & y, 1: lambda x, y: x | y, 2: lambda x, y: x ^ y}[op]
    if isinstance(a, FixedVariable) or isinstance(b, FixedVariable):
        return _fn(a, b)
    assert isinstance(a, float) and isinstance(b, float), f'{type(a)} {type(b)}'
    assert qint0 is not None and qint1 is not None and qint is not None
    return _binary_bit_op(a, b, op, qint0, qint1, qint)


def unary_bit_op(a: T, op: int, qint_from: QInterval, qint_to: QInterval) -> T:
    if isinstance(a, FixedVariable):
        match op:
            case 0:
                return ~a << round(log2(qint_to.step / qint_from.step))
            case 1:
                return a.unary_bit_op('any')
            case 2:
                return a.unary_bit_op('all')
    assert isinstance(a, float)
    assert qint_to is not None and qint_from is not None
    return _unary_bit_op(a, op, qint_from, qint_to)
