from functools import cached_property
from uuid import uuid4

from .._binary import get_lsb_loc
from ..types import QInterval


class AtomicInterval:
    def __init__(self, qint: QInterval, uuid=None):
        self.qint = qint
        self.uuid = uuid or uuid4()

    def __hash__(self):
        return hash(self.uuid)

    def __repr__(self):
        return str(self.qint)


class AffineInterval:
    def __init__(self, coeffs: dict[AtomicInterval, float], bias: float):
        self.coeffs = coeffs
        self.bias = bias
        self._cached: QInterval | None = None

    @classmethod
    def new(cls, qint: QInterval) -> 'AffineInterval':
        if qint.min == qint.max:
            return cls({}, qint.min)
        return cls({AtomicInterval(qint): 1.0}, 0.0)

    @cached_property
    def qint(self) -> QInterval:
        min_val = self.bias
        max_val = self.bias

        if not self.coeffs:
            # Pure constant
            step = 2.0 ** get_lsb_loc(min_val)
            return QInterval(min_val, max_val, step)

        step = 2.0 ** get_lsb_loc(self.bias)
        for atom, coeff in self.coeffs.items():
            if coeff == 0:
                continue
            if coeff > 0:
                min_val += coeff * atom.qint.min
                max_val += coeff * atom.qint.max
            else:
                min_val += coeff * atom.qint.max
                max_val += coeff * atom.qint.min
            step = min(step, 2.0 ** get_lsb_loc(coeff * atom.qint.step))

        if min_val == max_val:
            step = 2.0 ** get_lsb_loc(min_val)

        return QInterval(min_val, max_val, step)

    def __add__(self, other):
        if not isinstance(other, AffineInterval):
            if other == 0:
                return self
            return AffineInterval(self.coeffs.copy(), self.bias + other)
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0.0) + v
        return AffineInterval(new_coeffs, self.bias + other.bias)

    def __neg__(self):
        new_coeffs = {k: -v for k, v in self.coeffs.items()}
        return AffineInterval(new_coeffs, -self.bias)

    def __mul__(self, other):
        if isinstance(other, AffineInterval):
            raise TypeError('AffineInterval * AffineInterval is non-linear; use AffineInterval.new()')
        if other == 0:
            return AffineInterval({}, 0.0)
        new_coeffs = {k: v * other for k, v in self.coeffs.items()}
        return AffineInterval(new_coeffs, self.bias * other)

    def __lshift__(self, other: int):
        if not isinstance(other, int):
            raise TypeError('Shift amount must be an integer')
        return self * (2**other)

    def __rshift__(self, other: int):
        return self.__lshift__(-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        terms = [f'{v}*{k}' for k, v in self.coeffs.items()]
        if self.bias:
            terms.append(str(self.bias))
        return f'AA({" + ".join(terms) if terms else "0"})'
