from math import ceil, log2

from numba import int32
from numba import types as nb_types
from numba.experimental import jitclass

spec = [
    ('int_min', int32),
    ('int_max', int32),
    ('shift', int32),
    ('symmetric', nb_types.boolean),
    ('_depth', int32),
]


@jitclass(spec=spec)  # type: ignore
class NBFixedPrecision:
    def __init__(
        self,
        int_min: int,
        int_max: int,
        shift: int,
        symmetric: bool = False,
        _depth: int = 0,
    ):
        self.int_min = int_min
        self.int_max = int_max
        self.shift = shift
        self.symmetric = symmetric
        self._depth = _depth

        if self.int_min > self.int_max:
            raise ValueError('int_min must be less than or equal to int_max')

    @property
    def k(self) -> int:
        return int(self.min < 0)

    @property
    def b(self) -> int:
        return ceil(log2(max(self.int_max + 1, -self.int_min)))

    @property
    def i(self) -> int:
        return self.b - self.shift

    @property
    def min(self) -> float:
        return self.int_min * 2.0 ** (-self.shift)

    @property
    def max(self) -> float:
        return self.int_max * 2.0 ** (-self.shift)

    def __str__(self) -> str:
        s = '' if self.k else 'u'
        p = f'ap_{s}fixed({self.b+self.k}, {self.i+self.k})'
        if self.int_min == self.int_max:
            return f'{p}({self.min})'
        return p

    def __add__(self, other: 'NBFixedPrecision'):
        shift = max(self.shift, other.shift)
        _shift0, _shift1 = shift - self.shift, shift - other.shift
        int_min = (self.int_min << _shift0) + (other.int_min << _shift1)
        int_max = (self.int_max << _shift0) + (other.int_max << _shift1)

        return NBFixedPrecision(
            int_min,
            int_max,
            shift,
            False,
            max(self._depth, other._depth) + 1,
        )

    def __neg__(self):
        return NBFixedPrecision(
            -self.int_max,
            -self.int_min,
            self.shift,
            False,
            self._depth,
        )

    def __sub__(self, other: 'NBFixedPrecision'):
        return self + (-other)

    def __lshift__(self, other: int):
        return NBFixedPrecision(
            self.int_min,
            self.int_max,
            self.shift - other,
            False,
            self._depth,
        )

    def __rshift__(self, other: int):
        return self << -other
