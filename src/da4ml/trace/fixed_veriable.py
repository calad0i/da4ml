from decimal import Decimal
from math import ceil, log2

from ..cmvm.core import cost_add
from ..cmvm.types import QInterval
from .tracer import _TraceManager


class FixedVariable:
    def __init__(
        self,
        low: float | Decimal,
        high: float | Decimal,
        step: float | Decimal,
        latency: float = 0.0,
        cost: float = 0.0,
        _from: tuple['FixedVariable', 'FixedVariable'] | None = None,
        _factor: float | Decimal = 1.0,
        _id: int | None = None,
    ):
        assert low <= high, f'low {low} must be less than high {high}'
        self.low = Decimal(low)
        self.high = Decimal(high)
        self.step = Decimal(step)
        self._from = _from
        self._factor = Decimal(_factor)
        self.latency = latency
        self.cost = cost

        mgr = _TraceManager()
        if _id is None:
            self._id = mgr.count
            mgr.track(self)
        else:
            self._id = _id

    @property
    def unscaled(self):
        return self * (1 / self._factor)

    @property
    def qint(self) -> QInterval:
        return QInterval(float(self.low), float(self.high), float(self.step))

    @property
    def kif(self) -> tuple[bool, int, int]:
        if self.step == 0:
            return False, 0, 0
        f = -int(log2(self.step))
        i = ceil(log2(max(-self.low, self.high + self.step)))
        k = self.low < 0
        return k, i, f

    @classmethod
    def from_const(cls, const: float | Decimal, latency: float = 0.0):
        const = float(const)
        if const == 0:
            return cls(0, 0, 0)
        _low, _high = -32, 32
        while _high - _low > 1:
            _mid = (_high + _low) // 2
            _value = const * (2.0**_mid)
            if _value == int(_value):
                _high = _mid
            else:
                _low = _mid
        return cls(const, const, 2.0**-_high, latency=latency)

    def __repr__(self) -> str:
        k, i, f = self.kif
        if self._factor == 1:
            return f'FixedVariable({k}, {i}, {f})'
        return f'({self._factor} FixedVariable({k}, {i}, {f}))'

    def __neg__(self):
        return FixedVariable(
            -self.high,
            -self.low,
            self.step,
            _from=self._from,
            _factor=-self._factor,
            latency=self.latency,
            cost=self.cost,
            _id=self._id,
        )

    def __add__(self, other: 'FixedVariable|float|Decimal'):
        if not isinstance(other, FixedVariable):
            return self + FixedVariable.from_const(other)

        if other.low == other.high == 0:  # +0
            return self

        f0, f1 = self._factor, other._factor
        if f0 < 0:
            if f1 > 0:
                return other + self
            else:
                return -((-self) + (-other))

        mgr = _TraceManager()
        adder_size = mgr.hwconf.adder_size
        carry_size = mgr.hwconf.carry_size
        latency_cutoff = mgr.hwconf.latency_cutoff

        int0 = QInterval(float(self.low), float(self.high), float(self.step))
        int1 = QInterval(float(other.low), float(other.high), float(other.step))
        dlat, cost = cost_add(int0, int1, 0, False, adder_size, carry_size)
        base_lat = max(self.latency, other.latency)
        latency = dlat + base_lat
        if latency_cutoff > 0 and ceil(latency / latency_cutoff) > ceil(base_lat / latency_cutoff):
            # Crossed the latency cutoff boundry
            assert (
                dlat <= latency_cutoff
            ), f'Latency of an atomic operation {dlat} is larger than the pipelining latency cutoff {latency_cutoff}'
            latency = ceil(base_lat / latency_cutoff) * latency_cutoff + dlat

        # For constants, assign the latency of the output
        # make sure it will be defined within the same pipeline stage as the computation needs it
        if other.low == other.high:
            mgr.update_latency(other._id, latency)
        if self.low == self.high:
            mgr.update_latency(self._id, latency)

        return FixedVariable(
            self.low + other.low,
            self.high + other.high,
            min(self.step, other.step),
            _from=(self, other),
            cost=cost,
            latency=latency,
            _factor=f0,
        )

    def __sub__(self, other: 'FixedVariable|float|Decimal'):
        return self + (-other)

    def __mul__(
        self,
        other: 'float|Decimal',
    ):
        if other == 0:
            return FixedVariable(0, 0, 0, _id=-1)

        assert log2(abs(other)) % 1 == 0, 'Only support pow2 multiplication'

        other = Decimal(other)

        low = min(self.low * other, self.high * other)
        high = max(self.low * other, self.high * other)
        step = abs(self.step * other)
        _factor = self._factor * other

        return FixedVariable(
            low,
            high,
            step,
            _from=self._from,
            _factor=_factor,
            cost=self.cost,
            latency=self.latency,
            _id=self._id,
        )

    def __radd__(self, other: 'float|Decimal|FixedVariable'):
        return self + other

    def __rsub__(self, other: 'float|Decimal|FixedVariable'):
        return (-self) + other

    def __rmul__(self, other: 'float|Decimal|FixedVariable'):
        return self * other

    def relu(self, i: int | None = None, f: int | None = None):
        step = max(Decimal(2) ** -f, self.step) if f is not None else self.step
        low = max(Decimal(0), self.low)
        high = max(Decimal(0), self.high)
        if i is not None:
            high = min(high, Decimal(2) ** i - step)
        _factor = self._factor
        _from = (self, DummyVariable(-2, _factor))
        # sub flag depends on sign(_factor1)
        latency = self.latency
        cost = self.cost
        return FixedVariable(
            low,
            high,
            step,
            _from=_from,
            _factor=abs(_factor),
            latency=latency,
            cost=cost,
        )


class DummyVariable(FixedVariable):
    def __init__(self, _id: int, _factor: float | Decimal):
        super().__init__(0, 0, 0, _id=_id, _factor=_factor)
        self._from = None
