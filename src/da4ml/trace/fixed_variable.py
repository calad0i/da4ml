from decimal import Decimal
from math import ceil, floor, log2
from typing import NamedTuple
from uuid import UUID, uuid4

from ..cmvm.core import cost_add
from ..cmvm.types import QInterval


class HWConfig(NamedTuple):
    adder_size: int
    carry_size: int
    latency_cutoff: float


def _const_f(const: float | Decimal):
    const = float(const)
    _low, _high = -32, 32
    while _high - _low > 1:
        _mid = (_high + _low) // 2
        _value = const * (2.0**_mid)
        if _value == int(_value):
            _high = _mid
        else:
            _low = _mid
    return _high


class FixedVariable:
    def __init__(
        self,
        low: float | Decimal,
        high: float | Decimal,
        step: float | Decimal,
        latency: float | None = None,
        hwconf=HWConfig(-1, -1, -1),
        opr: str = 'new',
        cost: float | None = None,
        _from: tuple['FixedVariable', ...] = (),
        _factor: float | Decimal = 1.0,
        _data: Decimal | None = None,
        _id: UUID | None = None,
    ) -> None:
        assert low <= high, f'low {low} must be less than high {high}'

        if low == high:
            opr = 'const'
            _factor = 1.0
            _from = ()

        low, high, step = Decimal(low), Decimal(high), Decimal(step)
        low, high = floor(low / step) * step, ceil(high / step) * step
        self.low = low
        self.high = high
        self.step = step
        self._factor = Decimal(_factor)
        self._from: tuple[FixedVariable, ...] = _from
        opr = opr
        self.opr = opr
        self._data = _data
        self.id = _id or uuid4()
        self.hwconf = hwconf

        if opr == 'cadd':
            assert _data is not None, 'cadd must have data'

        if cost is None or latency is None:
            _cost, _latency = self.get_cost_and_latency()
        else:
            _cost, _latency = cost, latency

        self.latency = _latency
        self.cost = _cost

    def get_cost_and_latency(self):
        if self.opr == 'const':
            return 0.0, 0.0
        if self.opr in ('vadd', 'cadd'):
            adder_size = self.hwconf.adder_size
            carry_size = self.hwconf.carry_size
            latency_cutoff = self.hwconf.latency_cutoff

            if self.opr == 'vadd':
                assert len(self._from) == 2
                v0, v1 = self._from
                int0, int1 = v0.qint, v1.qint
                base_latency = max(v0.latency, v1.latency)
                dlat, _cost = cost_add(int0, int1, 0, False, adder_size, carry_size)
            else:
                assert len(self._from) == 1
                assert self._data is not None, 'cadd must have data'
                # int0 = self._from[0].qint
                # int1 = QInterval(float(self._data), float(self._data), float(self.step))
                _f = _const_f(self._data)
                _cost = float(ceil(log2(abs(self._data) + Decimal(2) ** -_f))) + _f
                base_latency = self._from[0].latency
                dlat = 0.0

            _latency = dlat + base_latency
            if latency_cutoff > 0 and ceil(_latency / latency_cutoff) > ceil(base_latency / latency_cutoff):
                # Crossed the latency cutoff boundry
                assert (
                    dlat <= latency_cutoff
                ), f'Latency of an atomic operation {dlat} is larger than the pipelining latency cutoff {latency_cutoff}'
                _latency = ceil(base_latency / latency_cutoff) * latency_cutoff + dlat
        elif self.opr in ('relu', 'wrap'):
            assert len(self._from) == 1
            _latency = self._from[0].latency
            _cost = 0.0
            # Assume LUT5 used here (2 fan-out per LUT6, thus *1/2)
            if self._from[0]._factor < 0:
                _cost += sum(self.kif) / 2
            if self.opr == 'relu':
                _cost += sum(self.kif) / 2

        elif self.opr == 'new':
            # new variable, no cost
            _latency = 0.0
            _cost = 0.0
        else:
            raise NotImplementedError(f'Operation {self.opr} is unknown')
        return _cost, _latency

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

    def __repr__(self) -> str:
        if self._factor == 1:
            return f'FixedVariable({self.low}, {self.high}, {self.step})'
        return f'({self._factor}) FixedVariable({self.low}, {self.high}, {self.step})'

    def __neg__(self):
        return FixedVariable(
            -self.high,
            -self.low,
            self.step,
            _from=self._from,
            _factor=-self._factor,
            latency=self.latency,
            cost=self.cost,
            opr=self.opr,
            _id=self.id,
            _data=self._data,
            hwconf=self.hwconf,
        )

    def __add__(self, other: 'FixedVariable|float|Decimal|int'):
        if not isinstance(other, FixedVariable):
            return self._const_add(other)
        if other.high == other.low:
            return self._const_add(other.low)
        if self.high == self.low:
            return other._const_add(self.low)

        assert self.hwconf == other.hwconf, 'FixedVariable must have the same hwconf'

        f0, f1 = self._factor, other._factor
        if f0 < 0:
            if f1 > 0:
                return other + self
            else:
                return -((-self) + (-other))

        return FixedVariable(
            self.low + other.low,
            self.high + other.high,
            min(self.step, other.step),
            _from=(self, other),
            _factor=f0,
            opr='vadd',
            hwconf=self.hwconf,
        )

    def _const_add(self, other: float | Decimal):
        if not isinstance(other, (int, float, Decimal)):
            other = float(other)  # direct numpy to decimal raises error
        other = Decimal(other)
        if other == 0:
            return self

        if self.opr != 'cadd':
            cstep = Decimal(2.0 ** -_const_f(other))

            return FixedVariable(
                self.low + other,
                self.high + other,
                min(self.step, cstep),
                _from=(self,),
                _factor=self._factor,
                _data=other / self._factor,
                opr='cadd',
                hwconf=self.hwconf,
            )

        # cadd, combine the constant
        assert len(self._from) == 1
        parent = self._from[0]
        assert self._data is not None, 'cadd must have data'
        sf = self._factor / parent._factor
        other1 = (self._data * parent._factor) + other / sf
        return (parent + other1) * sf

    def __sub__(self, other: 'FixedVariable|int|float|Decimal'):
        return self + (-other)

    def __mul__(
        self,
        other: 'float|Decimal',
    ):
        if other == 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf)

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
            opr=self.opr,
            latency=self.latency,
            cost=self.cost,
            _id=self.id,
            _data=self._data,
            hwconf=self.hwconf,
        )

    def __radd__(self, other: 'float|Decimal|int|FixedVariable'):
        return self + other

    def __rsub__(self, other: 'float|Decimal|int|FixedVariable'):
        return (-self) + other

    def __rmul__(self, other: 'float|Decimal|int|FixedVariable'):
        return self * other

    def relu(self, i: int | None = None, f: int | None = None, round_mode: str = 'TRN'):
        round_mode = round_mode.upper()
        assert round_mode in ('TRN', 'RND')

        if self.opr == 'const':
            val = self.low * (self.low > 0)
            f = _const_f(val) if not f else f
            step = Decimal(2) ** -f
            i = ceil(log2(val + step)) if not i else i
            eps = step / 2 if round_mode == 'RND' else 0
            val = (floor(val / step + eps) * step) % (Decimal(2) ** i)
            return FixedVariable(val, val, step, hwconf=self.hwconf)

        step = max(Decimal(2) ** -f, self.step) if f is not None else self.step
        if step > self.step and round_mode == 'RND':
            return (self + step / 2).relu(i, f, 'TRN')
        low = max(Decimal(0), self.low)
        high = max(Decimal(0), self.high)
        if i is not None:
            _high = Decimal(2) ** i - step
            if _high < high:
                # overflows
                low = Decimal(0)
                high = _high
        _factor = self._factor
        return FixedVariable(
            low,
            high,
            step,
            _from=(self,),
            _factor=abs(_factor),
            opr='relu',
            hwconf=self.hwconf,
            cost=sum(self.kif) * (1 if _factor > 0 else 2),
        )

    def quantize(
        self,
        k: int | bool,
        i: int,
        f: int,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        overflow_mode, round_mode = overflow_mode.upper(), round_mode.upper()
        assert overflow_mode in ('WRAP', 'SAT')
        assert round_mode in ('TRN', 'RND')

        _k, _i, _f = self.kif

        if k >= _k and i >= _i and f >= _f:
            return self

        if f < _f and round_mode == 'RND':
            return (self + 2.0 ** (-f - 1)).quantize(k, i, f, overflow_mode, 'TRN')

        if self.low == self.high:
            val = self.low
            step = Decimal(2) ** -f
            _high = Decimal(2) ** i
            high, low = _high - step, -_high * k
            val = (floor(val / step) * step - low) % (2 * _high) + low
            return FixedVariable(val, val, step, hwconf=self.hwconf)

        # TODO: corner cases exists (e.g., overflow to negative, or negative overflow to high value)
        # bit-exactness will be lost in these cases, but they should never happen (quantizers are used in a weird way)
        # Keeping this for now; change if absolutely necessary
        f = min(f, _f)
        k = min(k, _k) if i >= _i else k
        i = min(i, _i)

        step = max(Decimal(2) ** -f, self.step)

        low = -k * Decimal(2) ** i
        high = Decimal(2) ** i - step
        _low, _high = self.low, self.high

        if _low >= low and _high <= high:
            low, high = _low, _high

        if low > high:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf)

        return FixedVariable(
            low,
            high,
            step,
            _from=(self,),
            _factor=abs(self._factor),
            opr='wrap' if overflow_mode == 'WRAP' else 'sat',
            latency=self.latency,
            hwconf=self.hwconf,
        )

    @classmethod
    def from_kif(cls, k: int | bool, i: int, f: int, **kwargs):
        step = Decimal(2) ** -f
        _high = Decimal(2) ** i
        low, high = k * _high, _high - step
        return cls(low, high, step, **kwargs)
