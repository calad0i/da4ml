import random
import typing
from collections.abc import Callable, Generator
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from math import ceil, floor, log2
from typing import Any, NamedTuple, overload
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from ..cmvm.core import cost_add
from ..cmvm.types import QInterval, _minimal_kif
from ..cmvm.util.bit_decompose import _shift_centering

rd = random.Random()

if typing.TYPE_CHECKING:
    pass


class HWConfig(NamedTuple):
    adder_size: int
    carry_size: int
    latency_cutoff: float


ufunc_t = Callable[[NDArray[np.floating]], NDArray[np.floating]]


class TraceContext:
    _tables: 'dict[str, tuple[LookupTable, int]]' = {}
    hwconf: HWConfig = HWConfig(1, -1, -1)
    _table_counter = 0

    def register_table(self, table: 'LookupTable|np.ndarray'):
        if isinstance(table, np.ndarray):
            table = LookupTable(table)
        if table.spec.hash in self._tables:
            return self._tables[table.spec.hash]
        self._tables[table.spec.hash] = (table, self._table_counter)

        self._table_counter += 1
        return self._tables[table.spec.hash]

    def index_table(self, hash: str) -> int:
        return self._tables[hash][1]

    def get_table_from_index(self, index: int) -> 'LookupTable':
        for table, idx in self._tables.values():
            if idx == index:
                return table
        raise KeyError(f'No table found with index {index}')


table_context = TraceContext()


@dataclass
class TableSpec:
    hash: str
    out_qint: QInterval
    inp_width: int

    @property
    def out_kif(self) -> tuple[bool, int, int]:
        return _minimal_kif(self.out_qint)


def to_spec(table: NDArray[np.floating]) -> tuple[TableSpec, NDArray[np.int32]]:
    f_out = -_shift_centering(np.array(table))
    int_table = (table * 2**f_out).astype(np.int32)
    h = sha256(int_table.data)
    h.update(f'{f_out}'.encode())
    inp_width = ceil(log2(table.size))
    out_qint = QInterval(float(np.min(table)), float(np.max(table)), float(2**-f_out))
    return TableSpec(hash=h.hexdigest(), inp_width=inp_width, out_qint=out_qint), int_table


@overload
def interpret_as(
    x: NDArray[np.integer],
    k: int,
    i: int,
    f: int,
) -> NDArray[np.floating]: ...


@overload
def interpret_as(
    x: int,
    k: int,
    i: int,
    f: int,
) -> float: ...


def interpret_as(
    x: Any,
    k: int,
    i: int,
    f: int,
) -> Any:
    b = k + i + f
    bias = 2.0 ** (b - 1) * k
    eps = 2.0**-f
    floor_fn = np.floor if isinstance(x, np.ndarray) else floor
    return eps * (floor_fn(x + bias) % 2.0**b - bias)


class LookupTable:
    def __init__(self, values: NDArray, spec: TableSpec | None = None):
        assert values.ndim == 1, 'Lookup table values must be 1-dimensional'
        if spec is not None:
            assert values.dtype == np.int32, f'{values.dtype}'
            self.spec = spec
            self.table = values
        else:
            self.spec, self.table = to_spec(values)

    @overload
    def lookup(self, var: 'FixedVariable', qint_in: QInterval) -> 'FixedVariable': ...

    @overload
    def lookup(self, var: np.floating | float, qint_in: QInterval | tuple[float, float, float]) -> float: ...

    def lookup(self, var, qint_in: QInterval | tuple[float, float, float]):
        if isinstance(var, FixedVariable):
            return var.lookup(self)
        else:
            _min, _max, _step = qint_in
            assert _min <= var <= _max, f'Value {var} out of range [{_min}, {_max}]'
            index = round((var - _min) / _step)
            return interpret_as(int(self.table[index]), *self.spec.out_kif)

    @property
    def float_table(self) -> NDArray[np.floating]:
        k, i, f = self.spec.out_kif
        return interpret_as(self.table, k, i, f)  # type: ignore

    def to_dict(self) -> dict:
        return {
            'spec': {
                'hash': self.spec.hash,
                'out_qint': {
                    'min': self.spec.out_qint.min,
                    'max': self.spec.out_qint.max,
                    'step': self.spec.out_qint.step,
                },
                'inp_width': self.spec.inp_width,
            },
            'table': self.table.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LookupTable':
        spec_data = data['spec']
        out_qint_data = spec_data['out_qint']
        spec = TableSpec(
            hash=spec_data['hash'],
            out_qint=QInterval(out_qint_data['min'], out_qint_data['max'], out_qint_data['step']),
            inp_width=spec_data['inp_width'],
        )
        table = np.array(data['table'], dtype=np.int32)
        return cls(table, spec=spec)

    def _get_pads(self, qint: QInterval) -> tuple[int, int]:
        k, i, f = _minimal_kif(qint)
        if k:
            pad_left = round((qint.min + 2**i) / qint.step)
        else:
            pad_left = round(qint.min / qint.step)
        size = 2 ** (k + i + f)
        pad_right = size - len(self.table) - pad_left
        return pad_left, pad_right

    def padded_table(self, key_qint: QInterval) -> NDArray[np.int32]:
        pad_left, pad_right = self._get_pads(key_qint)
        data = np.pad(self.table, (pad_left, pad_right), mode='constant', constant_values=0)
        if key_qint.min < 0:
            size = len(data)
            data = np.roll(data, size // 2)
        return data


def _const_f(const: float | Decimal):
    """Get the minimum f such that const * 2^f is an integer."""
    const = float(const)
    if const == 0:
        return 0
    _low, _high = -32, 32
    while _high - _low > 1:
        _mid = (_high + _low) // 2
        _value = const * (2.0**_mid)
        if _value == int(_value):
            _high = _mid
        else:
            _low = _mid
    return _high


def to_csd_powers(x: float) -> Generator[float, None, None]:
    """Convert a float to a list of +/- powers of two in CSD representation."""
    if x == 0:
        return
    f = _const_f(abs(x))
    x = x * 2**f
    s = 2**-f
    N = ceil(log2(abs(x) * 1.5 + 1e-19))
    for n in range(N - 1, -1, -1):
        _2pn = 2**n
        thres = _2pn / 1.5
        bit = int(x > thres) - int(x < -thres)
        v = _2pn * bit
        x -= v
        if v != 0:
            yield v * s


class FixedVariable:
    __normal__variable__ = True

    def __init__(
        self,
        low: float | Decimal,
        high: float | Decimal,
        step: float | Decimal,
        latency: float | None = None,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(-1, -1, -1),
        opr: str = 'new',
        cost: float | None = None,
        _from: tuple['FixedVariable', ...] = (),
        _factor: float | Decimal = 1.0,
        _data: Decimal | None = None,
        _id: UUID | None = None,
    ) -> None:
        if self.__normal__variable__:
            assert low <= high, f'low {low} must be less than high {high}'

        if low != high and opr == 'const':
            raise ValueError('Constant variable must have low == high')

        if low == high:
            opr = 'const'
            _from = ()
            step = 2.0 ** -_const_f(low)

        low, high, step = Decimal(low), Decimal(high), Decimal(step)
        self.low = low
        self.high = high
        self.step = step
        self._factor = Decimal(_factor)
        self._from: tuple[FixedVariable, ...] = _from
        opr = opr
        self.opr = opr
        self._data = _data
        self.id = _id or UUID(int=rd.getrandbits(128), version=4)
        self.hwconf = HWConfig(*hwconf)

        if opr == 'cadd':
            assert _data is not None, 'cadd must have data'

        if cost is None or latency is None:
            _cost, _latency = self.get_cost_and_latency()
        else:
            _cost, _latency = cost, latency

        self.latency = _latency
        self.cost = _cost

        self._from = tuple(v if v.opr != 'const' else v._with(latency=self.latency) for v in self._from)

    def _with(self, renew_id=True, **kwargs):
        if not kwargs:
            return self
        _var = copy(self)
        for k, v in kwargs.items():
            setattr(_var, k, v)
        if renew_id:
            _var.id = UUID(int=rd.getrandbits(128), version=4)
        return _var

    def get_cost_and_latency(self) -> tuple[float, float]:
        if self.opr == 'const':
            return 0.0, 0.0

        if self.opr == 'lookup':
            assert len(self._from) == 1
            b_in = sum(self._from[0].kif)
            b_out = sum(self.kif)
            _latency = max(b_in - 6, 1) + self._from[0].latency
            _cost = 2 ** max(b_in - 5, 0) * ceil(b_out / 2)
            if b_in < 5:
                _cost *= b_in / 5
            # Assume LUT6 with extra o5 output
            return _cost, _latency

        if self.opr in ('vadd', 'cadd', 'min', 'max', 'vmul'):
            adder_size = self.hwconf.adder_size
            carry_size = self.hwconf.carry_size
            latency_cutoff = self.hwconf.latency_cutoff

            if self.opr in ('min', 'max', 'vadd'):
                assert len(self._from) == 2
                v0, v1 = self._from
                int0, int1 = v0.qint, v1.qint
                base_latency = max(v0.latency, v1.latency)
                dlat, _cost = cost_add(int0, int1, 0, False, adder_size, carry_size)
            elif self.opr == 'cadd':
                assert len(self._from) == 1
                assert self._data is not None, 'cadd must have data'
                _f = _const_f(self._data)
                _cost = float(ceil(log2(abs(self._data) + Decimal(2) ** -_f))) + _f
                base_latency = self._from[0].latency
                dlat = 0.0
            elif self.opr == 'vmul':
                assert len(self._from) == 2
                v0, v1 = self._from
                b0, b1 = sum(v0.kif), sum(v1.kif)
                int0, int1 = v0.qint, v1.qint
                dlat0, _cost0 = cost_add(int0, int0, 0, False, adder_size, carry_size)
                dlat1, _cost1 = cost_add(int1, int1, 0, False, adder_size, carry_size)
                dlat = max(dlat0 * b1, dlat1 * b0)
                _cost = min(_cost0 * b1, _cost1 * b0)
                base_latency = max(v0.latency, v1.latency)
            else:
                raise NotImplementedError(f'Operation {self.opr} is unknown')

            _latency = dlat + base_latency
            if latency_cutoff > 0 and ceil(_latency / latency_cutoff) > ceil(base_latency / latency_cutoff):
                # Crossed the latency cutoff boundry
                assert dlat <= latency_cutoff, (
                    f'Latency of an atomic operation {dlat} is larger than the pipelining latency cutoff {latency_cutoff}'
                )
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

    @classmethod
    def from_const(cls, const: float | Decimal, hwconf: HWConfig, _factor: float | Decimal = 1):
        return cls(const, const, -1, hwconf=hwconf, opr='const', _factor=_factor)

    def __repr__(self) -> str:
        if self._factor == 1:
            return f'FixedVariable({self.low}, {self.high}, {self.step})'
        return f'({self._factor}) FixedVariable({self.low}, {self.high}, {self.step})'

    def __neg__(self):
        opr = self.opr if self.low != self.high else 'const'
        return FixedVariable(
            -self.high,
            -self.low,
            self.step,
            _from=self._from,
            _factor=-self._factor,
            latency=self.latency,
            cost=self.cost,
            opr=opr,
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

        assert self.hwconf == other.hwconf, f'FixedVariable must have the same hwconf, got {self.hwconf} and {other.hwconf}'

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

    def _const_add(self, other: float | Decimal | None) -> 'FixedVariable':
        if other is None:
            return self
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

    def __truediv__(self, other: 'int|float|Decimal'):
        assert not isinstance(other, FixedVariable), 'Division by variable is not supported'
        return self * (1 / other)

    def __mul__(self, other: 'FixedVariable|int|float|Decimal') -> 'FixedVariable':
        if isinstance(other, FixedVariable):
            if self.high == self.low:
                return other * self.low
            if other.high > other.low:
                return self._var_mul(other)
            assert other.high == other.low
            other = float(other.low)

        if other == 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        if log2(abs(other)) % 1 == 0:
            return self._pow2_mul(other)

        variables = [(self._pow2_mul(v), Decimal(v)) for v in to_csd_powers(float(other))]
        while len(variables) > 1:
            v1, p1 = variables.pop()
            v2, p2 = variables.pop()
            v, p = v1 + v2, p1 + p2
            if p > 0:
                high, low = self.high * p, self.low * p
            else:
                high, low = self.low * p, self.high * p
            v.high, v.low = high, low
            variables.append((v, p))
        return variables[0][0]

    def _var_mul(self, other: 'FixedVariable') -> 'FixedVariable':
        if other is not self:
            a, b, c, d = self.high * other.low, self.low * other.high, self.high * other.high, self.low * other.low
            low = min(a, b, c, d)
            high = max(a, b, c, d)
        else:
            a, b = self.low * other.low, self.high * other.high
            if self.low < 0 and self.high > 0:
                low = min(a, b, 0)
                high = max(a, b, 0)
            else:
                low = min(a, b)
                high = max(a, b)

        step = self.step * other.step
        _factor = self._factor * other._factor
        opr = 'vmul'
        return FixedVariable(
            low,
            high,
            step,
            _from=(self, other),
            hwconf=self.hwconf,
            _factor=_factor,
            opr=opr,
        )

    def _pow2_mul(
        self,
        other: float | Decimal,
    ):
        other = Decimal(other)

        low = min(self.low * other, self.high * other)
        high = max(self.low * other, self.high * other)
        step = abs(self.step * other)
        _factor = self._factor * other
        opr = self.opr
        return FixedVariable(
            low,
            high,
            step,
            _from=self._from,
            _factor=_factor,
            opr=opr,
            latency=self.latency,
            cost=self.cost,
            _id=self.id,
            _data=self._data,
            hwconf=self.hwconf,
        )

    def __lshift__(self, other: int):
        assert isinstance(other, int), 'Shift amount must be an integer'
        shift_amount = 2.0**other
        return self * shift_amount

    def __rshift__(self, other: int):
        assert isinstance(other, int), 'Shift amount must be an integer'
        shift_amount = 2.0**-other
        return self * shift_amount

    def __radd__(self, other: 'float|Decimal|int|FixedVariable'):
        return self + other

    def __rsub__(self, other: 'float|Decimal|int|FixedVariable'):
        return (-self) + other

    def __rmul__(self, other: 'float|Decimal|int|FixedVariable'):
        return self * other

    def __pow__(self, other):
        _power = int(other)
        assert _power == other, 'Power must be an integer'
        assert _power >= 0, 'Power must be non-negative'
        if _power == 0:
            return FixedVariable(1, 1, 1, hwconf=self.hwconf, opr='const')
        if _power == 1:
            return self

        pow0 = _power // 2
        ret = (self**pow0) * (self ** (_power - pow0))
        if other % 2 == 0:
            ret.low = max(ret.low, 0)
        return ret

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
            return self.from_const(val, hwconf=self.hwconf)

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

        if self.low == low and self.high == high and self.step == step:
            return self

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
    ) -> 'FixedVariable':
        """Quantize the variable to the specified fixed-point format.

        Parameters
        ----------
        k : int | bool
            Sign bit (True for signed, False for unsigned)
        i : int
            Integer bits, excluding sign bit
        f : int
            Fraction bits
        overflow_mode : str, optional
            Overflow mode, one of 'WRAP', 'SAT', 'SAT_SYM', by default 'WRAP'
        round_mode : str, optional
            Rounding mode, one of 'TRN' (truncate), 'RND' (round to nearest, half up), by default 'TRN'
        """

        overflow_mode, round_mode = overflow_mode.upper(), round_mode.upper()
        assert overflow_mode in ('WRAP', 'SAT', 'SAT_SYM')
        assert round_mode in ('TRN', 'RND')

        if k + i + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')
        _k, _i, _f = self.kif

        if k >= _k and i >= _i and f >= _f:
            if overflow_mode != 'SAT_SYM' or i > _i:
                return self

        if f < _f and round_mode == 'RND':
            return (self + 2.0 ** (-f - 1)).quantize(k, i, f, overflow_mode, 'TRN')

        if overflow_mode in ('SAT', 'SAT_SYM'):
            step = Decimal(2) ** -f
            _high = Decimal(2) ** i
            high = _high - step
            low = -_high * k if overflow_mode == 'SAT' else -high * k
            ff = f + 1 if round_mode == 'RND' else f
            v = self.quantize(_k, _i, ff, 'WRAP', 'TRN') if _k + _i + ff > 0 else self
            return v.max_of(low).min_of(high).quantize(k, i, f, 'WRAP', round_mode)

        if self.low == self.high:
            val = self.low
            step = Decimal(2) ** -f
            _high = Decimal(2) ** i
            high, low = _high - step, -_high * k
            val = (floor(val / step) * step - low) % (2 * _high) + low
            return FixedVariable.from_const(val, hwconf=self.hwconf, _factor=1)

        f = min(f, _f)
        k = min(k, _k) if i >= _i else k

        step = Decimal(2) ** -f

        if self.low < 0:
            _low = floor(self.low / step) * step
            _i = max(_i, ceil(log2(-_low)))

        i = min(i, _i + (k == 0 and _k == 1))

        if i + k + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        low = -int(k) * Decimal(2) ** i

        high = Decimal(2) ** i - step
        _low, _high = self.low, self.high

        if _low >= low and _high <= high:
            low = floor(_low / step) * step
            high = floor(_high / step) * step

        return FixedVariable(
            low,
            high,
            step,
            _from=(self,),
            _factor=abs(self._factor),
            opr='wrap',
            latency=self.latency,
            hwconf=self.hwconf,
        )

    @classmethod
    def from_kif(cls, k: int | bool, i: int, f: int, **kwargs):
        step = Decimal(2) ** -f
        _high = Decimal(2) ** i
        low, high = -k * _high, _high - step
        return cls(low, high, step, **kwargs)

    def msb_mux(
        self,
        a: 'FixedVariable|float|Decimal',
        b: 'FixedVariable|float|Decimal',
        qint: tuple[Decimal, Decimal, Decimal] | None = None,
    ):
        """If the MSB of this variable is 1, return a, else return b.
        When the variable is signed, the MSB is determined by the sign bit (1 for <0, 0 for >=0)
        """
        if not isinstance(a, FixedVariable):
            a = FixedVariable.from_const(a, hwconf=self.hwconf, _factor=1)
        if not isinstance(b, FixedVariable):
            b = FixedVariable.from_const(b, hwconf=self.hwconf, _factor=1)
        if self._factor < 0:
            return (-self).msb_mux(b, a, qint)

        if self.opr == 'const':
            if self.low >= 0:
                return b
            else:
                return b if log2(abs(self.low)) % 1 == 0 else a
        elif self.opr == 'quantize':
            k, i, _ = self.kif
            pk, pi, _ = self._from[0].kif
            if k + i == pk + pi:
                return self._from[0].msb_mux(a, b, qint=qint)

        if a._factor < 0:
            qint = (-qint[1], -qint[0], qint[2]) if qint else None
            return -(self.msb_mux(-a, -b, qint=qint))

        _factor = a._factor

        if qint is None:
            qint = (min(a.low, b.low), max(a.high, b.high), min(a.step, b.step))

        dlat, dcost = cost_add(a.qint, b.qint, 0, False, self.hwconf.adder_size, self.hwconf.carry_size)
        return FixedVariable(
            *qint,
            _from=(self, a, b),
            _factor=_factor,
            opr='msb_mux',
            latency=max(a.latency, b.latency, self.latency) + dlat,
            hwconf=self.hwconf,
            cost=dcost,
        )

    def is_negative(self) -> 'FixedVariable|bool':
        if self.low >= 0:
            return False
        if self.high < 0:
            return True
        _, i, _ = self.kif
        sign_bit = self.quantize(0, i + 1, -i) >> i
        return sign_bit

    def is_positive(self) -> 'FixedVariable|bool':
        return (-self).is_negative()

    def __abs__(self):
        if self.low >= 0:
            return self
        step = self.step
        high = max(-self.low, self.high)
        return self.msb_mux(-self, self, (Decimal(0), high, step))

    def abs(self):
        """Get the absolute value of this variable."""
        return abs(self)

    def __gt__(self, other: 'FixedVariable|float|Decimal|int'):
        """Get a variable that is 1 if this variable is greater than other, else 0."""
        return (self - other).is_positive()

    def __lt__(self, other: 'FixedVariable|float|Decimal|int'):
        """Get a variable that is 1 if this variable is less than other, else 0."""
        return (other - self).is_positive()

    # def __ge__(self, other: 'FixedVariable|float|Decimal|int'):
    #     """Get a variable that is 1 if this variable is greater than or equal to other, else 0."""
    #     r = (other - self).is_negative()
    #     if isinstance(r, bool):
    #         return not r
    #     return ~r

    # def __le__(self, other: 'FixedVariable|float|Decimal|int'):
    #     """Get a variable that is 1 if this variable is less than or equal to other, else 0."""
    #     r = (self - other).is_negative()
    #     if isinstance(r, bool):
    #         return not r
    #     return ~r

    def max_of(self, other):
        """Get the maximum of this variable and another variable or constant."""
        if other == -float('inf'):
            return self
        if other == float('inf'):
            raise ValueError('Cannot apply max_of with inf')
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=abs(self._factor))

        if self.low >= other.high:
            return self
        if self.high <= other.low:
            return other
        if other.high == other.low == 0:
            return self.relu()

        qint = (max(self.low, other.low), max(self.high, other.high), min(self.step, other.step))
        return (self - other).msb_mux(other, self, qint=qint)

    def min_of(self, other):
        """Get the minimum of this variable and another variable or constant."""

        if other == float('inf'):
            return self
        if other == -float('inf'):
            raise ValueError('Cannot apply min_of with -inf')
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=(self._factor))

        if self.high <= other.low:
            return self
        if self.low >= other.high:
            return other
        if other.high == other.low == 0:
            return -(-self).relu()

        qint = (min(self.low, other.low), min(self.high, other.high), min(self.step, other.step))
        return (self - other).msb_mux(self, other, qint=qint)

    def lookup(self, table: LookupTable | np.ndarray) -> 'FixedVariable':
        """Use a lookup table to map the variable.
        When the table is a numpy array, the table starts at the lowest possible value of the variable
        When the table is in LookupTable format, the table starts at the normalized lowest value of the variable. (i.e., if the variable has negative _factor, the table is reversed)

        Parameters
        ----------
        table : LookupTable | np.ndarray
            Lookup table to use

        Returns
        -------
        FixedVariable
        """
        if isinstance(table, np.ndarray):
            if len(table) == 1:
                return self.from_const(float(table[0]), hwconf=self.hwconf)
            if self._factor < 0:
                table = table[::-1]  # Reverse the table for negative factor

        _table, table_id = table_context.register_table(table)
        size = len(table.table) if isinstance(table, LookupTable) else len(table)
        assert round((self.high - self.low) / self.step) + 1 == size, (
            f'Input variable size does not match lookup table size ({round((self.high - self.low) / self.step) + 1} != {size})'
        )

        return FixedVariable(
            _table.spec.out_qint.min,
            _table.spec.out_qint.max,
            _table.spec.out_qint.step,
            _from=(self,),
            _factor=Decimal(1),
            opr='lookup',
            hwconf=self.hwconf,
            _data=Decimal(table_id),
        )


class FixedVariableInput(FixedVariable):
    __normal__variable__ = False

    def __init__(
        self,
        latency: float | None = None,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(-1, -1, -1),
        opr: str = 'new',
    ) -> None:
        super().__init__(
            low=Decimal(1e10),
            high=Decimal(-1e10),
            step=Decimal(1e10),
            latency=latency if latency is not None else 0.0,
            hwconf=HWConfig(*hwconf),
            opr=opr,
            cost=0.0,
            _factor=Decimal(1),
            _from=(),
            _data=None,
            _id=None,
        )

    def __add__(self, other):
        if other == 0:
            return self
        raise ValueError('Cannot operate on unquantized input variable')

    def __sub__(self, other):
        if other == 0:
            return self
        raise ValueError('Cannot operate on unquantized input variable')

    def __neg__(self):
        raise ValueError('Cannot negate unquantized input variable')

    def __mul__(self, other):
        if other == 1:
            return self
        raise ValueError('Cannot multiply unquantized input variable')

    def __rmul__(self, other):
        if other == 1:
            return self
        raise ValueError('Cannot multiply unquantized input variable')

    def __radd__(self, other):
        if other == 0:
            return self
        raise ValueError('Cannot add unquantized input variable')

    def __rsub__(self, other):
        raise ValueError('Cannot subtract unquantized input variable')

    def relu(self, *args, **kwargs):
        raise ValueError('Cannot apply relu on unquantized input variable')

    def max_of(self, other):
        raise ValueError('Cannot apply max_of on unquantized input variable')

    def min_of(self, other):
        raise ValueError('Cannot apply min_of on unquantized input variable')

    def quantize(
        self,
        k: int | bool,
        i: int,
        f: int,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        assert overflow_mode == 'WRAP'

        if k + i + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        if round_mode == 'RND':
            return (self.quantize(k, i, f + 1) + 2.0 ** (-f - 1)).quantize(k, i, f, overflow_mode, 'TRN')
        else:
            round_mode = 'TRN'

        step = Decimal(2) ** -f
        _high = Decimal(2) ** i
        low, high = -_high * k, _high - step
        self.high = max(self.high, high)
        self.low = min(self.low, low)
        self.step = min(self.step, step)

        return FixedVariable(
            low,
            high,
            step,
            _from=(self,),
            _factor=self._factor,
            opr='wrap',
            latency=self.latency,
            hwconf=self.hwconf,
        )
