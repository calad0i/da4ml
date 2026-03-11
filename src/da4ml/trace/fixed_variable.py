import random
from collections.abc import Callable, Generator
from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, floor, log2
from typing import Any, NamedTuple, overload
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from .._binary.cmvm_bin import cost_add, get_lsb_loc, iceil_log2, solve
from ..types import QInterval, minimal_kif
from .affine_interval import AffineInterval

rd = random.Random()


class HWConfig(NamedTuple):
    adder_size: int
    carry_size: int
    latency_cutoff: float


ufunc_t = Callable[[NDArray[np.floating]], NDArray[np.floating]]


@dataclass
class TableSpec:
    hash: str
    out_qint: QInterval
    inp_width: int

    @property
    def out_kif(self) -> tuple[bool, int, int]:
        return minimal_kif(self.out_qint)


def to_spec(table: NDArray[np.floating]) -> tuple[TableSpec, NDArray[np.int32], NDArray[np.bool_] | None]:
    mask = np.isnan(table)
    f_out = max(-get_lsb_loc(float(x)) for x in table.ravel() if not np.isnan(x))
    int_table = np.where(mask, 0, table * 2**f_out).astype(np.int32)
    h = sha256(int_table.data)
    if mask.any():
        h.update(b'mask')
        h.update(mask.data)
    h.update(f'{f_out}'.encode())
    inp_width = ceil(log2(table.size))
    out_qint = QInterval(float(np.min(table[~mask])), float(np.max(table[~mask])), float(2**-f_out))
    mask = mask if mask.any() else None
    return TableSpec(hash=h.hexdigest(), inp_width=inp_width, out_qint=out_qint), int_table, mask


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
    def __init__(self, values: NDArray, spec: TableSpec | None = None, mask: NDArray[np.bool_] | None = None):
        assert values.ndim == 1, 'Lookup table values must be 1-dimensional'
        if spec is not None:
            assert values.dtype == np.int32, f'{values.dtype}'
            self.spec = spec
            self.table = values
            self.mask = mask
        else:
            self.spec, self.table, self.mask = to_spec(values)

    @overload
    def lookup(self, var: 'FixedVariable', qint_in: QInterval) -> 'FixedVariable': ...

    @overload
    def lookup(self, var: np.floating | float, qint_in: QInterval | tuple[float, float, float]) -> float: ...

    def lookup(self, var, qint_in: QInterval | tuple[float, float, float]):
        if isinstance(var, FixedVariable):
            return var.lookup(self, original_qint=qint_in)
        else:
            _min, _max, _step = qint_in
            assert _min <= var <= _max, f'Value {var} out of range [{_min}, {_max}]'
            index = round((var - _min) / _step)
            assert self.mask is None or not self.mask[index], f'Value {var} is masked out in the lookup table'
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
        k, i, f = minimal_kif(qint)
        if k:
            pad_left = round((qint.min + 2**i) / qint.step)
        else:
            pad_left = round(qint.min / qint.step)
        size = 2 ** (k + i + f)
        pad_right = size - len(self.table) - pad_left
        return pad_left, pad_right

    def padded_table(self, key_qint: QInterval) -> NDArray[np.float64]:
        pad_left, pad_right = self._get_pads(key_qint)
        _table = self.table.astype(np.float64)
        _table = np.where(self.mask, np.nan, _table) if self.mask is not None else _table
        data = np.pad(_table, (pad_left, pad_right), mode='constant', constant_values=np.nan)
        if key_qint.min < 0:
            size = len(data)
            data = np.roll(data, size // 2)
        return data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LookupTable):
            return False
        return self.spec == other.spec and np.array_equal(self.table, other.table)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, item) -> 'LookupTable':
        table = self.float_table[item]
        _mask = self.mask[item] if self.mask is not None else None
        _table, _spec, _ = to_spec(table)
        return LookupTable(_spec, _table, _mask)


def to_csd_powers(x: float) -> Generator[float, None, None]:
    """Convert a float to a list of +/- powers of two in CSD representation."""
    if x == 0:
        return
    f = -get_lsb_loc(x)
    x = x * 2**f
    s = 2**-f
    N = iceil_log2(x * 1.5)
    for n in range(N - 1, -1, -1):
        _2pn = 2**n
        thres = _2pn / 1.5
        bit = int(x > thres) - int(x < -thres)
        v = _2pn * bit
        x -= v
        if v != 0:
            yield v * s


def _binary_bit_op(a: float, b: float, op: int, qint0: QInterval, qint1: QInterval, qint: QInterval):
    _fn = {0: lambda x, y: x & y, 1: lambda x, y: x | y, 2: lambda x, y: x ^ y}[op]
    assert isinstance(a, float) and isinstance(b, float)
    assert qint0 is not None and qint1 is not None and qint is not None
    k, i, f = minimal_kif(qint)
    step = min(qint0.step, qint1.step)
    _a, _b = round(a / step), round(b / step)
    return interpret_as(_fn(_a, _b), k, i, f)


def _unary_bit_op(a: float, op: int, qint_from: QInterval, qint_to: QInterval | None = None) -> float:
    assert isinstance(a, float)
    assert qint_from is not None
    k, i, f = minimal_kif(qint_from) if qint_from.min != 0 or qint_from.max != 0 else (False, 1, 0)
    _a = round(a / qint_from.step)
    match op:
        case 0:
            if not qint_to:
                return interpret_as(~_a, k, i, f)
            kk, ii, ff = minimal_kif(qint_to)
            return interpret_as((~_a) % 2 ** (k + i + f), kk, ii, ff)
        case 1:
            return float(_a != 0)
        case 2:
            return float(_a == qint_from.max) if qint_from.min >= 0 else float(_a == -1)
        case _:
            raise ValueError(f'Invalid unary bit op {op}')


class FixedVariable:
    __normal__variable__ = True

    def __init__(
        self,
        low: float | None = None,
        high: float | None = None,
        step: float | None = None,
        *,
        latency: float | None = None,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(-1, -1, -1),
        opr: str = 'new',
        cost: float | None = None,
        _from: tuple['FixedVariable', ...] = (),
        _factor: float = 1.0,
        _data: int | None = None,
        _id: UUID | None = None,
        _affine: AffineInterval | None = None,
        _table: LookupTable | None = None,
    ) -> None:
        self._factor = float(_factor)
        self._from: tuple[FixedVariable, ...] = _from
        self.opr = opr
        self._data = _data
        self.id = _id or UUID(int=rd.getrandbits(128), version=4)
        self.hwconf = HWConfig(*hwconf)
        self._table = _table

        if _affine is not None:
            assert low is None and high is None and step is None, 'Cannot specify both affine and low/high/step'
            self._affine = _affine
            q = _affine.qint
            if q.min == q.max:
                self.opr = 'const'
                self._from = ()
        else:
            assert low is not None and high is not None and step is not None, (
                'Must specify low, high, and step if affine is not provided'
            )
            low, high, step = float(low), float(high), float(step)
            if self.__normal__variable__:
                assert low <= high, f'low {low} must be less than high {high}'
            if low != high and opr == 'const':
                raise ValueError('Constant variable must have low == high')
            if low == high:
                self.opr = 'const'
                self._from = ()
                self._affine = AffineInterval({}, low)
            else:
                self._affine = AffineInterval.new(QInterval(low, high, step))

        if self.opr == 'cadd':
            assert self._data is not None, 'cadd must have data'

        if cost is None or latency is None:
            _cost, _latency = self.get_cost_and_latency()
        else:
            _cost, _latency = cost, latency

        self.latency = _latency
        self.cost = _cost

        # self._from = tuple(v if v.opr != 'const' else v._with(latency=self.latency) for v in self._from)

    @property
    def low(self) -> float:
        return self._affine.qint.min

    @property
    def high(self) -> float:
        return self._affine.qint.max

    @property
    def step(self) -> float:
        return self._affine.qint.step

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
                # Reconstruct float bias for cost estimation
                unscaled_step = self.step / abs(self._factor)
                float_data = self._data * unscaled_step
                _f = -get_lsb_loc(float_data)
                _cost = float(ceil(log2(abs(float_data) + 2.0**-_f))) + _f
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
                assert dlat <= latency_cutoff, (
                    f'Latency of an atomic operation {dlat} is larger than the pipelining latency cutoff {latency_cutoff}'
                )
                _latency = ceil(base_latency / latency_cutoff) * latency_cutoff + dlat

        elif self.opr in ('relu', 'wrap'):
            assert len(self._from) == 1
            _latency = self._from[0].latency
            _cost = 0.0
            if self._from[0]._factor < 0:
                _cost += sum(self.kif) / 2
            if self.opr == 'relu':
                _cost += sum(self.kif) / 2

        elif self.opr == 'bit_binary':
            _cost = sum(self.kif) * 0.2
            _latency = 1.0 + max(v.latency for v in self._from)

        elif self.opr == 'bit_unary':
            if self._data == 0:
                _cost = 0.0
                _latency = self._from[0].latency
            else:
                _cost = sum(self._from[0].kif) / 6
                _latency = 1.0 + max(v.latency for v in self._from)
        elif self.opr == 'new':
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
        return self._affine.qint

    @property
    def kif(self) -> tuple[bool, int, int]:
        if self.step == 0:
            return False, 0, 0
        f = -int(log2(self.step))
        xx = max(-self.low, self.high + self.step)
        i = ceil(log2(xx))
        k = self.low < 0
        return k, i, f

    @classmethod
    def from_const(cls, const: float, hwconf: HWConfig, _factor: float = 1):
        const = float(const)
        return cls(const, const, -1, hwconf=hwconf, opr='const', _factor=_factor)

    def __repr__(self) -> str:
        if self._factor == 1:
            return f'FixedVariable({self.low}, {self.high}, {self.step})'
        return f'({self._factor}) FixedVariable({self.low}, {self.high}, {self.step})'

    def __neg__(self):
        return FixedVariable(
            _affine=-self._affine,
            _from=self._from,
            _factor=-self._factor,
            latency=self.latency,
            cost=self.cost,
            opr=self.opr,
            _id=self.id,
            _data=self._data,
            hwconf=self.hwconf,
        )

    def __add__(self, other: 'FixedVariable|float|int'):
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

        _affine = self._affine + other._affine

        return FixedVariable(
            _affine=_affine,
            _from=(self, other),
            _factor=f0,
            opr='vadd',
            hwconf=self.hwconf,
        )

    def _const_add(self, other: float | None) -> 'FixedVariable':
        if other is None:
            return self
        other = float(other)
        if other == 0:
            return self

        if self.opr != 'cadd':
            _affine = self._affine + other
            q = _affine.qint
            unscaled_step = q.step / abs(self._factor)
            _data = round(other / self._factor / unscaled_step)

            return FixedVariable(
                _affine=_affine,
                _from=(self,),
                _factor=self._factor,
                _data=_data,
                opr='cadd',
                hwconf=self.hwconf,
            )

        assert len(self._from) == 1
        parent = self._from[0]
        assert self._data is not None, 'cadd must have data'
        unscaled_bias = self._data * self.step / abs(self._factor)
        sf = self._factor / parent._factor
        other1 = (unscaled_bias * parent._factor) + other / sf
        return (parent + other1) * sf

    def __sub__(self, other: 'FixedVariable|int|float'):
        return self + (-other)

    def __truediv__(self, other: 'int|float'):
        assert not isinstance(other, FixedVariable), 'Division by variable is not supported'
        return self * (1 / other)

    def __mul__(self, other: 'FixedVariable|int|float') -> 'FixedVariable':
        if isinstance(other, FixedVariable):
            if self.high == self.low:
                return other * self.low
            if other.high > other.low:
                return self._var_mul(other)
            assert other.high == other.low
            other = float(other.low)

        if self.high == self.low:
            return self.from_const(float(self.low) * float(other), hwconf=self.hwconf)

        if np.all(other == 0):
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        if log2(abs(other)) % 1 == 0:
            return self._pow2_mul(other)

        ker = np.array([[other]], dtype=np.float32)
        sol = solve(
            ker,
            decompose_dc=-1,
            qintervals=[self._affine.qint],
            latencies=[self.latency],
            adder_size=self.hwconf.adder_size,
            carry_size=self.hwconf.carry_size,
        )
        return sol([self])[0]

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
        other: float,
    ):
        other = float(other)

        _affine = self._affine * other
        return self._with(_affine=_affine, _factor=self._factor * other, renew_id=False)

    def __lshift__(self, other: int):
        assert isinstance(other, int), 'Shift amount must be an integer'
        shift_amount = 2.0**other
        return self * shift_amount

    def __rshift__(self, other: int):
        assert isinstance(other, int), 'Shift amount must be an integer'
        shift_amount = 2.0**-other
        return self * shift_amount

    def __radd__(self, other: 'float|int|FixedVariable'):
        return self + other

    def __rsub__(self, other: 'float|int|FixedVariable'):
        return (-self) + other

    def __rmul__(self, other: 'float|int|FixedVariable'):
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
            # Even power: result is non-negative. Tighten via new affine.
            new_low = max(ret.low, 0)
            ret._affine = AffineInterval.new(QInterval(new_low, ret.high, ret.step))
        return ret

    def relu(self, i: int | None = None, f: int | None = None, round_mode: str = 'TRN'):
        round_mode = round_mode.upper()
        assert round_mode in ('TRN', 'RND')

        if self.opr == 'const':
            val = self.low * (self.low > 0)
            f = -get_lsb_loc(val) if not f else f
            step = 2.0**-f
            i = ceil(log2(val + step)) if not i else i
            eps = step / 2 if round_mode == 'RND' else 0
            val = (floor(val / step + eps) * step) % (2.0**i)
            return self.from_const(val, hwconf=self.hwconf)

        step = max(2.0**-f, self.step) if f is not None else self.step
        if step > self.step and round_mode == 'RND':
            return (self + step / 2).relu(i, f, 'TRN')
        low = max(0.0, self.low)
        high = self.high
        high, low = floor(high / step) * step, floor(low / step) * step

        if i is not None:
            _high = 2.0**i - step
            if _high < high:
                low = 0.0
                high = _high
        _factor = self._factor
        high = max(0.0, high)

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
        _force_factor_clear=False,
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
        _force_factor_clear : bool, optional
            Whether to force clear the scaling factor (set to 1) in the output variable, by default False.
        """

        overflow_mode, round_mode = overflow_mode.upper(), round_mode.upper()
        assert overflow_mode in ('WRAP', 'SAT', 'SAT_SYM')
        assert round_mode in ('TRN', 'RND')

        if k + i + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')
        _k, _i, _f = self.kif

        if k >= _k and i >= _i and f >= _f and not _force_factor_clear:
            if overflow_mode != 'SAT_SYM' or i > _i:
                return self

        if f < _f and round_mode == 'RND':
            return (self + 2.0 ** (-f - 1)).quantize(k, i, f, overflow_mode, 'TRN')

        if overflow_mode in ('SAT', 'SAT_SYM'):
            step = 2.0**-f
            _high = 2.0**i
            high = _high - step
            low = -_high * k if overflow_mode == 'SAT' else -high * k
            ff = f + 1 if round_mode == 'RND' else f
            v = self.quantize(_k, _i, ff, 'WRAP', 'TRN') if _k + _i + ff > 0 else self
            return v.max_of(low).min_of(high).quantize(k, i, f, 'WRAP', round_mode)

        if self.low == self.high:
            val = self.low
            step = 2.0**-f
            _high = 2.0**i
            high, low = _high - step, -_high * k
            val = (floor(val / step) * step - low) % (2 * _high) + low
            return FixedVariable.from_const(val, hwconf=self.hwconf, _factor=1)

        f = min(f, _f)
        k = min(k, _k) if i >= _i else k

        step = 2.0**-f

        if self.low < 0:
            _low = floor(self.low / step) * step
            _i = max(_i, ceil(log2(-_low)))

        i = min(i, _i + (k == 0 and _k == 1))

        if i + k + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        low = -int(k) * 2.0**i

        high = 2.0**i - step
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
        step = 2.0**-f
        _high = 2.0**i
        low, high = -k * _high, _high - step
        return cls(low, high, step, **kwargs)

    def msb_mux(
        self,
        a: 'FixedVariable|float',
        b: 'FixedVariable|float',
        qint: tuple[float, float, float] | None = None,
        zt_sensitive: bool = True,
    ):
        """If the MSB of this variable is 1, return a, else return b.
        When the variable is signed, the MSB is determined by the sign bit (1 for <0, 0 for >=0)
        """

        if not isinstance(a, FixedVariable):
            a = FixedVariable.from_const(a, hwconf=self.hwconf, _factor=1)
        if not isinstance(b, FixedVariable):
            b = FixedVariable.from_const(b, hwconf=self.hwconf, _factor=1)
        if self._factor < 0:
            if zt_sensitive:
                return self.msb().msb_mux(a, b, qint)
            else:
                return (-self).msb_mux(b, a, qint, zt_sensitive=False)

        if self.opr == 'const':
            if self.low >= 0:
                return b if self.high == 0 else a
            else:
                return b if log2(abs(self.low)) % 1 == 0 else a
        if self.opr == 'wrap':
            k, i, _ = self.kif
            k0, i0, _ = self._from[0].kif
            _factor = self._factor
            _factor0 = self._from[0]._factor
            if k + i == k0 + i0 + log2(abs(_factor / _factor0)):
                if _factor * _factor0 > 0 or not zt_sensitive:
                    return self._from[0].msb_mux(a, b, qint=qint, zt_sensitive=zt_sensitive)

        if a._factor < 0:
            qint = (-qint[1], -qint[0], qint[2]) if qint else None
            return -(self.msb_mux(-a, -b, qint=qint, zt_sensitive=zt_sensitive))

        _factor = a._factor

        if qint is None:
            qint = (float(min(a.low, b.low)), float(max(a.high, b.high)), float(min(a.step, b.step)))
        else:
            _min, _max, _step = qint
            step = float(min(a.step, b.step))
            assert _step <= step, (
                f'MSB mux cannot imply rounding operation, but its {_step} is larger than min(a.step {a.step}, b.step {b.step})'
            )
            _min = max(floor(_min / step) * step, float(min(a.low, b.low)))
            _max = min(floor(_max / step) * step, float(max(a.high, b.high)))
            qint = (_min, _max, step)

        dlat, dcost = cost_add(a.qint, b.qint, 0, False, self.hwconf.adder_size, self.hwconf.carry_size)
        dcost = dcost / 2

        if a.opr == 'const' and a._factor != b._factor:
            _factor = b._factor
            a = a._with(_factor=b._factor, renew_id=True)
        if b.opr == 'const' and a._factor != b._factor:
            _factor = a._factor
            b = b._with(_factor=a._factor, renew_id=True)

        return FixedVariable(
            *qint,
            _from=(self, a, b),
            _factor=_factor,
            opr='msb_mux',
            latency=max(a.latency, b.latency, self.latency) + dlat,
            hwconf=self.hwconf,
            cost=dcost,
        )

    def is_negative(self) -> 'FixedVariable':
        if self.low >= 0:
            return self.from_const(0, hwconf=self.hwconf)
        if self.high < 0:
            return self.from_const(1, hwconf=self.hwconf)
        return self.msb()

    def msb(self) -> 'FixedVariable':
        k, i, f = self.kif
        return self.quantize(0, i + k, -i - k + 1, _force_factor_clear=True) >> i + k - 1

    def is_positive(self) -> 'FixedVariable':
        return (-self).is_negative()

    def __abs__(self):
        if self.low >= 0:
            return self
        step = self.step
        high = max(-self.low, self.high)
        return self.msb_mux(-self, self, (0, float(high), float(step)), zt_sensitive=False)

    def abs(self):
        """Get the absolute value of this variable."""
        return abs(self)

    def __gt__(self, other: 'FixedVariable|float|int'):
        """Get a variable that is 1 if this variable is greater than other, else 0."""
        if not isinstance(other, FixedVariable) or other.opr == 'const':
            _other = float(other) if not isinstance(other, FixedVariable) else other.low
            _other_align = ceil(_other / self.step) * self.step
            if _other != _other_align:
                return self >= _other_align
            if self.low == _other:
                return self._ne(other)
        return (self - other).is_positive()

    def __lt__(self, other: 'FixedVariable|float|int'):
        """Get a variable that is 1 if this variable is less than other, else 0."""
        if not isinstance(other, FixedVariable) or other.opr == 'const':
            _other = float(other) if not isinstance(other, FixedVariable) else other.low
            _other_align = ceil(_other / self.step) * self.step
            if _other != _other_align:
                return self < _other_align
            if self.high == _other:
                return self._ne(other)
        return (other - self).is_positive()

    def __ge__(self, other: 'FixedVariable|float|int'):
        """Get a variable that is 1 if this variable is greater than or equal to other, else 0."""
        return ~(self < other)

    def __le__(self, other: 'FixedVariable|float|int'):
        """Get a variable that is 1 if this variable is less than or equal to other, else 0."""
        return ~(self > other)

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

        _qint = (max(self.low, other.low), max(self.high, other.high), min(self.step, other.step))
        qint = (float(_qint[0]), float(_qint[1]), float(_qint[2]))
        return (self - other).msb_mux(other, self, qint=qint, zt_sensitive=False)

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

        _qint = (min(self.low, other.low), min(self.high, other.high), min(self.step, other.step))
        qint = (float(_qint[0]), float(_qint[1]), float(_qint[2]))
        return (self - other).msb_mux(self, other, qint=qint, zt_sensitive=False)

    def lookup(self, table: LookupTable | np.ndarray, original_qint: tuple[float, float, float] | None = None) -> 'FixedVariable':
        """Use a lookup table to map the variable.

        Parameters
        ----------
        table : LookupTable | np.ndarray
            Lookup table to use
        original_qint : tuple[float, float, float] | None
            The original quantization interval of the variable where the original table is applied to.

        Returns
        -------
        FixedVariable
        """

        size = len(table)

        was_numpy_table = isinstance(table, np.ndarray)
        if original_qint is not None:
            o_min, o_max, o_step = original_qint
            assert round((o_max - o_min) / o_step) + 1 == size, f'table size {size} does not match original qint {original_qint}'
            _min, _max, _step = self.qint
            assert o_step <= _step and o_max >= _max and o_min <= _min, (
                f'Original quantization interval {original_qint} does not cover all values of the variable {self.qint}.'
            )
            _bias_0 = round((_min - o_min) / o_step)
            _bias_1 = round((o_max - _max) / o_step)
            stride = round(_step / o_step)
            s = slice(_bias_0, size - _bias_1, stride)
            table = table[s]
            size = len(table)

        assert round((self.high - self.low) / self.step) + 1 == size, (
            f'Input variable size does not match lookup table size ({round((self.high - self.low) / self.step) + 1} != {size})'
        )

        if was_numpy_table and isinstance(table, np.ndarray):
            if len(table) == 1:
                return self.from_const(float(table[0]), hwconf=self.hwconf)
            if self._factor < 0:
                table = table[::-1]

        if isinstance(table, np.ndarray):
            table = LookupTable(table)

        return FixedVariable(
            *table.spec.out_qint, _from=(self,), _factor=1.0, opr='lookup', hwconf=self.hwconf, _data=None, _table=table
        )

    def unary_bit_op(self, _type: str):
        ops = {
            'not': 0,
            'any': 1,
            'all': 2,
        }
        if self.opr == 'const':
            qint = QInterval(float(self.low), float(self.high), float(self.step))
            v = _unary_bit_op(float(self.low), ops[_type], qint)
            return self.from_const(v, hwconf=self.hwconf)

        if sum(self.kif) == 1 and _type in ('any', 'all'):
            return self.msb()

        _data = ops[_type]
        if _type == 'not':
            if self.opr == 'bit_unary' and self._data == 0:
                return self._from[0]
            k, i, f = self.kif
            return FixedVariable.from_kif(
                k, i, f, hwconf=self.hwconf, opr='bit_unary', _data=_data, _from=(self,), _factor=abs(self._factor)
            )
        if _type == 'all':
            if self.low > 0:
                return self.from_const(0, hwconf=self.hwconf)
            if self.high < -self.step:
                return self.from_const(0, hwconf=self.hwconf)
            if self.low == 0:
                _max = log2(self.high + self.step)
                if _max % 1 != 0:
                    return self.from_const(0, hwconf=self.hwconf)

        return FixedVariable(
            0, 1, 1, hwconf=self.hwconf, opr='bit_unary', _data=int(_data), _from=(self,), _factor=abs(self._factor)
        )

    def binary_bit_op(self, other: 'FixedVariable', _type: str):
        ops = {
            'and': 0,
            'or': 1,
            'xor': 2,
        }
        k, i, f = self.kif
        k_other, i_other, f_other = other.kif
        k, i, f = max(k, k_other), max(i, i_other), max(f, f_other)
        qint = QInterval(-k * 2.0**i, 2.0**i - 2.0**-f, 2.0**-f)
        if self.opr == 'const' and other.opr == 'const':
            qint0 = QInterval(float(self.low), float(self.high), float(self.step))
            qint1 = QInterval(float(other.low), float(other.high), float(other.step))
            v = _binary_bit_op(float(self.low), float(other.low), ops[_type], qint0, qint1, qint)
            return self.from_const(v, hwconf=self.hwconf)
        if self.opr == 'const' and other.opr != 'const':
            return other.binary_bit_op(self, _type)

        if other.opr == 'const':
            if other.low == 0:  # 0
                if _type == 'and':
                    return other
                if _type == 'or' or _type == 'xor':
                    return self
            _ones_neg = -self.step
            _ones_pos = (2.0**i) - self.step
            if (k == 0 and other.low == _ones_pos) or (k == 1 and other.low == _ones_neg):
                if _type == 'and':
                    return self
                if _type == 'or':
                    return other
                if _type == 'xor':
                    return self.unary_bit_op('not')

        _data = ops[_type]
        return FixedVariable(
            *qint, hwconf=self.hwconf, opr='bit_binary', _data=_data, _from=(self, other), _factor=abs(self._factor)
        )

    def __and__(self, other: 'FixedVariable|float|int'):
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=abs(self._factor))
        return self.binary_bit_op(other, 'and')

    def __or__(self, other: 'FixedVariable|float|int'):
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=abs(self._factor))
        return self.binary_bit_op(other, 'or')

    def __xor__(self, other: 'FixedVariable|float|int'):
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=abs(self._factor))
        return self.binary_bit_op(other, 'xor')

    def __rand__(self, other: 'float|int|FixedVariable'):
        return self.__and__(other)

    def __ror__(self, other: 'float|int|FixedVariable'):
        return self.__or__(other)

    def __rxor__(self, other: 'float|int|FixedVariable'):
        return self.__xor__(other)

    def __invert__(self):
        return self.unary_bit_op('not')

    def _ne(self, other):
        if not isinstance(other, FixedVariable):
            other = FixedVariable.from_const(other, hwconf=self.hwconf, _factor=abs(self._factor))
        return (self ^ other).unary_bit_op('any')

    def _eq(self, other):
        return ~(self._ne(other))


class FixedVariableInput(FixedVariable):
    __normal__variable__ = False

    def __init__(
        self,
        latency: float | None = None,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(-1, -1, -1),
        opr: str = 'new',
    ) -> None:
        # Accumulators for tracking widest quantization range seen
        self._bounds_low = 1e10
        self._bounds_high = -1e10
        self._bounds_step = 1e10
        super().__init__(
            low=self._bounds_low,
            high=self._bounds_high,
            step=self._bounds_step,
            latency=latency if latency is not None else 0.0,
            hwconf=HWConfig(*hwconf),
            opr=opr,
            cost=0.0,
            _factor=1.0,
            _from=(),
            _data=None,
            _id=None,
        )

    @property
    def low(self) -> float:
        return self._bounds_low

    @property
    def high(self) -> float:
        return self._bounds_high

    @property
    def step(self) -> float:
        return self._bounds_step

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
        _force_factor_clear=False,
    ):
        assert overflow_mode == 'WRAP'
        k, i, f = int(k), int(i), int(f)

        if k + i + f <= 0:
            return FixedVariable(0, 0, 1, hwconf=self.hwconf, opr='const')

        if round_mode == 'RND':
            return (self.quantize(k, i, f + 1) + 2.0 ** (-f - 1)).quantize(k, i, f, overflow_mode, 'TRN')
        else:
            round_mode = 'TRN'

        step = 2.0**-f
        _high = 2.0**i
        low, high = -_high * k, _high - step

        # Update accumulators
        self._bounds_high = max(self._bounds_high, high)
        self._bounds_low = min(self._bounds_low, low)
        self._bounds_step = min(self._bounds_step, step)
        # Rebuild affine from accumulated bounds
        self._affine = AffineInterval.new(QInterval(self._bounds_low, self._bounds_high, self._bounds_step))

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
