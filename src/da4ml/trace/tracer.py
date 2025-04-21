from collections.abc import Sequence
from decimal import Decimal
from math import ceil, log2
from typing import NamedTuple
from uuid import uuid4

import numpy as np

from ..cmvm.core import cost_add
from ..cmvm.types import Op, QInterval, Solution


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class HWConfig(NamedTuple):
    adder_size: int
    carry_size: int
    latency_cutoff: float


class Trace:
    def __init__(
        self,
        hwconf: HWConfig,
        ops: Sequence[Op] | None = None,
        out_idx: Sequence[int] | None = None,
        out_factors: Sequence[float] | None = None,
    ):
        self.ops: list[Op] = list(ops) if ops is not None else []
        self.hwconf = hwconf
        self.out_idx: list[int] = list(out_idx) if out_idx is not None else []
        self.out_factors: list[float] = list(out_factors) if out_factors is not None else []

    def add(self, v: 'FixedVariable'):
        if v._from is None:
            id0 = v._id
            id1 = -1
            op = Op(id0, id1, False, 0, v.unscaled.qint, v.latency, 0.0)
        else:
            v0, v1 = v._from
            assert v0._factor > 0 or v._from[1]._id == -2  # relu being exception here
            id0, id1 = v0._id, v1._id
            sub = v1._factor < 0
            shift = int(log2(abs(v1._factor / v0._factor)))
            op = Op(id0, id1, sub, shift, v.unscaled.qint, v.latency, v.cost)
        self.ops.append(op)


class _TraceManager(metaclass=Singleton):
    def __init__(self):
        self._data = {}
        self._scope = 'global'

    @property
    def data(self) -> dict:
        return self._data.setdefault(self._scope, {})

    @property
    def hwconf(self) -> HWConfig:
        return self.data.setdefault('config', HWConfig(-1, -1, 1e9))

    @property
    def trace(self) -> Trace:
        return self.data.setdefault('record', Trace(self.hwconf))

    @property
    def count(self) -> int:
        v = self.data.setdefault('count', 0)
        self.data['count'] += 1
        return v

    def track(self, v: 'FixedVariable'):
        self.trace.add(v)

    def set_outputs(self, outputs: 'list[FixedVariable]'):
        self.trace.out_idx = [v._id for v in outputs]
        self.trace.out_factors = [float(v._factor) for v in outputs]


class Tracer:
    def __init__(
        self,
        scope_name: str | None = None,
        adder_size=-1,
        carry_size=-1,
        latency_cutoff: float = -1.0,
        persist: bool = False,
    ):
        self._scope = scope_name or str(uuid4())
        self._persist = persist
        self.hw_conf = HWConfig(adder_size, carry_size, latency_cutoff)
        self._trace = None

    def __enter__(self):
        mgr = _TraceManager()
        self._prev_scope = mgr._scope
        mgr._scope = self._scope
        mgr.data['config'] = self.hw_conf
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mgr = _TraceManager()
        self._trace = order_trace(mgr.trace)
        mgr._scope = self._prev_scope
        if not self._persist:
            del mgr._data[self._scope]
        return False

    def track(self, v: 'FixedVariable'):
        return _TraceManager().track(v)

    def set_outputs(self, outputs: 'list[FixedVariable]'):
        return _TraceManager().set_outputs(outputs)

    @property
    def trace(self) -> Trace:
        if self._trace is None:
            return _TraceManager().trace
        return self._trace


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
    def from_const(cls, const: float | Decimal):
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
        return cls(const, const, 2.0**-_high)

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

        f0, f1 = self._factor, other._factor
        if f0 < 0:
            if f1 > 0:
                return other + self
            else:
                return -((-self) + (-other))

        mgr = _TraceManager()
        int0 = QInterval(float(self.low), float(self.high), float(self.step))
        int1 = QInterval(float(other.low), float(other.high), float(other.step))
        adder_size = mgr.hwconf.adder_size
        carry_size = mgr.hwconf.carry_size
        latency_cutoff = mgr.hwconf.latency_cutoff
        dlat, cost = cost_add(int0, int1, 0, False, adder_size, carry_size)
        base_lat = max(self.latency, other.latency)
        latency = dlat + base_lat
        if latency_cutoff > 0 and ceil(latency / latency_cutoff) > ceil(base_lat / latency_cutoff):
            # Crossed the latency cutoff boundry
            assert (
                dlat <= latency_cutoff
            ), f'Latency of an atomic operation {dlat} is larger than the pipelining latency cutoff {latency_cutoff}'
            latency = ceil(base_lat / latency_cutoff) * latency_cutoff + dlat

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


def order_trace(
    trace: Trace,
):
    n = len(trace.ops)
    order = np.argsort([op.latency * n + i for i, op in enumerate(trace.ops)])
    rev_order = np.argsort(order)
    ops: list[Op] = []
    for i in range(n):
        op = trace.ops[int(order[i])]
        id0, id1 = op.id0, op.id1
        id0 = int(rev_order[id0])
        id1 = int(rev_order[id1]) if id1 >= 0 else id1
        assert (id0 <= i and id1 <= i) or id1 == -1, f'Invalid id0={id0}, id1={id1}, i={i}'
        _op = Op(id0, id1, *op[2:])
        ops.append(_op)
    out_idx = [int(rev_order[i]) for i in trace.out_idx]
    return Trace(trace.hwconf, ops, out_idx, trace.out_factors)


def trace_to_solution(trace: Trace):
    assert len(trace.ops) > 0, 'No operations in the record'
    n_in = sum(op.id1 == -1 for op in trace.ops)
    n_out = len(trace.out_idx)
    shape = (n_in, n_out)
    inp_shift = [0] * n_in
    out_shift = [int(log2(abs(sf))) for sf in trace.out_factors]
    out_neg = [sf < 0 for sf in trace.out_factors]
    out_idx = trace.out_idx
    return Solution(
        shape,
        inp_shift,
        out_idx,
        out_shift,
        out_neg,
        trace.ops,
        carry_size=trace.hwconf.carry_size,
        adder_size=trace.hwconf.adder_size,
    )
