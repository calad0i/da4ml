import typing
from collections.abc import Sequence
from math import log2
from typing import NamedTuple
from uuid import uuid4

import numpy as np

from ..cmvm.types import Op, Solution

if typing.TYPE_CHECKING:
    from .fixed_veriable import FixedVariable


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
