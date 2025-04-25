from collections.abc import Sequence
from decimal import Decimal
from math import log2
from uuid import UUID

from ..cmvm.types import Op, Solution
from .fixed_veriable import FixedVariable


def _recursive_trace(v: FixedVariable, gathered: dict[UUID, FixedVariable]):
    if v in gathered:
        return
    assert v._from is not None
    for _v in v._from:
        if _v.id not in gathered:
            _recursive_trace(_v, gathered)
    gathered[v.id] = v


def gather_variables(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    gathered = {v.id: v for v in inputs}
    for o in outputs:
        _recursive_trace(o, gathered)

    variables = list(gathered.values())

    N = len(variables)
    _index = sorted(list(range(N)), key=lambda i: variables[i].latency * N + i)
    variables = [variables[i] for i in _index]
    index = {variables[i].id: i for i in range(N)}

    return variables, index


def _trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    variables, index = gather_variables(inputs, outputs)
    ops: list[Op] = []
    inp_uuids = {v.id: i for i, v in enumerate(inputs)}
    for i, v in enumerate(variables):
        match v.opr:
            case 'vadd':
                v0, v1 = v._from
                f0, f1 = v0._factor, v1._factor
                id0, id1 = index[v0.id], index[v1.id]
                sub = int(f1 < 0)
                shift = int(log2(abs(f1 / f0)))
                assert id0 < i and id1 < i, f'{id0} {id1} {i} {v.id}'
                ops.append(Op(id0, id1, sub, shift, v.unscaled.qint, v.latency, v.cost))
            case 'cadd':
                v0 = v._from[0]
                f0 = v0._factor
                id0 = index[v0.id]
                assert v._data is not None, 'cadd must have data'
                qint = v.unscaled.qint
                shift = int(v._data / Decimal(qint.step))
                assert id0 < i, f'{id0} {i} {v.id}'
                ops.append(Op(id0, -4, 0, shift, qint, v.latency, v.cost))
            case 'wrap':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                ops.append(Op(id0, -3, 0, 0, v.unscaled.qint, v.latency, v.cost))
            case 'relu':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                ops.append(Op(id0, -2, int(v._from[0]._factor < 0), 0, v.unscaled.qint, v.latency, v.cost))
            case 'new':
                id0 = inp_uuids[v.id]
                ops.append(Op(id0, -1, 0, 0, v.unscaled.qint, v.latency, v.cost))
            case _:
                raise NotImplementedError(f'Operation "{v.opr}" is not supported in tracing')
    out_index = [index[v.id] for v in outputs]
    return ops, out_index


def trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    ops, out_index = _trace(inputs, outputs)
    shape = len(inputs), len(outputs)
    inp_shift = [0] * shape[0]
    out_sf = [v._factor for v in outputs]
    out_shift = [int(log2(abs(sf))) for sf in out_sf]
    out_neg = [sf < 0 for sf in out_sf]

    return Solution(
        shape,
        inp_shift,
        out_index,
        out_shift,
        out_neg,
        ops,
        outputs[0]._hwconf.carry_size,
        outputs[0]._hwconf.adder_size,
    )
