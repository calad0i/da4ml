from collections.abc import Sequence
from decimal import Decimal
from itertools import chain
from math import log2
from typing import overload
from uuid import UUID

import numpy as np

from ..cmvm.types import Op, QInterval, Solution
from .fixed_variable import FixedVariable, _const_f
from .fixed_variable_array import FixedVariableArray


def _recursive_gather(v: FixedVariable, gathered: dict[UUID, FixedVariable]):
    if v.id in gathered:
        return
    assert v._from is not None
    for _v in v._from:
        if _v.id not in gathered:
            _recursive_gather(_v, gathered)
    gathered[v.id] = v


def gather_variables(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    gathered = {v.id: v for v in inputs}
    for o in outputs:
        _recursive_gather(o, gathered)

    variables = list(gathered.values())

    N = len(variables)
    _index = sorted(list(range(N)), key=lambda i: variables[i].latency * N + i)
    variables = [variables[i] for i in _index]
    index = {variables[i].id: i for i in range(N)}

    return variables, index


def _comb_trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    variables, index = gather_variables(inputs, outputs)
    ops: list[Op] = []
    inp_uuids = {v.id: i for i, v in enumerate(inputs)}
    for i, v in enumerate(variables):
        if v.id in inp_uuids and v.opr != 'const':
            id0 = inp_uuids[v.id]
            ops.append(Op(id0, -1, -1, 0, v.unscaled.qint, v.latency, v.cost))
            continue
        if v.opr == 'new':
            raise NotImplementedError('Operation "new" is only expected in the input list')
        match v.opr:
            case 'vadd':
                v0, v1 = v._from
                f0, f1 = v0._factor, v1._factor
                id0, id1 = index[v0.id], index[v1.id]
                sub = int(f1 < 0)
                data = int(log2(abs(f1 / f0)))
                assert id0 < i and id1 < i, f'{id0} {id1} {i} {v.id}'
                ops.append(Op(id0, id1, sub, data, v.unscaled.qint, v.latency, v.cost))
            case 'cadd':
                v0 = v._from[0]
                f0 = v0._factor
                id0 = index[v0.id]
                assert v._data is not None, 'cadd must have data'
                qint = v.unscaled.qint
                data = int(v._data / Decimal(qint.step))
                assert id0 < i, f'{id0} {i} {v.id}'
                ops.append(Op(id0, -1, 4, data, qint, v.latency, v.cost))
            case 'wrap':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                opcode = -3 if v._from[0]._factor < 0 else 3
                ops.append(Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost))
            case 'relu':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                opcode = -2 if v._from[0]._factor < 0 else 2
                ops.append(Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost))
            case 'const':
                qint = v.unscaled.qint
                assert qint.min == qint.max, f'const {v.id} {qint.min} {qint.max}'
                f = _const_f(qint.min)
                step = 2.0**-f
                qint = QInterval(qint.min, qint.min, step)
                data = qint.min / step
                ops.append(Op(-1, -1, 5, int(data), qint, v.latency, v.cost))
            case 'msb_mux':
                qint = v.unscaled.qint
                key, in0, in1 = v._from
                opcode = 6 if in1._factor > 0 else -6
                idk, id0, id1 = index[key.id], index[in0.id], index[in1.id]
                f0, f1 = in0._factor, in1._factor
                shift = int(log2(abs(f1 / f0)))
                data = idk + (shift << 32)
                assert idk < i and id0 < i and id1 < i
                assert key._factor > 0, f'Cannot mux on v{key.id} with negative factor {key._factor}'
                op = Op(id0, id1, opcode, data, qint, v.latency, v.cost)
                ops.append(op)

            case _:
                raise NotImplementedError(f'Operation "{v.opr}" is not supported in tracing')
    out_index = [index[v.id] for v in outputs]
    return ops, out_index


@overload
def comb_trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]) -> Solution: ...


@overload
def comb_trace(inputs: FixedVariableArray, outputs: FixedVariableArray) -> Solution: ...


def comb_trace(inputs, outputs):
    inputs, outputs = list(np.ravel(inputs)), list(np.ravel(outputs))

    if any(not isinstance(v, FixedVariable) for v in outputs):
        hwconf = inputs[0].hwconf
        latency = max(v.latency for v in chain(inputs, outputs) if isinstance(v, FixedVariable))
        outputs = list(outputs)
        for i, v in enumerate(outputs):
            if not isinstance(v, FixedVariable):
                outputs[i] = FixedVariable.from_const(v, hwconf, latency, 1)

    ops, out_index = _comb_trace(inputs, outputs)
    shape = len(inputs), len(outputs)
    inp_shift = [0] * shape[0]
    out_sf = [v._factor for v in outputs]
    out_shift = [int(log2(abs(sf))) for sf in out_sf]
    out_neg = [sf < 0 for sf in out_sf]

    sol = Solution(
        shape,
        inp_shift,
        out_index,
        out_shift,
        out_neg,
        ops,
        outputs[0].hwconf.carry_size,
        outputs[0].hwconf.adder_size,
    )

    ref_count = sol.ref_count

    for i in range(len(ops)):
        if ref_count[i] == 0:
            op = ops[i]
            sol.ops[i] = Op(-1, -1, op[2], 0, QInterval(0, 0, 1), op[5], op[6])

    return sol
