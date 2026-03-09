from collections.abc import Sequence
from math import log2
from uuid import UUID

import numpy as np

from .._binary import get_lsb_loc
from ..types import CombLogic, Op, QInterval
from .fixed_variable import FixedVariable
from .passes import optimize as _optimize


def _recursive_gather(v: FixedVariable, gathered: dict[UUID, FixedVariable]):
    if v.id in gathered:
        return
    assert v._from is not None
    for _v in v._from:
        _recursive_gather(_v, gathered)
    gathered[v.id] = v


def gather_variables(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    input_ids = {v.id for v in inputs}
    gathered = {v.id: v for v in inputs}
    for o in outputs:
        _recursive_gather(o, gathered)
    variables = list(gathered.values())

    N = len(variables)
    _index = sorted(list(range(N)), key=lambda i: variables[i].latency * N + i)
    variables = [variables[i] for i in _index]

    # Remove variables with 0 refcount
    refcount = {v.id: 0 for v in variables}
    for v in variables:
        if v.id in input_ids:
            continue
        for _v in v._from:
            refcount[_v.id] += 1
    for v in outputs:
        refcount[v.id] += 1

    variables = [v for v in variables if refcount[v.id] > 0 or v.id in input_ids]

    return variables


def needs_negative(variables: Sequence[FixedVariable], outputs: Sequence[FixedVariable]) -> set[UUID]:
    needs_neg = set()
    for v in variables:
        if v.opr == 'vadd' or v.opr == 'vmul':
            continue
        for _v in v._from:
            if _v._factor < 0:
                needs_neg.add(_v.id)
    for v in outputs:
        if v._factor < 0:
            needs_neg.add(v.id)
    return needs_neg


def _comb_trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    variables = gather_variables(inputs, outputs)
    ops: list[Op] = []
    inp_uuids = {v.id: i for i, v in enumerate(inputs)}
    table_id_map: dict[str, int] = {}
    lookup_tables = []

    index: dict[UUID, int] = {}
    needs_neg = needs_negative(variables, outputs)
    ii = -1
    for v in variables:
        ii += 1
        index[v.id] = ii
        if v.id in inp_uuids and v.opr != 'const':
            id0 = inp_uuids[v.id]
            ops.append(Op(id0, -1, -1, 0, v.unscaled.qint, v.latency, 0.0))
            if v.id in needs_neg:
                op_neg = Op(ii, -1, -2, 0, (-v.unscaled).qint, v.latency, 0.0)
                ops.append(op_neg)
                ii += 1
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
                assert id0 < ii and id1 < ii, f'{id0} {id1} {ii} {v.id}'
                op = Op(id0, id1, sub, data, v.unscaled.qint, v.latency, v.cost)
            case 'cadd':
                v0 = v._from[0]
                f0 = v0._factor
                id0 = index[v0.id]
                assert v._data is not None, 'cadd must have data'
                qint = v.unscaled.qint
                data = v._data
                assert id0 < ii, f'{id0} {ii} {v.id}'
                op = Op(id0, -1, 4, data, qint, v.latency, v.cost)
            case 'wrap':
                v0 = v._from[0]
                id0 = index[v0.id] + (v0._factor < 0)
                assert id0 < ii, f'{id0} {ii} {v.id}'
                opcode = 3
                op = Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'relu':
                v0 = v._from[0]
                id0 = index[v0.id] + (v0._factor < 0)
                assert id0 < ii, f'{id0} {ii} {v.id}'
                opcode = 2
                op = Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'const':
                qint = v.unscaled.qint
                assert qint.min == qint.max, f'const {v.id} {qint.min} {qint.max}'
                f = -get_lsb_loc(qint.min)
                step = 2.0**-f
                qint = QInterval(qint.min, qint.min, step)
                data = qint.min / step
                op = Op(-1, -1, 5, int(data), qint, v.latency, v.cost)
            case 'msb_mux':
                qint = v.unscaled.qint
                key, in0, in1 = v._from
                opcode = 6
                idk, id0, id1 = index[key.id], index[in0.id], index[in1.id] + (in1._factor < 0)
                f0, f1 = in0._factor, in1._factor
                shift = int(log2(abs(f1 / f0)))
                data = idk + (shift << 32)
                assert idk < ii and id0 < ii and id1 < ii, f'{idk} {id0} {id1} {ii} {v.id}'
                assert key._factor > 0, f'Cannot mux on v{key.id} with negative factor {key._factor}'
                op = Op(id0, id1, opcode, data, qint, v.latency, v.cost)
            case 'vmul':
                v0, v1 = v._from
                opcode = 7
                id0, id1 = index[v0.id], index[v1.id]
                assert id0 < ii and id1 < ii, f'{id0} {id1} {ii} {v.id}'
                op = Op(id0, id1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'lookup':
                opcode = 8
                v0 = v._from[0]
                id0 = index[v0.id]
                data = v._data
                assert id0 < ii, f'{id0} {ii} {v.id}'
                assert v._table is not None, 'lookup must have a table'
                tb_bash = v._table.spec.hash
                if tb_bash in table_id_map:
                    data = table_id_map[tb_bash]
                else:
                    data = len(table_id_map)
                    table_id_map[tb_bash] = data
                    lookup_tables.append(v._table)
                op = Op(id0, -1, opcode, data, v.unscaled.qint, v.latency, v.cost)
            case 'bit_unary':
                v0 = v._from[0]
                id0 = index[v0.id] + (v._factor < 0)
                assert id0 < ii, f'{id0} {ii} {v.id}'
                assert v._data is not None, 'bit_unary must have data'
                opcode = 9
                op = Op(id0, -1, opcode, int(v._data), v.unscaled.qint, v.latency, v.cost)
            case 'bit_binary':
                v0, v1 = v._from
                id0, id1 = index[v0.id], index[v1.id]
                f0, f1 = v0._factor, v1._factor
                id0, id1 = id0 + (f0 < 0), id1 + (f1 < 0)
                assert id0 < ii and id1 < ii, f'{id0} {id1} {ii} {v.id}'
                assert v._data is not None, 'bit_binary must have data'
                # data: {subopcode[63:56], pad0, v1_neg[33], v0_neg[32], shift[31:0]}
                _data = int(log2(abs(f1 / f0))) & 0xFFFFFFFF
                _data += int(v._data) << 56
                op = Op(id0, id1, 10, _data, v.unscaled.qint, v.latency, v.cost)
            case _:
                raise NotImplementedError(f'Operation "{v.opr}" is not supported in tracing')

        ops.append(op)
        if v.id in needs_neg:
            op_neg = Op(ii, -1, -2, 0, (-v.unscaled).qint, v.latency, 0)
            ops.append(op_neg)
            ii += 1

    out_index = [index[v.id] for v in outputs]
    lookup_tables = None if not lookup_tables else tuple(lookup_tables)
    return ops, out_index, lookup_tables


def comb_trace(inputs, outputs, optimize=True, keep_dead_inputs: bool = False) -> CombLogic:
    if isinstance(inputs, FixedVariable):
        inputs = [inputs]
    if isinstance(outputs, FixedVariable):
        outputs = [outputs]

    inputs, outputs = list(np.ravel(inputs)), list(np.ravel(outputs))  # type: ignore

    assert all(inp._factor > 0 for inp in inputs), 'Input variables must have positive scaling factor'

    if any(not isinstance(v, FixedVariable) for v in outputs):
        hwconf = inputs[0].hwconf
        outputs = list(outputs)
        for i, v in enumerate(outputs):
            if not isinstance(v, FixedVariable):
                outputs[i] = FixedVariable.from_const(v, hwconf, 1)

    ops, out_index, lookup_tables = _comb_trace(inputs, outputs)
    shape = len(inputs), len(outputs)
    inp_shifts = [0] * shape[0]
    out_sf = [v._factor for v in outputs]
    out_shift = [int(log2(abs(sf))) for sf in out_sf]
    out_neg = [sf < 0 for sf in out_sf]

    comb = CombLogic(
        shape,
        inp_shifts,
        out_index,
        out_shift,
        out_neg,
        ops,
        outputs[0].hwconf.carry_size,
        outputs[0].hwconf.adder_size,
        lookup_tables,
    )

    return _optimize(comb, keep_dead_inputs) if optimize else comb
