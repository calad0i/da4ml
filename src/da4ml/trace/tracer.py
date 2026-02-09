from collections.abc import Sequence
from decimal import Decimal
from math import log2
from uuid import UUID

import numpy as np

from ..cmvm.types import CombLogic, Op, QInterval
from .fixed_variable import FixedVariable, _const_f, table_context


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
        if v in inputs:
            continue
        for _v in v._from:
            refcount[_v.id] += 1
    for v in outputs:
        refcount[v.id] += 1

    variables = [v for v in variables if refcount[v.id] > 0 or v.id in input_ids]
    index = {variables[i].id: i for i in range(len(variables))}

    return variables, index


def _comb_trace(inputs: Sequence[FixedVariable], outputs: Sequence[FixedVariable]):
    variables, index = gather_variables(inputs, outputs)
    ops: list[Op] = []
    inp_uuids = {v.id: i for i, v in enumerate(inputs)}
    lookup_tables = []

    table_map: dict[int, int] = {}
    for v in variables:
        if not v.opr == 'lookup':
            continue
        assert v._data is not None
        idx = int(v._data)
        if idx in table_map:
            continue
        table_map[idx] = len(lookup_tables)
        lookup_tables.append(table_context.get_table_from_index(idx))

    for i, v in enumerate(variables):
        if v.id in inp_uuids and v.opr != 'const':
            id0 = inp_uuids[v.id]
            ops.append(Op(id0, -1, -1, 0, v.unscaled.qint, v.latency, 0.0))
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
                op = Op(id0, id1, sub, data, v.unscaled.qint, v.latency, v.cost)
            case 'cadd':
                v0 = v._from[0]
                f0 = v0._factor
                id0 = index[v0.id]
                assert v._data is not None, 'cadd must have data'
                qint = v.unscaled.qint
                data = int(v._data / Decimal(qint.step))
                assert id0 < i, f'{id0} {i} {v.id}'
                op = Op(id0, -1, 4, data, qint, v.latency, v.cost)
            case 'wrap':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                opcode = -3 if v._from[0]._factor < 0 else 3
                op = Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'relu':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                opcode = -2 if v._from[0]._factor < 0 else 2
                op = Op(id0, -1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'const':
                qint = v.unscaled.qint
                assert qint.min == qint.max, f'const {v.id} {qint.min} {qint.max}'
                f = _const_f(qint.min)
                step = 2.0**-f
                qint = QInterval(qint.min, qint.min, step)
                data = qint.min / step
                op = Op(-1, -1, 5, int(data), qint, v.latency, v.cost)
            case 'msb_mux':
                qint = v.unscaled.qint
                key, in0, in1 = v._from
                opcode = 6 if in1._factor > 0 else -6
                idk, id0, id1 = index[key.id], index[in0.id], index[in1.id]
                f0, f1 = in0._factor, in1._factor
                shift = int(log2(abs(f1 / f0)))
                data = idk + (shift << 32)
                assert idk < i and id0 < i and id1 < i, f'{idk} {id0} {id1} {i} {v.id}'
                assert key._factor > 0, f'Cannot mux on v{key.id} with negative factor {key._factor}'
                op = Op(id0, id1, opcode, data, qint, v.latency, v.cost)
            case 'vmul':
                v0, v1 = v._from
                opcode = 7
                id0, id1 = index[v0.id], index[v1.id]
                assert id0 < i and id1 < i, f'{id0} {id1} {i} {v.id}'
                op = Op(id0, id1, opcode, 0, v.unscaled.qint, v.latency, v.cost)
            case 'lookup':
                opcode = 8
                v0 = v._from[0]
                id0 = index[v0.id]
                data = v._data
                assert data is not None, 'lookup must have data'
                assert id0 < i, f'{id0} {i} {v.id}'
                op = Op(id0, -1, opcode, table_map[int(data)], v.unscaled.qint, v.latency, v.cost)
            case 'bit_unary':
                v0 = v._from[0]
                id0 = index[v0.id]
                assert id0 < i, f'{id0} {i} {v.id}'
                assert v._data is not None, 'bit_unary must have data'
                opcode = 9 if v._factor > 0 else -9
                op = Op(id0, -1, opcode, int(v._data), v.unscaled.qint, v.latency, v.cost)
            case 'bit_binary':
                id0, id1 = index[v._from[0].id], index[v._from[1].id]
                assert id0 < i and id1 < i, f'{id0} {id1} {i} {v.id}'
                assert v._data is not None, 'bit_binary must have data'
                v0, v1 = v._from
                f0, f1 = v0._factor, v1._factor
                # data: {subopcode[63:56], pad0, v1_neg[33], v0_neg[32], shift[31:0]}
                _data = int(log2(abs(f1 / f0))) & 0xFFFFFFFF
                _data += (int(v._data) << 56) + (int(f0 < 0) << 32) + (int(f1 < 0) << 33)
                op = Op(id0, id1, 10, _data, v.unscaled.qint, v.latency, v.cost)
            case _:
                raise NotImplementedError(f'Operation "{v.opr}" is not supported in tracing')

        ops.append(op)
    out_index = [index[v.id] for v in outputs]
    lookup_tables = None if not lookup_tables else tuple(lookup_tables)
    return ops, out_index, lookup_tables


def _index_remap(op: Op, idx_map: dict[int, int]) -> Op:
    if op.opcode == -1:
        return op
    id0 = op.id0
    id1 = op.id1
    id0 = idx_map[id0] if id0 >= 0 else id0
    id1 = idx_map[id1] if id1 >= 0 else id1
    if abs(op.opcode) == 6:  # msb_mux
        id_c = op.data & 0xFFFFFFFF
        shift = (op.data >> 32) & 0xFFFFFFFF
        id_c = idx_map[id_c]
        data = id_c + (shift << 32)
    else:
        data = op.data
    return Op(id0, id1, op.opcode, data, op.qint, op.latency, op.cost)


def compactify_comb(comb: CombLogic, keep_dead_inputs: bool = False) -> CombLogic:
    no_ref = comb.ref_count == 0
    if keep_dead_inputs:
        no_ref &= np.array([op.opcode != -1 for op in comb.ops], dtype=np.bool_)
    if not np.any(no_ref):
        return comb

    idx_map: dict[int, int] = {}
    new_ops: list[Op] = []

    cnt = 0
    for i, op in enumerate(comb.ops):
        if no_ref[i]:
            continue
        new_ops.append(_index_remap(op, idx_map))
        idx_map[i] = cnt
        cnt += 1
    out_idxs = [idx_map[i] for i in comb.out_idxs]

    comb = CombLogic(
        comb.shape,
        comb.inp_shifts,
        out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
    return compactify_comb(comb, keep_dead_inputs)


def comb_trace(inputs, outputs, keep_dead_inputs: bool = False) -> CombLogic:
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

    return compactify_comb(comb, keep_dead_inputs)
