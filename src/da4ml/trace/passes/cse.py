from ...types import CombLogic, Op, QInterval
from .dce import _index_remap


def is_used_in(comb: CombLogic) -> dict[int, set[int]]:
    used_in = {i: set() for i in range(len(comb.ops))}
    for i, op in enumerate(comb.ops):
        if op.opcode == -1:
            continue
        for j in op.input_ids:
            used_in[j].add(i)
    for i, j in enumerate(comb.out_idxs):
        if j < 0:
            continue
        used_in[j].add(-1 - i)
    return used_in


def to_key(op: Op):
    if op.opcode in (3, 4, 5):
        return tuple(op)
    else:
        return op[:4]


def common_subexpr_elimin(comb: CombLogic) -> CombLogic:
    if len(set(comb.ops)) == len(comb.ops):
        return comb
    new_ops = comb.ops.copy()
    used_in = is_used_in(comb)
    new_out_idxs = comb.out_idxs.copy()
    seen: dict[tuple, int] = {}
    for i, op in enumerate(new_ops):
        k = to_key(op)
        if k not in seen:
            seen[k] = i
            continue
        j = seen[k]
        qint0, qint1 = new_ops[i].qint, new_ops[j].qint
        qint = QInterval(max(qint0.min, qint1.min), min(qint0.max, qint1.max), max(qint0.step, qint1.step))
        op = Op(op.id0, op.id1, op.opcode, op.data, qint, op.latency, op.cost)
        new_ops[j] = op
        redirect_all(used_in, new_ops, new_out_idxs, i, j)

    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        new_out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )


def redirect_all(used_in, new_ops, new_out_idxs, i_from, i_to):
    _map = {i_from: i_to}
    for j in used_in[i_from]:
        if j >= 0:
            new_ops[j] = _index_remap(new_ops[j], _map)
        else:
            new_out_idxs[-1 - j] = i_to
