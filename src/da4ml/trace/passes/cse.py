from ...types import CombLogic, Op
from .dce import _index_remap


def is_used_in(comb: CombLogic) -> dict[int, set[int]]:
    used_in = {i: set() for i in range(len(comb.ops))}
    for i, op in enumerate(comb.ops):
        if op.opcode == -1:
            continue
        if op.id0 >= 0:
            used_in[op.id0].add(i)
        if op.id1 >= 0:
            used_in[op.id1].add(i)
        if abs(op.opcode) == 6:  # msb_mux
            id_c = op.data & 0xFFFFFFFF
            used_in[id_c].add(i)
    for i, j in enumerate(comb.out_idxs):
        if j < 0:
            continue
        used_in[j].add(-1 - i)
    return used_in


def common_subexpr_elimin(comb: CombLogic) -> CombLogic:
    if len(set(comb.ops)) == len(comb.ops):
        return comb
    new_ops = comb.ops.copy()
    used_in = is_used_in(comb)
    new_out_idxs = comb.out_idxs.copy()
    seen: dict[Op, int] = {}
    for i, op in enumerate(new_ops):
        if op not in seen:
            seen[op] = i
            continue
        if op.opcode == 5:
            continue

        idx = seen[op]
        redirect_all(used_in, new_ops, new_out_idxs, i, idx)

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
