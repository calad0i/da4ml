import numpy as np

from ...types import CombLogic, Op


def _index_remap(op: Op, idx_map: dict[int, int]) -> Op:
    if op.opcode == -1:
        return op
    id0 = op.id0
    id1 = op.id1
    id0 = idx_map.get(id0, id0)
    id1 = idx_map.get(id1, id1)
    if abs(op.opcode) == 6:  # msb_mux
        id_c = op.data & 0xFFFFFFFF
        shift = (op.data >> 32) & 0xFFFFFFFF
        id_c = idx_map.get(id_c, id_c)
        data = id_c + (shift << 32)
    else:
        data = op.data
    return Op(id0, id1, op.opcode, data, op.qint, op.latency, op.cost)


def dead_code_elimin(comb: CombLogic, keep_dead_inputs=False) -> CombLogic:
    "dead code elimination"
    dead = np.ones(len(comb.ops), dtype=bool)
    for idx in comb.out_idxs:
        if idx < 0:
            continue
        dead[idx] = False

    for i in range(len(comb.ops) - 1, -1, -1):
        op = comb.ops[i]
        if dead[i] and not (keep_dead_inputs and op.opcode == -1):
            continue
        if op.opcode != -1:
            if op.id0 >= 0:
                dead[op.id0] = False
            if op.id1 >= 0:
                dead[op.id1] = False
        if abs(op.opcode) == 6:  # msb_mux
            id_c = op.data & 0xFFFFFFFF
            dead[id_c] = False

    new_idxs = np.cumsum(~dead) - 1
    idx_map = {int(i): int(new_idxs[i]) for i in range(len(comb.ops))}
    new_ops = [_index_remap(op, idx_map) for i, op in enumerate(comb.ops) if not dead[i]]
    new_out_idxs = [idx_map[idx] if idx >= 0 else -1 for idx in comb.out_idxs]
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
