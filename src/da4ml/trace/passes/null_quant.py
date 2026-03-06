from ...types import CombLogic
from .cse import _index_remap, gen_used_in


def null_quant_elimin(comb: CombLogic) -> CombLogic:
    _map: dict[int, int] = {}
    for i, op in enumerate(comb.ops):
        if op.opcode not in (2, 3):
            continue
        src = comb.ops[op.id0]
        if src.qint != op.qint:
            continue
        _map[i] = op.id0
    if not _map:
        return comb

    new_ops = comb.ops.copy()
    used_in = gen_used_in(comb)
    new_out_idxs = comb.out_idxs.copy()
    for i in _map.keys():
        depends = used_in[i]
        for j in depends:
            if j >= 0:
                new_ops[j] = _index_remap(new_ops[j], _map)
            else:
                new_out_idxs[-1 - j] = _map[i]

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
