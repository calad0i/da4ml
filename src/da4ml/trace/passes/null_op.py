from ...types import CombLogic
from .cse import _index_remap, is_used_in


def null_quant_elimin(comb: CombLogic) -> CombLogic:
    _map: dict[int, int] = {}
    for i, op in enumerate(comb.ops):
        if op.opcode in (2, 3):  # null quantizer/relu
            src = comb.ops[_map.get(op.id0, op.id0)]
            if src.qint != op.qint:
                continue
            _map[i] = _map.get(op.id0, op.id0)
        elif op.opcode == 9 and op.data == 0:
            op_from = comb.ops[_map.get(op.id0, op.id0)]
            if op_from.opcode == 9 and op_from.data == 0:  # double NOT
                _map[i] = _map.get(op_from.id0, op_from.id0)
                continue

    if not _map:
        return comb

    new_ops = comb.ops.copy()
    used_in = is_used_in(comb)
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
