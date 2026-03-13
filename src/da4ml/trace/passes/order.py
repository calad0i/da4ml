import numpy as np

from ...types import CombLogic, Op
from .dce import _index_remap


def topo_ordering(ops: list[Op]) -> np.ndarray:
    order = np.zeros((len(ops), 7), dtype=np.float32)
    for i, op in enumerate(ops):
        order[i, 6] = op.latency
        order[i, 5] = max((order[idx, 5] for idx in op.input_ids), default=-1) + 1
        order[i, 4] = -op.opcode
        order[i, 1:4] = op.qint
        order[i, 0] = i
    return np.lexsort(order.T)


def order_ops(comb: CombLogic) -> CombLogic:
    new_ops = topo_ordering(comb.ops)
    idx_map = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(new_ops)}
    remapped_ops = [_index_remap(comb.ops[old_idx], idx_map) for old_idx in new_ops]
    new_out_idxs = [idx_map[idx] if idx >= 0 else -1 for idx in comb.out_idxs]
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        new_out_idxs,
        comb.out_shifts,
        comb.out_negs,
        remapped_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
