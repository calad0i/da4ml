import numpy as np

from ...types import CombLogic
from .dce import _index_remap


def order_by_latency(comb: CombLogic) -> CombLogic:
    new_ops = np.argsort([op.latency for op in comb.ops], stable=True)
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
