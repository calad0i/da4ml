from ..._binary import get_lsb_loc
from ...types import CombLogic, Op, QInterval


def canonicalize_outputs(comb: CombLogic) -> CombLogic:
    ops = comb.ops.copy()
    out_idxs = comb.out_idxs.copy()
    out_shifts = comb.out_shifts.copy()
    out_negs = comb.out_negs.copy()
    for i in range(comb.shape[1]):
        idx, s, n = comb.out_idxs[i], comb.out_shifts[i], comb.out_negs[i]
        if idx < 0:
            out_idxs[i] = len(ops)
            ops.append(Op(-1, -1, 5, 0, QInterval(0, 0, 2**127), 0, 0))
            out_shifts[i] = 0
            out_negs[i] = False
            continue

        op = comb.ops[idx]
        if ops[idx].opcode == 5:
            if s or n:
                val = op.data * op.qint.step * 2**s * (-1 if n else 1)
                out_idxs[i] = len(ops)
                step = 2 ** get_lsb_loc(val)
                op = Op(-1, -1, 5, int(val / step), QInterval(val, val, step), 0, 0)
                ops.append(op)
                out_shifts[i] = 0
                out_negs[i] = False
            continue

        if n:
            out_idxs[i] = len(ops)
            qint = ops[idx].qint
            ops.append(Op(idx, -1, -2, 0, QInterval(-qint.max, -qint.min, qint.step), op.latency, 0))
            out_negs[i] = False

    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        out_idxs,
        out_shifts,
        out_negs,
        ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
