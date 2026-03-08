from da4ml._binary import get_lsb_loc, iceil_log2
from da4ml.trace import HWConfig
from da4ml.types import CombLogic, Op, QInterval

from .cse import is_used_in


def overlap_counts(qint0: QInterval, qint1: QInterval, shift1: int, is_sub: bool):
    if is_sub:
        qint1 = QInterval(-qint1.max, -qint1.min, qint1.step)
    r0, r1 = -get_lsb_loc(qint0.step), -get_lsb_loc(qint1.step) - shift1
    _min0, _max0 = min(qint0.min, 0), max(qint0.max, 0)
    _min1, _max1 = min(qint1.min, 0), max(qint1.max, 0)
    b0, b1 = iceil_log2((_max0 - _min0) / qint0.step + 1), iceil_log2((_max1 - _min1) / qint1.step + 1)
    l0, l1 = r0 - b0, r1 - b1
    a, b, c, d = sorted([l0, r0, l1, r1])
    if r0 < l0 or r1 < l0:  # no overlap
        b, c = c, b
    return b - a, c - b, d - c


def cost_lat_add(qint0: QInterval, qint1: QInterval, shift1: int, is_sub: bool, n_add: int, n_accum: int):
    left, overlap, right = overlap_counts(qint0, qint1, shift1, is_sub)
    if overlap <= 0:
        return 0, 0

    cost = (overlap * 2 + left + n_add + 1 - 1) // n_add * 0.5
    lat = (overlap + left + n_accum + 1 - 1) // n_accum
    return cost, lat


def cost_lat_mul(qint0: QInterval, qint1: QInterval, n_add: int, n_accum: int):
    _min0, _max0 = min(qint0.min, 0), max(qint0.max, 0)
    _min1, _max1 = min(qint1.min, 0), max(qint1.max, 0)
    b0, b1 = iceil_log2((_max0 - _min0) / qint0.step), iceil_log2((_max1 - _min1) / qint1.step)
    cost1 = b0 * (b1 + n_add - 1) // n_add
    cost2 = b1 * (b0 + n_add - 1) // n_add
    cost = min(cost1, cost2)
    lat1 = b0 * (b1 + n_accum - 1) // n_accum
    lat2 = b1 * (b0 + n_accum - 1) // n_accum
    lat = min(lat1, lat2)
    return cost, lat


def cost_lat_lut(qint_in: QInterval, qint_out: QInterval, LUT_X: int, LUT_Y: int):
    _min_in, _max_in = min(qint_in.min, 0), max(qint_in.max, 0)
    _min_out, _max_out = min(qint_out.min, 0), max(qint_out.max, 0)
    b_in, b_out = iceil_log2((_max_in - _min_in) / qint_in.step), iceil_log2((_max_out - _min_out) / qint_out.step)
    cost = 2 ** (b_in - LUT_X) * b_out if b_in > LUT_Y else b_in / LUT_Y * 2 ** (LUT_Y - LUT_X)
    lat = 1
    lat = max(1, b_in - LUT_X)
    return cost, lat


def cost_lat_mux(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    return sum(overlap_counts(qint0, qint1, shift1, False)) * 2.0 ** (LUT_Y - LUT_X), 1


def cost_relu(qint: QInterval, LUT_X: int = 6, LUT_Y: int = 5):
    return 0, 0


def cost_lat_bin_bitops(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    x, y, z = overlap_counts(qint0, qint1, shift1, False)
    if y <= 0:
        return 0, 0
    cost = 2 * y / LUT_Y * 2 ** (LUT_X - LUT_Y)
    lat = 1
    return cost, lat


def cost_neg(qint: QInterval, LUT_X: int, LUT_Y: int):
    return 0, 0


def cost_lat_op(ops: list[Op], op: Op, hwconf: HWConfig) -> tuple[float, float]:
    LUT_X, LUT_Y = 6, 5
    n_add, n_carry = hwconf.adder_size % 65535, hwconf.carry_size % 65535
    match op.opcode:
        case -2:  # neg
            return cost_neg(ops[op.id0].qint, LUT_X, LUT_Y)
        case -1:  # READ
            return 0, 0
        case 0 | 1:  # +/-
            op0, op1 = ops[op.id0], ops[op.id1]
            qint0, qint1 = op0.qint, op1.qint
            shift1 = op.data
            is_sub = op.opcode == 1
            c, l = cost_lat_add(qint0, qint1, shift1, is_sub, n_add, n_carry)
            return c, l
        case 2:  # relu(-)
            qint = ops[op.id0].qint
            c, l = cost_relu(qint, LUT_X, LUT_Y)
            return c, l
        case 3:  # WRAP
            c, l = cost_neg(ops[op.id0].qint, LUT_X, LUT_Y)
            return c, l
        case 4:  # cadd
            f = -get_lsb_loc(op.data)
            c = iceil_log2(abs(op.data) + 2**-f) + f
            l = 0
            return c, l
        case 5:  # const
            return 0, 0
        case 6:  # msb_mux
            qint0, qint1 = ops[op.id0].qint, ops[op.id1].qint
            shift = (op.data >> 32) & 0xFFFFFFFF
            shift = shift if shift < 0x80000000 else shift - 0x100000000
            c0, l0 = cost_lat_mux(qint0, qint1, shift, LUT_X, LUT_Y)
            return c0, l0
        case 7:  # mul
            qint0, qint1 = ops[op.id0].qint, ops[op.id1].qint
            c, l = cost_lat_mul(qint0, qint1, n_add, n_carry)
            return c, l
        case 8:  # lut
            qint_in = ops[op.id0].qint
            qint_out = op.qint
            c, l = cost_lat_lut(qint_in, qint_out, LUT_X, LUT_Y)
            return c, l
        case 9:
            c, l = cost_neg(ops[op.id0].qint, LUT_X, LUT_Y)
            return c, l
        case 10:  # bin bitops
            qint0, qint1 = ops[op.id0].qint, ops[op.id1].qint
            shift = ((int(op.data) & 0xFFFFFFFF) + (1 << 31)) % (1 << 32) - (1 << 31)
            c, l = cost_lat_bin_bitops(qint0, qint1, shift, LUT_X, LUT_Y)
            return c, l
        case _:
            raise NotImplementedError(f'Unsupported opcode: {op.opcode}')


def _with_cost_lat(op: Op, cost, lat) -> Op:
    return Op(op.id0, op.id1, op.opcode, op.data, op.qint, lat, cost)


def add_surrogate(comb: CombLogic) -> CombLogic:
    "Add surrogate cost and latency"
    new_ops = []
    hwconf = HWConfig(comb.adder_size, comb.carry_size, -1.0)
    used_in = is_used_in(comb)
    for i, op in enumerate(comb.ops):
        cost, lat = cost_lat_op(new_ops, op, hwconf)
        lat = lat + max(tuple(new_ops[j].latency for j in op.input_ids) + (0,))
        if op.opcode == 5:
            # assert len(used_in[i]) == 1, f'Const op at idx {i} should be used exactly once, but got {len(used_in[i])}'
            for idx in used_in[i]:
                if idx >= 0:
                    lat = comb.ops[idx].latency
                else:
                    lat = 0
        new_ops.append(_with_cost_lat(op, cost, lat))
    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        comb.out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        comb.lookup_tables,
    )
