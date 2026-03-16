from math import ceil

import numpy as np

from ..._binary import get_lsb_loc, iceil_log2
from ...trace import HWConfig
from ...trace.fixed_variable import LookupTable
from ...types import CombLogic, Op, QInterval, minimal_kif


def overlap_counts(qint0: QInterval, qint1: QInterval, shift1: int, is_sub: bool):
    r0, r1 = -get_lsb_loc(qint0.step), -get_lsb_loc(qint1.step) - shift1
    b0, b1 = sum(minimal_kif(qint0)), sum(minimal_kif(qint1))
    l0, l1 = r0 - b0, r1 - b1
    a, b, c, d = sorted([l0, r0, l1, r1])
    if r0 < l0 or r1 < l0:  # no overlap
        b, c = c, b
    if is_sub:
        b = a
    return b - a, c - b, d - c


def cost_lat_add(qint0: QInterval, qint1: QInterval, shift1: int, is_sub: bool, n_add: int, n_accum: int):
    left, overlap, right = overlap_counts(qint0, qint1, shift1, is_sub)
    if overlap <= 0:  # bit concat
        return 0, 0

    bw_add = left + overlap + right
    cost = (max(bw_add - 1, 1) + n_add - 1) // n_add
    lat = (cost + n_accum - 1) // n_accum if n_accum > 0 else 1
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


def _count_luts_rec(bit_nd: np.ndarray, LUT_X: int = 6) -> float:
    """Count LUT6s for one output bit. Greedy: picks axis with most identical halves."""
    d = bit_nd.ndim
    if d <= LUT_X:
        return int(np.unique(bit_nd).size > 1)

    flat_size = 1 << (d - 1)
    halves = np.stack([np.moveaxis(bit_nd, ax, 0).reshape(2, flat_size) for ax in range(d)])  # (d, 2, flat_size)
    matches = np.sum(halves[:, 0] == halves[:, 1], axis=1)
    best_ax = int(np.argmax(matches))

    if matches[best_ax] == flat_size:
        left = np.take(bit_nd, 0, axis=best_ax)
        return _count_luts_rec(left, LUT_X)

    left = np.take(bit_nd, 0, axis=best_ax)
    right = np.take(bit_nd, 1, axis=best_ax)
    return _count_luts_rec(left, LUT_X) + _count_luts_rec(right, LUT_X)


def _count_luts(bit_nd: np.ndarray, LUT_X: int = 6) -> float:
    """Count LUT6s needed for one output bit.

    Tries all axes at the top level (exhaustive), greedy below.
    """
    d = bit_nd.ndim
    if d <= LUT_X:
        return 0.0 if len(np.unique(bit_nd)) == 1 else 1.0

    best_cost = float('inf')
    for ax in range(d):
        left = np.take(bit_nd, 0, axis=ax)
        right = np.take(bit_nd, 1, axis=ax)
        if np.array_equal(left, right):
            c = _count_luts_rec(left, LUT_X)
        else:
            c = _count_luts_rec(left, LUT_X) + _count_luts_rec(right, LUT_X)
        best_cost = min(best_cost, c)
    return best_cost


def cost_lat_lut(qint_in: QInterval, table: LookupTable, LUT_X: int, LUT_Y: int):
    data = table.padded_table(qint_in)
    int_data = np.where(np.isnan(data), 0, data).astype(int)  # fill DCs with 0
    out_bw = sum(table.spec.out_kif)

    n = max(int(np.ceil(np.log2(max(len(data), 2)))), 1)

    # Pad to 2^n
    full_size = 2**n
    if len(int_data) < full_size:
        int_data = np.pad(int_data, (0, full_size - len(int_data)))

    total_cost = 0.0
    for b in range(out_bw):
        bit_vals = ((int_data >> b) & 1).astype(np.int8)

        if np.all(bit_vals == bit_vals[0]):
            continue

        bit_nd = bit_vals.reshape((2,) * n)
        total_cost += _count_luts(bit_nd, LUT_X)

    lat = max(n - LUT_X, 1)
    return ceil(total_cost), lat


def cost_lat_mux(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    return sum(overlap_counts(qint0, qint1, shift1, False)) * 2.0 ** (LUT_Y - LUT_X), 1


def cost_relu(qint: QInterval, LUT_X: int = 6, LUT_Y: int = 5):
    # LUT6_2 packs 2 ReLU bits (shared sign), but ~1/3 of bits can't share
    # due to routing constraints. Effective: 0.67 LUT/bit.
    return sum(minimal_kif(qint)) * 2 / 3, 0


def cost_lat_bin_bitops(qint0: QInterval, qint1: QInterval, shift1: int, LUT_X: int, LUT_Y: int):
    x, y, z = overlap_counts(qint0, qint1, shift1, False)
    if y <= 0:
        return 0, 0
    cost = 2 * y / LUT_Y * 2 ** (LUT_X - LUT_Y)
    lat = 1
    return cost, lat


def cost_neg(qint: QInterval, LUT_X: int, LUT_Y: int):  # noqa: ARG001
    # NEG = 0 - x: one constant input (0). Synthesis absorbs the carry chain
    # into downstream ops (79% into RELU via LUT6_2 sharing, 21% into WRAP).
    return 0, 0


def cost_lat_op(ops: list[Op], op: Op, hwconf: HWConfig, lut: tuple[LookupTable, ...] | None) -> tuple[float, float]:
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
            qint_in = ops[op.id0].qint
            if qint_in.min >= 0:
                return 0, 0  # no-op for non-negative
            c, l = cost_relu(qint_in, LUT_X, LUT_Y)
            return c, l
        case 3:  # WRAP — pure routing (bit-select / sign-extend)
            return 0, 0
        case 4:  # cadd — absorbed by synthesis
            return 0, 0
        case 5:  # const
            return 0, 0
        case 6:  # msb_mux
            out_bw = sum(minimal_kif(op.qint))
            c0 = out_bw * 2.0 ** (LUT_Y - LUT_X)  # 0.5 LUT/bit
            return c0, 1
        case 7:  # mul
            qint0, qint1 = ops[op.id0].qint, ops[op.id1].qint
            c, l = cost_lat_mul(qint0, qint1, n_add, n_carry)
            return c, l
        case 8:  # lut
            qint_in = ops[op.id0].qint
            # qint_out = op.qint
            assert lut is not None
            c, l = cost_lat_lut(qint_in, lut[op.data], LUT_X, LUT_Y)
            return c, l
        case 9:  # unary bitops — absorbed by synthesis
            return 0, 0
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
    for op in comb.ops:
        cost, lat = cost_lat_op(new_ops, op, hwconf, comb.lookup_tables)
        lat = lat + max(tuple(new_ops[j].latency for j in op.input_ids) + (0,))
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
