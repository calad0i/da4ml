from math import ceil, log2

import numpy as np
from numba import jit

from ..types import DAState, Op, Pair, QInterval
from ..util import csd_decompose


@jit
def qint_add(qint0: QInterval, qint1: QInterval, shift: int, sub0=False, sub1=False) -> QInterval:
    min0, max0, step0 = qint0
    min1, max1, step1 = qint1
    if sub0:
        min0, max0 = -max0, -min0
    if sub1:
        min1, max1 = -max1, -min1

    s = 2.0**shift
    min1, max1, step1 = min1 * s, max1 * s, step1 * s

    return QInterval(min0 + min1, max0 + max1, min(step0, step1))


@jit
def cost_add(
    qint0: QInterval, qint1: QInterval, shift: int, sub: bool = False, adder_size: int = -1, carry_size: int = -1
) -> tuple[float, float]:
    """Calculate the latency and cost of an addition operation.

    Parameters
    ----------
    qint1 : QInterval
        The first QInterval.
    qint2 : QInterval
        The second QInterval.
    sub : bool
        If True, the operation is a subtraction (a - b) instead of an addition (a + b).
    adder_size : int
        The atomic size of the adder.
    carry_size : int
        The size of the look-ahead carry.

    Returns
    -------
    tuple[float, float]
        The latency and cost of the addition operation.
    """
    if adder_size < 0 and carry_size < 0:
        return 1.0, 1.0
    if adder_size < 0:
        adder_size = 65535
    if carry_size < 0:
        carry_size = 65535

    min0, max0, step0 = qint0
    min1, max1, step1 = qint1
    if sub:
        min1, max1 = max1, min1
    sf = 2.0**shift
    min1, max1, step1 = min1 * sf, max1 * sf, step1 * sf
    max0, max1 = max0 + step0, max1 + step1

    f = -log2(max(step0, step1))
    i = ceil(log2(max(abs(min0), abs(min1), abs(max0), abs(max1))))
    k = int(qint0.min < 0 or qint1.min < 0)
    n_accum = k + i + f
    # Align to the number of carry and adder bits, when they are block-based (e.g., 4/8 bits look-ahead carry in Xilinx FPGAs)
    # For Altera, the carry seems to be single bit adder chains, but need to check
    return float(ceil(n_accum / carry_size)), float(ceil(n_accum / adder_size))


@jit
def create_state(
    kernel: np.ndarray,
    qintervals: list[QInterval],
    inp_latencies: list[float],
    no_stat_init: bool = False,
):
    assert len(qintervals) == kernel.shape[0]
    assert len(inp_latencies) == kernel.shape[0]
    assert kernel.ndim == 2

    kernel = kernel.astype(np.float64)
    n_in, n_out = kernel.shape
    kernel = np.asarray(kernel)
    csd, shift0, shift1 = csd_decompose(kernel)
    for i, qint in enumerate(qintervals):
        if qint.min == qint.max == 0:
            csd[i] = 0
    n_bits = csd.shape[-1]
    expr = list(csd)
    shifts = (shift0, shift1)

    # Dirty numba typing trick
    stat = {Pair(-1, -1, False, 0): 0}
    del stat[Pair(-1, -1, False, 0)]

    # Loop over outputs, in0, in1, shift0, shift1 to gather all two-term pairs
    # Force i1>=i0
    if not no_stat_init:
        # Initialize the stat dictionary
        # Skip if no_stat_init is True (skip optimization)
        for i_out in range(n_out):
            for i0 in range(n_in):
                for j0 in range(n_bits):
                    bit0 = csd[i0, i_out, j0]
                    if not bit0:
                        continue
                    for i1 in range(i0, n_in):
                        for j1 in range(n_bits):
                            bit1 = csd[i1, i_out, j1]
                            if not bit1:
                                continue
                            # Avoid count the same bit
                            if i0 == i1 and j0 <= j1:
                                continue
                            pair = Pair(i0, i1, bit0 != bit1, j1 - j0)
                            stat[pair] = stat.get(pair, 0) + 1

        for k in list(stat.keys()):
            if stat[k] < 2.0:
                del stat[k]

    ops = [Op(i, -1, -1, 0, qintervals[i], inp_latencies[i], 0.0) for i in range(n_in)]

    return DAState(
        shifts=shifts,
        expr=expr,
        ops=ops,
        freq_stat=stat,
        kernel=kernel,
    )


@jit
def update_stats(
    state: DAState,
    pair: Pair,
):
    """Updates the statistics of any 2-term pair in the state that may be affected by implementing op."""
    id0, id1 = pair.id0, pair.id1

    ks = list(state.freq_stat.keys())
    for k in ks:
        if k.id0 == id0 or k.id1 == id1 or k.id1 == id0 or k.id0 == id1:
            del state.freq_stat[k]

    n_constructed = len(state.expr)
    modified = [n_constructed - 1]
    modified.append(id0)
    if id1 != id0:
        modified.append(id1)

    n_bits = state.expr[0].shape[-1]

    # Loop over outputs, in0, in1, shift0, shift1 to gather all two-term pairs
    for i_out in range(state.kernel.shape[1]):
        for _in0 in modified:
            for _in1 in range(n_constructed):
                if _in1 in modified and _in0 > _in1:
                    # Avoid double counting of the two locations when _i0 != _i1
                    continue
                # Order inputs, as _in0 can be either in0 or in1, range of _in is not restricted
                id0, id1 = (_in0, _in1) if _in0 <= _in1 else (_in1, _in0)
                for j0 in range(n_bits):
                    bit0 = state.expr[id0][i_out, j0]
                    if not bit0:
                        continue
                    for j1 in range(n_bits):
                        bit1 = state.expr[id1][i_out, j1]
                        if not bit1:
                            continue
                        if id0 == id1 and j0 <= j1:
                            continue
                        pair = Pair(id0, id1, bit0 != bit1, j1 - j0)
                        state.freq_stat[pair] = state.freq_stat.get(pair, 0) + 1

    ks, vs = list(state.freq_stat.keys()), list(state.freq_stat.values())
    for k, v in zip(ks, vs):
        if v < 2.0:
            del state.freq_stat[k]
    return state


@jit
def gather_matching_idxs(state: DAState, pair: Pair):
    """Generates all i_out, j0, j1 ST expr[i_out][in0, j0] and expr[i_out][in1, j1] corresponds to op provided."""
    id0, id1 = pair.id0, pair.id1
    shift = pair.shift
    sub = pair.sub
    n_out = state.kernel.shape[1]
    n_bits = state.expr[0].shape[-1]

    flip = False
    if shift < 0:
        id0, id1 = id1, id0
        shift = -shift
        flip = True

    sign = 1 if not sub else -1

    for j0 in range(n_bits - shift):
        for i_out in range(n_out):
            bit0 = state.expr[id0][i_out, j0]
            j1 = j0 + shift
            bit1 = state.expr[id1][i_out, j1]
            if sign * bit1 * bit0 != 1:
                continue

            if flip:
                yield i_out, j1, j0
            else:
                yield i_out, j0, j1


@jit
def pair_to_op(pair: Pair, state: DAState, adder_size: int = -1, carry_size: int = -1):
    id0, id1 = pair.id0, pair.id1
    dlat, cost = cost_add(
        state.ops[pair.id0].qint,
        state.ops[pair.id1].qint,
        pair.shift,
        pair.sub,
        adder_size=adder_size,
        carry_size=carry_size,
    )
    lat = max(state.ops[id0].latency, state.ops[id1].latency) + dlat
    qint = qint_add(
        state.ops[pair.id0].qint,
        state.ops[pair.id1].qint,
        shift=pair.shift,
        sub1=pair.sub,
    )
    return Op(id0, id1, int(pair.sub), pair.shift, qint, lat, cost)


@jit
def update_expr(
    state: DAState,
    pair: Pair,
    adder_size: int,
    carry_size: int,
):
    "Updates the state by implementing the operation op, excepts common 2-term pair freq update."
    id0, id1 = pair.id0, pair.id1
    op = pair_to_op(pair, state, adder_size=adder_size, carry_size=carry_size)
    n_out = state.kernel.shape[1]
    n_bits = state.expr[0].shape[-1]

    expr = state.expr.copy()
    ops = state.ops.copy()

    ops.append(op)

    new_slice = np.zeros((n_out, n_bits), dtype=np.int8)

    for i_out, j0, j1 in gather_matching_idxs(state, pair):
        new_slice[i_out, j0] = expr[id0][i_out, j0]
        expr[id0][i_out, j0] = 0
        expr[id1][i_out, j1] = 0

    expr.append(new_slice)

    return DAState(
        shifts=state.shifts,
        expr=expr,
        ops=ops,
        freq_stat=state.freq_stat,
        kernel=state.kernel,
    )


@jit
def update_state(
    state: DAState,
    pair_chosen: Pair,
    adder_size: int,
    carry_size: int,
):
    """Update the state by removing all occurrences of pair_chosen from the state, register op code, and update the statistics."""
    state = update_expr(state, pair_chosen, adder_size=adder_size, carry_size=carry_size)
    state = update_stats(state, pair_chosen)
    return state
