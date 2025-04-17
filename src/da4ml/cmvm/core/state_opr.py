from math import ceil, log2

import numpy as np
from numba import jit

from ..types import DAState, Op, QInterval
from ..util import csd_decompose


@jit
def _qint_add(qi1: QInterval, qi2: QInterval) -> QInterval:
    return QInterval(min=qi1.min + qi2.min, max=qi1.max + qi2.max, step=min(qi1.step, qi2.step))


@jit
def _qint_sub(qi1: QInterval, qi2: QInterval) -> QInterval:
    return QInterval(min=qi1.min - qi2.max, max=qi1.max - qi2.min, step=min(qi1.step, qi2.step))


@jit
def qint_add(qi1: QInterval, qi2: QInterval, sub0=False, sub1=False) -> QInterval:
    if sub0 != sub1:
        r = _qint_sub(qi1, qi2)
    else:
        r = _qint_add(qi1, qi2)
    if sub0:
        r = QInterval(min=-r.max, max=-r.min, step=r.step)
    return r


@jit
def cost_add(
    qint1: QInterval, qint2: QInterval, sub: bool = False, adder_size: int = -1, carry_size: int = -1
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

    f = -log2(min(qint1.step, qint2.step))
    min1, min2 = qint1.min, qint2.min
    max1, max2 = qint1.max, qint2.max
    if sub:
        min2, max2 = max2, min2
    max1, max2 = max1 + qint1.step, max2 + qint2.step
    i = ceil(log2(max(abs(min1), abs(min2), abs(max1), abs(max2))))
    k = int(qint1.min < 0 or qint2.min < 0)
    n_accum = k + i + f + 1
    # Align to the number of carry and adder bits, when they are block-based (e.g., 4/8 bits look-ahead carry in Xilinx FPGAs)
    # For Altera, the carry seems to be single bit adder chains, but need to check
    return float(ceil(n_accum / carry_size)), float(ceil(n_accum / adder_size))


@jit
def create_state(
    kernel: np.ndarray,
    qintervals: list[QInterval],
    inp_latencies: list[float],
    adder_size: int = -1,
    carry_size: int = -1,
    no_stat_init: bool = False,
):
    assert len(qintervals) == kernel.shape[0]
    assert len(inp_latencies) == kernel.shape[0]
    assert kernel.ndim == 2

    kernel = kernel.astype(np.float64)
    n_in, n_out = kernel.shape
    kernel = np.asarray(kernel)
    csd, shift0, shift1 = csd_decompose(kernel)
    n_bits = csd.shape[-1]
    expr = list(csd)
    shifts = (shift0, shift1)

    # Dirty numba typing trick
    stat = {Op(-1, -1, False, 0, 0.0, 0.0): 0}
    del stat[Op(-1, -1, False, 0, 0.0, 0.0)]
    expr_idx = list(range(len(expr)))

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
                            dlat, dcost = cost_add(
                                qintervals[i0],
                                qintervals[i1],
                                bit0 != bit1,
                                adder_size=adder_size,
                                carry_size=carry_size,
                            )
                            pair = Op(i0, i1, bit0 != bit1, j1 - j0, dlat, dcost)
                            stat[pair] = stat.get(pair, 0) + 1

        for k in list(stat.keys()):
            if stat[k] < 2.0:
                del stat[k]

    ops = [Op(i, -1, False, 0, 0.0, 0.0) for i in range(n_in)]

    return DAState(
        shifts=shifts,
        expr_idx=expr_idx,
        expr=expr,
        ops=ops,
        latencies=inp_latencies,
        qintervals=qintervals,
        freq_stat=stat,
        kernel=kernel,
    )


@jit
def update_stats(
    state: DAState,
    op: Op,
    adder_size: int = -1,
    carry_size: int = -1,
):
    """Updates the statistics of any 2-term pair in the state that may be affected by implementing op."""
    id0, id1 = op.id0, op.id1

    ks = list(state.freq_stat.keys())
    for k in ks:
        if k.id0 == id0 or k.id1 == id1 or k.id1 == id0 or k.id0 == id1:
            del state.freq_stat[k]

    n_constructed = len(state.expr)
    modified = [n_constructed - 1]
    if id0 in state.expr_idx:
        modified.append(state.expr_idx.index(id0))
    if id1 != id0 and id1 in state.expr_idx:
        modified.append(state.expr_idx.index(id1))

    n_bits = state.expr[0].shape[-1]

    # Loop over outputs, in0, in1, shift0, shift1 to gather all two-term pairs
    for i_out in range(state.kernel.shape[1]):
        for _in0 in modified:
            for _in1 in range(n_constructed):
                if _in1 in modified and _in0 > _in1:
                    # Avoid double counting of the two locations when _i0 != _i1
                    continue
                # Order inputs, as _in0 can be either in0 or in1, range of _in is not restricted
                in0, in1 = (_in0, _in1) if _in0 <= _in1 else (_in1, _in0)
                for j0 in range(n_bits):
                    bit0 = state.expr[in0][i_out, j0]
                    if not bit0:
                        continue
                    for j1 in range(n_bits):
                        bit1 = state.expr[in1][i_out, j1]
                        if not bit1:
                            continue
                        if in0 == in1 and j0 <= j1:
                            continue
                        id0, id1 = state.expr_idx[in0], state.expr_idx[in1]
                        dlat, dcost = cost_add(
                            state.qintervals[in0],
                            state.qintervals[in1],
                            bit0 != bit1,
                            adder_size=adder_size,
                            carry_size=carry_size,
                        )
                        pair = Op(id0, id1, bit0 != bit1, j1 - j0, dlat, dcost)
                        state.freq_stat[pair] = state.freq_stat.get(pair, 0) + 1

    ks, vs = list(state.freq_stat.keys()), list(state.freq_stat.values())
    for k, v in zip(ks, vs):
        if v < 2.0:
            del state.freq_stat[k]
    return state


@jit
def gather_matching_idxs(state: DAState, op: Op):
    """Generates all i_out, j0, j1 ST expr[i_out][in0, j0] and expr[i_out][in1, j1] corresponds to op provided."""
    id0, id1 = op.id0, op.id1
    in0, in1 = state.expr_idx.index(id0), state.expr_idx.index(id1)
    shift = op.shift
    sub = op.sub
    n_out = state.kernel.shape[1]
    n_bits = state.expr[0].shape[-1]

    flip = False
    if shift < 0:
        in0, in1 = in1, in0
        shift = -shift
        flip = True

    sign = 1 if not sub else -1

    for j0 in range(n_bits - shift):
        for i_out in range(n_out):
            bit0 = state.expr[in0][i_out, j0]
            j1 = j0 + shift
            bit1 = state.expr[in1][i_out, j1]
            if sign * bit1 * bit0 != 1:
                continue

            if flip:
                yield i_out, j1, j0
            else:
                yield i_out, j0, j1


@jit
def update_expr(
    state: DAState,
    op: Op,
):
    "Updates the state by implementing the operation op, excepts common 2-term pair freq update."
    id0, id1 = op.id0, op.id1
    in0, in1 = state.expr_idx.index(id0), state.expr_idx.index(id1)
    sub = op.sub
    n_out = state.kernel.shape[1]
    n_bits = state.expr[0].shape[-1]

    expr_idx = state.expr_idx.copy()
    expr = state.expr.copy()
    latencies = state.latencies.copy()
    qintervals = state.qintervals.copy()
    ops = state.ops.copy()

    ops.append(op)

    new_slice = np.zeros((n_out, n_bits), dtype=np.int8)

    for i_out, j0, j1 in gather_matching_idxs(state, op):
        new_slice[i_out, j0] = expr[in0][i_out, j0]
        expr[in0][i_out, j0] = 0
        expr[in1][i_out, j1] = 0

    expr.append(new_slice)
    expr_idx.append(len(ops) - 1)
    qint0, qint1 = qintervals[in0], qintervals[in1]
    latencies.append(max(latencies[in0], latencies[in1]) + op.dlatency)
    qintervals.append(qint_add(qint0, qint1, sub1=sub))

    if in0 != in1:
        if in0 > in1:
            it = [in0, in1]
        else:
            it = [in1, in0]
    else:
        it = [in0]

    for i in it:
        if np.all(expr[i] == 0):
            expr.pop(i)
            expr_idx.pop(i)
            latencies.pop(i)
            qintervals.pop(i)

    return DAState(
        shifts=state.shifts,
        expr_idx=expr_idx,
        expr=expr,
        ops=ops,
        latencies=latencies,
        qintervals=qintervals,
        freq_stat=state.freq_stat,
        kernel=state.kernel,
    )


@jit
def update_state(
    state: DAState,
    pair_chosen: Op,
    adder_size: int,
    carry_size: int,
):
    """Update the state by removing all occurrences of pair_chosen from the state, register op code, and update the statistics."""
    state = update_expr(state, pair_chosen)
    state = update_stats(state, pair_chosen, adder_size=adder_size, carry_size=carry_size)
    return state
