import heapq
from math import log2

import numpy as np
from numba import jit

from ..types import DAState, Op, QInterval, Solution
from .indexers import (
    idx_mc,
    idx_mc_dc,
    idx_wmc,
    idx_wmc_dc,
)
from .state_opr import cost_add, create_state, qint_add, update_state


@jit(cache=True)
def cmvm(
    kernel: np.ndarray,
    method: str = 'wmc',
    qintervals: list[QInterval] | None = None,
    inp_latencies: list[float] | None = None,
    adder_size: int = -1,
    carry_size: int = -1,
) -> DAState:
    """Optimizes the kernel using the CMVM algorithm.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel to optimize.
    method : str, optional
        Which indexing method to use, by default 'wmc' (weighted most common)
        Must be one of [`mc`, `mc-dc`, `mc-pdc`, `wmc`, `wmc-dc`, `wmc-pdc`, `dummy`]
    qintervals : list[QInterval] | None, optional
        List of QIntervals for each input, by default None
        If None, defaults to [-128., 127., 1.] for each input.
    inp_latencies : list[float] | None, optional
        List of latencies for each input, by default None
        If None, defaults to 0. for each input.
    adder_size : int, optional
        The atomic size of the adder for cost computation, by default -1
        if -1, each adder can be arbitrary large, and the cost will be the number of adders
    carry_size : int, optional
        The size of the carry unit for latency computation, by default -1
        if -1, each carry unit can be arbitrary large, and the cost will be the depth of the adder tree

    Returns
    -------
    DAState
        The optimized kernel as a DAState object.
    """

    if qintervals is None:
        _qintervals = [QInterval(-128.0, 127.0, 1.0)] * kernel.shape[0]
    else:
        _qintervals = [QInterval(*qi) for qi in qintervals]
    if inp_latencies is None:
        _inp_latencies = [0.0] * kernel.shape[0]
    else:
        _inp_latencies = [float(lat) for lat in inp_latencies]
    assert len(_qintervals) == kernel.shape[0]
    assert len(_inp_latencies) == kernel.shape[0]

    state = create_state(kernel, _qintervals, _inp_latencies)
    while True:
        if len(state.freq_stat) == 0:
            break
        match method:
            case 'mc':
                pair_idx = idx_mc(state)
            case 'mc-dc':
                pair_idx = idx_mc_dc(state, absolute=True)
            case 'mc-pdc':
                pair_idx = idx_mc_dc(state, absolute=False)
            case 'wmc':
                pair_idx = idx_wmc(state)
            case 'wmc-dc':
                pair_idx = idx_wmc_dc(state, absolute=True)
            case 'wmc-pdc':
                pair_idx = idx_wmc_dc(state, absolute=False)
            case 'dummy':
                break
            case _:
                raise ValueError(f'Unknown method: {method}')
        if pair_idx < 0:
            break
        pair_chosen = list(state.freq_stat.keys())[pair_idx]
        state = update_state(state, pair_chosen, adder_size=adder_size, carry_size=carry_size)
    return state


@jit(cache=True)
def to_solution(
    state: DAState,
    adder_size: int,
    carry_size: int,
):
    """Converts the DAState to a Solution object with balanced tree reduction for the non-extracted bits in the kernel.

    Parameters
    ----------
    state : DAState
        The DAState to convert.
    adder_size : int, optional
        The atomic size of the adder for cost computation, by default -1
        if -1, each adder can be arbitrary large, and the cost will be the number of adders
    carry_size : int, optional
        The size of the carry unit for latency computation, by default -1
        if -1, each carry unit can be arbitrary large, and the cost will be the depth of the adder tree

    Returns
    -------
    Solution
        The Solution object with the optimized kernel.
    """

    ops = state.ops.copy()
    n_out = state.kernel.shape[1]
    expr = np.empty((len(state.expr), *state.expr[0].shape), dtype=np.int8)
    for i, v in enumerate(state.expr):
        expr[i] = v
    in_shifts, out_shifts = state.shifts

    out_qints = []
    out_lats = []
    out_idx = []
    in_shift = in_shifts.copy()
    out_shift = out_shifts.copy()
    out_neg = []

    _global_id = len(ops)
    for i_out in range(n_out):
        idx, shifts = np.where(expr[:, i_out] != 0)
        sub = np.empty(len(idx), dtype=np.int64)
        for i, (i_in, shift) in enumerate(zip(idx, shifts)):
            sub[i] = expr[i_in, i_out, shift] == -1

        qints: list[QInterval] = [state.ops[i].qint for i in idx]
        lats: list[float] = [state.ops[i].latency for i in idx]

        # No reduction required, dump the realized value directly
        if len(sub) == 1:
            out_shift[i_out] = out_shift[i_out] + shifts[0]
            out_qints.append(qints[0])
            out_lats.append(lats[0])
            out_idx.append(idx[0])
            out_neg.append(sub[0])
            continue
        # Output is zero
        if len(sub) == 0:
            out_idx.append(-1)  # -1 means output constant zero
            out_qints.append(QInterval(0.0, 0.0, np.inf))
            out_lats.append(0.0)
            out_neg.append(False)
            continue

        # Sort by latency -> location of rightmost bit -> lower bound
        left_align: list[int] = []
        for i, qint in enumerate(qints):
            n_int = int(log2(max(abs(qint.max + qint.step), abs(qint.min))))
            left_align.append(n_int + shifts[i])
        heap = list(zip(lats, sub, left_align, qints, idx, shifts))
        heapq.heapify(heap)

        while len(heap) > 1:
            lat0, sub0, _, qint0, id0, shift0 = heapq.heappop(heap)
            lat1, sub1, _, qint1, id1, shift1 = heapq.heappop(heap)

            if sub0:
                shift = shift0 - shift1
                qint = qint_add(qint1, qint0, shift, sub1, sub0)
                dlat, dcost = cost_add(qint1, qint0, shift=shift, sub=1 ^ sub1, adder_size=adder_size, carry_size=carry_size)
                lat = max(lat0, lat1) + dlat
                op = Op(id1, id0, 1 ^ sub1, shift, qint, lat, dcost)
                shift = shift1
            else:
                shift = shift1 - shift0
                qint = qint_add(qint0, qint1, shift, sub0, sub1)
                dlat, dcost = cost_add(qint0, qint1, shift=shift, sub=sub1, adder_size=adder_size, carry_size=carry_size)
                lat = max(lat0, lat1) + dlat
                op = Op(id0, id1, sub1, shift, qint, lat, dcost)
                shift = shift0

            left_align = int(log2(max(abs(qint.max + qint.step), abs(qint.min)))) + shift
            heapq.heappush(heap, (lat, sub0 & sub1, left_align, qint, _global_id, shift))
            ops.append(op)
            _global_id += 1

        lat, sub, _, qint, id0, shift0 = heap[0]
        out_idx.append(_global_id - 1)
        out_qints.append(qint)
        out_lats.append(lat)
        out_neg.append(sub)
        out_shift[i_out] = out_shift[i_out] + shift0

    return Solution(
        shape=state.kernel.shape,  # type: ignore
        inp_shift=list(in_shift),
        out_idxs=out_idx,
        out_shifts=list(out_shift),
        out_negs=out_neg,
        ops=ops,
        carry_size=carry_size,
        adder_size=adder_size,
    )


@jit
def _solve(
    kernel: np.ndarray,
    method: str,
    qintervals: list[QInterval],
    latencies: list[float],
    adder_size: int,
    carry_size: int,
):
    state = cmvm(
        kernel, method=method, qintervals=qintervals, inp_latencies=latencies, adder_size=adder_size, carry_size=carry_size
    )
    return to_solution(state, adder_size=adder_size, carry_size=carry_size)
