from math import ceil, log2

import numpy as np
from numba import jit

from .core import _solve, create_state, to_solution
from .types import CascadedSolution, QInterval
from .util import kernel_decompose


@jit(cache=True)
def minimal_latency(
    kernel: np.ndarray,
    qintervals: list[QInterval],
    latencies: list[float],
    carry_size: int = -1,
    adder_size: int = -1,
):
    """Fast latency calculation for a given kernel, QInterval, and input latencies.
    When carry_size=-1, and the input latency is constant `l`:
    this will be the same as `l + max(ceiling(log2(max(#CSD bits for each column, 1))))`.

    Parameters
    ----------
    kernel : np.ndarray
        The input kernel matrix.
    qintervals : list[QInterval]
        List of QIntervals for each input.
    latencies : list[float]
        List of latencies for each input
    carry_size : int, optional
        The size of the carry unit for latency computation, by default -1 (fixed latency for each addition operation)
    adder_size : int, optional
        The size of the adder unit for latency computation, by default -1 (fixed cost for each addition operation)

    Returns
    -------
    float
        The minimal latency for the given kernel, QInterval, and input latencies.
    """

    state = create_state(kernel, qintervals, latencies, no_stat_init=True)
    solution = to_solution(state, latencies, qintervals, adder_size=adder_size, carry_size=carry_size)
    lat = max(solution.out_lat)
    return lat


@jit(cache=True)
def solve(
    kernel: np.ndarray,
    method0: str = 'wmc',
    method1: str = 'auto',
    hard_dc: int = -1,
    decompose_dc: int = -1,
    qintervals: list[QInterval] | None = None,
    inp_latencies: list[float] | None = None,
    adder_size: int = -1,
    carry_size: int = -1,
) -> CascadedSolution:
    """Optimized implementation of a CMVM computation with cascaded two matrices.

    Parameters
    ----------
    kernel : np.ndarray
        The input kernel matrix to be implemented.
    method0 : str, optional
        Optimization method for the first stage. Must be one of [`wmc`, `wmc-dc`, `wmc-pdc`, `mc`, `mc-dc`, `mc-pdc`].
    method1 : str, optional
        Optimization method for the second stage. When 'auto', it will select based on hard_dc and method0, by default 'auto'
    hard_dc : int, optional
        Hard depth constraint (additional latency allowed beyond minimal latency), by default -1 (no constraint)
    decompose_dc : int, optional
        Decomposition depth constraint, by default -1 (no constraint, follows hard_dc)
    qintervals : list[QInterval] | None, optional
        List of quantization intervals for each input, by default None ([-128, 127, 1] for all inputs)
    inp_latencies : list[float] | None, optional
        List of input latencies, by default None (0. for all inputs)
    adder_size : int, optional
        Size of the adder unit for latency computation, by default -1 (fixed cost for each addition)
    carry_size : int, optional
        Size of the carry unit for latency computation, by default -1 (fixed latency for each addition)

    Returns
    -------
    CascadedSolution
        A solution containing the optimized implementation of the CMVM computation with cascaded stages.
    """

    if hard_dc < 0:
        hard_dc = int(1e9)

    if method1 == 'auto':
        if hard_dc >= 6 or method0.endswith('dc'):
            method1 = method0
        else:
            method1 = method0 + '-dc'

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

    min_lat = minimal_latency(kernel, _qintervals, _inp_latencies, carry_size=carry_size, adder_size=adder_size)
    latency_allowed = hard_dc + min_lat
    if decompose_dc < 0:
        decompose_dc = min(hard_dc, ceil(log2(kernel.shape[0])))
    else:
        decompose_dc = min(hard_dc, decompose_dc, ceil(log2(kernel.shape[0])))

    while True:
        if decompose_dc <= 0:
            method0, method1 = 'wmc-dc', 'wmc-dc'
        mat0, mat1 = kernel_decompose(kernel, dc=decompose_dc)
        sol0 = _solve(
            mat0, method=method0, qintervals=_qintervals, latencies=_inp_latencies, adder_size=adder_size, carry_size=carry_size
        )
        if max(sol0.out_lat) > latency_allowed:
            # Prevent infinite loop, shouldn't happen though
            if not method0 == method1 == 'wmc-dc' or decompose_dc > 0:
                decompose_dc -= 1
                continue
        sol1 = _solve(
            mat1, method=method1, qintervals=sol0.out_qint, latencies=sol0.out_lat, adder_size=adder_size, carry_size=carry_size
        )
        if max(sol1.out_lat) > latency_allowed:
            # Prevent infinite loop, shouldn't happen though
            if not method0 == method1 == 'wmc-dc' or decompose_dc > 0:
                decompose_dc -= 1
                continue
        break
    if max(sol1.out_lat) > latency_allowed:
        # Should never happen
        print(f'Latency constraint not satisfied: {int(latency_allowed)} < {int(max(sol1.out_lat))}')
    return CascadedSolution((sol0, sol1))
