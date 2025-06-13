from math import ceil, log2

import numpy as np
from numba import jit, prange

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
    sol = to_solution(state, adder_size=adder_size, carry_size=carry_size)
    latencies = [sol.ops[i].latency if i >= 0 else 0.0 for i in sol.out_idxs]
    return max(latencies)


@jit(cache=True)
def jit_solve(
    kernel: np.ndarray,
    method0: str = 'wmc',
    method1: str = 'auto',
    hard_dc: int = -1,
    decompose_dc: int = -2,
    qintervals: list[QInterval] | None = None,
    latencies: list[float] | None = None,
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
    if hard_dc == 0 and not method0.endswith('dc'):
        method0 = method0 + '-dc'

    if qintervals is None:
        _qintervals = [QInterval(-128.0, 127.0, 1.0)] * kernel.shape[0]
    else:
        _qintervals = list(qintervals)
    if latencies is None:
        _inp_latencies = [0.0] * kernel.shape[0]
    else:
        _inp_latencies = [float(lat) for lat in latencies]
    assert len(_qintervals) == kernel.shape[0]
    assert len(_inp_latencies) == kernel.shape[0]

    min_lat = minimal_latency(kernel, _qintervals, _inp_latencies, carry_size=carry_size, adder_size=adder_size)
    latency_allowed = hard_dc + min_lat
    if decompose_dc == -2:
        decompose_dc = min(hard_dc, ceil(log2(kernel.shape[0])))
    else:
        decompose_dc = min(hard_dc, decompose_dc, ceil(log2(kernel.shape[0])))

    while True:
        if decompose_dc < 0 and hard_dc >= 0:
            if method0 != 'dummy':
                method0, method1 = 'wmc-dc', 'wmc-dc'
            else:
                method0, method1 = 'dummy', 'dummy'
        mat0, mat1 = kernel_decompose(kernel, dc=decompose_dc)
        sol0 = _solve(
            mat0, method=method0, qintervals=_qintervals, latencies=_inp_latencies, adder_size=adder_size, carry_size=carry_size
        )
        latencies0 = [sol0.ops[i].latency if i >= 0 else 0.0 for i in sol0.out_idxs]
        qintervals0 = [sol0.ops[i].qint if i >= 0 else QInterval(0.0, 0.0, np.inf) for i in sol0.out_idxs]
        if max(latencies0) > latency_allowed:
            if not method0 == method1 == 'wmc-dc' or decompose_dc >= 0:
                decompose_dc -= 1
                continue
        sol1 = _solve(
            mat1, method=method1, qintervals=qintervals0, latencies=latencies0, adder_size=adder_size, carry_size=carry_size
        )
        latencies1 = [sol1.ops[i].latency if i >= 0 else 0.0 for i in sol1.out_idxs]
        if max(latencies1) > latency_allowed:
            # Prevent infinite loop, shouldn't happen though
            if not method0 == method1 == 'wmc-dc' or decompose_dc >= 0:
                decompose_dc -= 1
                continue
        break
    if max(latencies1) > latency_allowed:
        # When latency depends on the bw, may happen
        print(f'Latency constraint not satisfied: {int(latency_allowed)} < {int(max(latencies1))}')
    return CascadedSolution((sol0, sol1))


@jit(cache=True, parallel=True)
def solve(
    kernel: np.ndarray,
    method0: str = 'wmc',
    method1: str = 'auto',
    hard_dc: int = -1,
    decompose_dc: int = -2,
    qintervals: list[QInterval] | None = None,
    latencies: list[float] | None = None,
    adder_size: int = -1,
    carry_size: int = -1,
    search_all_decompose_dc: bool = True,
) -> CascadedSolution:
    """Solve the CMVM problem with cascaded two matrices.

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
    search_all_decompose_dc : bool, optional
        If True, search for all possible decomposition depth constraints. If False, use the provided decompose_dc value.
        Default is True.

    Returns
    -------
    CascadedSolution
        A solution containing the optimized implementation of the CMVM computation with cascaded stages.
    """

    if qintervals is None:
        _qintervals = [QInterval(-128.0, 127.0, 1.0)] * kernel.shape[0]
    else:
        _qintervals = list(qintervals)
    if latencies is None:
        _latencies = [0.0] * kernel.shape[0]
    else:
        _latencies = [float(lat) for lat in latencies]

    if not search_all_decompose_dc:
        return jit_solve(
            kernel,
            method0=method0,
            method1=method1,
            hard_dc=hard_dc,
            decompose_dc=decompose_dc,
            qintervals=_qintervals,
            latencies=_latencies,
            adder_size=adder_size,
            carry_size=carry_size,
        )

    if hard_dc < 0:
        hard_dc = int(1e9)

    max_decompose_dc = min(hard_dc, ceil(log2(kernel.shape[0])))
    try_decompose_dcs = list(range(-1, max_decompose_dc + 1))

    costs = np.empty(len(try_decompose_dcs), dtype=np.float64)

    for i in prange(len(try_decompose_dcs)):
        decompose_dc = try_decompose_dcs[i]
        _csol = jit_solve(
            kernel,
            method0=method0,
            method1=method1,
            hard_dc=hard_dc,
            decompose_dc=decompose_dc,
            qintervals=_qintervals,
            latencies=_latencies,
            adder_size=adder_size,
            carry_size=carry_size,
        )
        _cost = sum([sum([op.cost for op in sol.ops]) for sol in _csol.solutions])
        costs[i] = _cost

    decompose_dc = try_decompose_dcs[np.argmin(costs)]
    csol = jit_solve(
        kernel,
        method0=method0,
        method1=method1,
        hard_dc=hard_dc,
        decompose_dc=decompose_dc,
        qintervals=_qintervals,
        latencies=_latencies,
        adder_size=adder_size,
        carry_size=carry_size,
    )
    return csol
