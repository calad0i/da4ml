from math import ceil, log2

import numpy as np
from numba import jit

from .bit_decompose import _center, _volatile_int_arr_to_csd


@jit
def prim_mst_dc(cost_mat: np.ndarray, dc: int = -1):
    """Minimum Spanning Tree (MST) using Prim's algorithm with a delay constraint. May not be optimal.
    Always start from the root node (0).

    Parameters
    ----------
    cost_mat : np.ndarray
        The adjacency matrix of the graph, where cost_mat[i, j] is the cost of the edge between i and j.

    dc : int, optional
        The delay constraint, by default -1
        If -1, no delay constraint is applied.

        Delay of each edge is ceiling(log2(cost_mat[i, j])).

        Delay from the root node to any node is the **maximum** latency of each edge connecting in between,
        plus ceiling(log2(#number of connection edges)).
        Latency is **NOT** the sum of the latencies.

    Returns
    -------
    np.ndarray
        The adjacency list of the MST, where each row is a pair of nodes (parent, child).
    """

    N = len(cost_mat)
    lat_mat = np.ceil(np.log2(np.maximum(cost_mat, 1)))
    parent = np.full(N, -2, dtype=np.int32)  # -2: not visited, -1: root

    parent[0] = -1
    idxs = np.arange(N)

    mapping = np.empty((N - 1, 2), dtype=np.int32)
    latency = np.zeros((N,), dtype=np.int32)

    if dc >= 0:
        _dc = (2**dc - 1) + ceil(log2(np.max(cost_mat[0]) + 1e-32))
    else:
        _dc = -1

    for n_impl in range(1, N):
        implemented = parent != -2
        _cost = cost_mat[~implemented][:, implemented]
        if dc >= 0:
            _lat = lat_mat[~implemented][:, implemented]
            _cost = np.where(np.maximum(_lat, latency[implemented]) + 1 <= _dc, _cost, np.iinfo(_cost.dtype).max // 2)
        _idx = int(np.argmin(_cost))
        _i, _j = _idx // n_impl, _idx % n_impl
        i, j = idxs[~implemented][_i], idxs[implemented][_j]
        parent[i] = j
        mapping[n_impl - 1, 0] = j
        mapping[n_impl - 1, 1] = i
        latency[i] = max(lat_mat[i, j], latency[j]) + 1  # type: ignore

    return mapping


@jit
def kernel_decompose(kernel: np.ndarray, dc: int = -2):
    """Decompose a 2D kernel matrix into two matrices with the delay-constrained approx MST.

    Parameters
    ----------
    kernel : np.ndarray
        The input kernel matrix to decompose.

    dc : int, optional
        Delay constraint, by default -1
        If -2, no delay constraint is applied.
        If -1, return trivial decomposition (m0 = kernel, m1 = I).

        The delay constraint limits the maximum latency (hops) of the decomposed
        multiplication structure.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The decomposed matrices (m0, m1): kernel = m0 @ m1
    """
    kernel, shift0, shift1 = _center(kernel)
    scale0, scale1 = 2.0**shift0, 2.0**shift1
    m, n = kernel.shape[0], kernel.shape[1] + 1
    mat_aug = np.zeros((m, n), dtype=kernel.dtype)
    mat_aug[:, 1:] = kernel
    diff0 = mat_aug[:, :, None] - mat_aug[:, None, :]
    diff1 = mat_aug[:, :, None] + mat_aug[:, None, :]
    dist0 = np.sum(np.sum(_volatile_int_arr_to_csd(diff0) != 0, axis=3), axis=0)
    dist1 = np.sum(np.sum(_volatile_int_arr_to_csd(diff1) != 0, axis=3), axis=0)
    sign = np.where(dist1 - dist0 < 0, -1, 1)
    dist = np.minimum(dist0, dist1)
    mapping = prim_mst_dc(dist, dc=dc)
    n_in, n_out = kernel.shape
    m0, m1 = np.zeros((n_in, n_out), dtype=kernel.dtype), np.zeros((n_out, n_out), dtype=kernel.dtype)

    if dc == -1:
        m0[:] = kernel
        m1[:] = np.eye(n_out, dtype=kernel.dtype)
        return m0 * scale0[:, None], m1 * scale1

    cnt = 0
    for _from, _to in mapping:
        col0 = mat_aug[:, _to] - mat_aug[:, _from] * sign[_to, _from]
        if _from != 0:
            col1 = m1[:, _from - 1].copy() * sign[_to, _from]
        else:
            col1 = np.zeros(n_out, dtype=kernel.dtype)
        if np.any(col0 != 0):
            col1[cnt] = 1
            m0[:, cnt] = col0
            cnt += 1
        m1[:, _to - 1] = col1
    return m0 * scale0[:, None], m1 * scale1
