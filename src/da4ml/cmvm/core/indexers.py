import numpy as np
from numba import jit

from ..types import DAState, QInterval, minimal_kif


@jit
def idx_mc(state: DAState):
    """Choose the pair with highest frequency."""
    freqs = list(state.freq_stat.values())
    max_freq = max(freqs)
    pair_idx = freqs.index(max_freq)
    return pair_idx


@jit
def idx_mc_dc(state: DAState, absolute: bool = False):
    """Choose the pair with highest frequency with latency penalty.
    If absolute is True, return -1 if any latency overhead may present."""
    freqs = list(state.freq_stat.values())
    if max(freqs) <= 2:
        return -1
    factor = max(freqs) + 1
    ops = state.ops
    lat_penalty = [abs(ops[pair.id1].latency - ops[pair.id0].latency) * factor for pair in state.freq_stat.keys()]
    score = [freq - lat_penalty[i] for i, freq in enumerate(freqs)]
    max_score = max(score)
    if absolute and max_score < 0:
        return -1
    pair_idx = score.index(max_score)
    return pair_idx


@jit
def overlap_and_total(qint0: QInterval, qint1: QInterval):
    """Calculate the overlap and total number of bits for two QIntervals, when represented in fixed-point format."""
    k0, i0, f0 = minimal_kif(qint0)
    k1, i1, f1 = minimal_kif(qint1)
    n_overlap = max(min(k0, k1) + min(i0, i1) + min(f0, f1), 0)
    n_total = max(k0, k1) + max(i0, i1) + max(f0, f1)
    return n_overlap, n_total


@jit
def idx_wmc(state: DAState):
    """Choose the pair with the highest weighted most common subexpression (WMC) score."""
    freqs = list(state.freq_stat.values())
    keys = list(state.freq_stat.keys())
    score = np.empty(len(freqs), dtype=np.float32)
    for i, (k, v) in enumerate(zip(keys, freqs)):
        id0, id1 = k.id0, k.id1
        qint0, qint1 = state.ops[id0].qint, state.ops[id1].qint
        n_overlap, n_total = overlap_and_total(qint0, qint1)
        score[i] = v * (2 * n_overlap - n_total)
    max_score = np.max(score)
    if max_score < 0:
        return -1
    return np.argmax(score)


@jit
def idx_wmc_dc(state: DAState, absolute: bool = False):
    """Choose the pair with the highest weighted most common subexpression (WMC) score with latency and cost penalty.
    When absolute is True, return -1 if any latency overhead may present."""
    freqs = list(state.freq_stat.values())
    if max(freqs) == 2:
        return -1
    keys = list(state.freq_stat.keys())
    score = np.empty(len(freqs), dtype=np.float32)
    for i, (k, v) in enumerate(zip(keys, freqs)):
        id0, id1 = k.id0, k.id1
        qint0, qint1 = state.ops[id0].qint, state.ops[id1].qint
        lat0, lat1 = state.ops[id0].latency, state.ops[id1].latency
        n_overlap, n_total = overlap_and_total(qint0, qint1)
        score[i] = v * (2 * n_overlap - n_total) - 256 * abs(lat0 - lat1)
    if absolute and np.max(score) < 0:
        return -1
    return np.argmax(score)
