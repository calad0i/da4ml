import heapq
from collections.abc import Sequence
from math import ceil

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .csd import to_csd
from .nb_fixed_precision import NBFixedPrecision
from .scoring import scorer
from .utils import DAState, OpCode, Score


@njit
def extract_pairs(
    csd: list[NDArray[np.int8]], precisions: list[NBFixedPrecision], updated: list[int] | None = None, dc: int | None = None
):
    d_in = len(csd)
    if d_in == 0:
        raise ValueError('csd must have at least one element')
    d_out, n_bit = csd[0].shape
    _stat = np.zeros((d_in, d_in, n_bit, 2), dtype=np.int64)
    process_locs = np.zeros((d_in, d_out), dtype=np.bool_)

    if updated is not None:
        for i in range(len(updated)):
            pos = updated[i]
            for n in range(d_out):
                process_locs[pos, n] = True
    else:
        for pos in range(d_in):
            for n in range(d_out):
                if np.any(csd[pos][n]) and precisions[pos].b != 0:
                    process_locs[pos, n] = True

    if dc is not None:
        depths = np.zeros(d_in, dtype=np.int64)
        for pos in range(d_in):
            depths[pos] = precisions[pos]._depth
        depth_min = np.min(depths)
        mask = depths <= depth_min + dc
        if np.count_nonzero(mask) >= 2:
            for n in range(d_out):
                process_locs[:, n] &= mask

    args: list[tuple[int, int, int]] = []
    for pos0 in range(d_in):
        for pos1 in range(d_in):
            for n in range(d_out):
                if process_locs[pos0, n] or process_locs[pos1, n]:
                    args.append((pos0, pos1, n))

    for idx in prange(len(args)):
        pos0, pos1, n = args[idx]
        for shift0 in range(n_bit):
            if csd[pos0][n, shift0] == 0:
                continue
            lower = shift0 if pos0 < pos1 else shift0 + 1
            for shift1 in range(lower, n_bit):
                if csd[pos1][n, shift1] == 0:
                    continue
                dsign = int(csd[pos0][n, shift0] != csd[pos1][n, shift1])
                _stat[pos0, pos1, shift1 - shift0, dsign] += 1

    stat: list[tuple[float, int, int, int, int, int]] = []
    for pos0 in range(d_in):
        for pos1 in range(d_in):
            for dshift in range(n_bit):
                for dsign in range(2):
                    if _stat[pos0, pos1, dshift, dsign] > 1:
                        count = int(_stat[pos0, pos1, dshift, dsign])
                        score = scorer(precisions[pos0], precisions[pos1], dshift, dsign)
                        n_count = (count - 1) if count > 1 else 0
                        data = (-score * n_count, count, pos0, pos1, dshift, dsign)
                        stat.append(data)
    if updated is None:
        heapq.heapify(stat)
    return stat


@njit
def init_var(
    k: bool,
    b: int,
    i: int,
    symmetric: bool = False,
    _depth: int = 0,
) -> NBFixedPrecision:
    shift = b - i
    int_max = 2**b - 1
    int_min = -int_max - 1 + symmetric if k else 0
    return NBFixedPrecision(
        int_min,
        int_max,
        shift,
        symmetric,
        _depth,
    )


@njit
def init_vars(
    ks: tuple[bool, ...],
    bs: tuple[int, ...],
    is_: tuple[int, ...],
    symmetrics: tuple[bool, ...],
    depths: tuple[int, ...],
):
    n = len(ks)
    vars_ = []
    for i in range(n):
        vars_.append(init_var(ks[i], bs[i], is_[i], symmetrics[i], depths[i]))
    return vars_


@njit
def init_state(
    kernel: np.ndarray,
    signs: tuple[bool, ...],
    bits: tuple[int, ...],
    int_bits: tuple[int, ...],
    symmetrics: tuple[bool, ...],
    depths: tuple[int, ...],
):
    assert kernel.ndim == 2
    assert len(signs) == len(bits) == len(int_bits) == len(symmetrics) == len(depths) == kernel.shape[0]
    csd, shifts = to_csd(kernel)
    shift = shifts[0]

    d_in = len(csd)
    vars_ = init_vars(signs, bits, int_bits, symmetrics, depths)
    op_codes = []
    for i in range(d_in):
        opr_code = OpCode(i, -10, shift, 0, 0, 0)
        op_codes.append(opr_code)
        vars_[i] = vars_[i] << shift
    pairs = extract_pairs(csd, vars_)

    potential = 0.0
    for i in range(len(pairs)):
        potential -= pairs[i][0]
    score = Score(potential, 0.0, 0.0, 0.0)

    state = DAState(csd, vars_, op_codes, pairs, score, kernel)
    return state


@njit
def update_state(state: DAState, pair: tuple[float, int, int, int, int, int], dc=None):
    neg_cum_score, count, pos0, pos1, dshift, dsign = pair
    variables = state.variables.copy()
    op_codes = state.op_codes.copy()
    csd = state.csd
    pairs = state.pairs

    realized = state.score.realized - neg_cum_score

    _d_in, d_out, n_bit = len(csd), *csd[0].shape
    new_csd_col = np.zeros((d_out, n_bit), dtype=np.int8)
    dsign = -1 if dsign else 1
    for n in range(d_out):
        for shift0 in range(0, n_bit - dshift):
            _dsign = csd[pos0][n, shift0] * csd[pos1][n, shift0 + dshift]
            if _dsign == dsign:
                new_csd_col[n, shift0] = csd[pos0][n, shift0]
                csd[pos0][n, shift0] = 0
                csd[pos1][n, shift0 + dshift] = 0

    csd.append(new_csd_col)
    v0, v1 = variables[pos0], variables[pos1] << dshift
    v = v0 + v1 if dsign == 0 else v0 - v1
    variables.append(v)
    op_code = OpCode(pos0, pos1, 0, dshift, 1, dsign)
    op_codes.append(op_code)
    updated = [pos0, pos1, len(variables) - 1]

    d_pairs = extract_pairs(csd, variables, updated, dc)
    for i in range(len(pairs) - 1, -1, -1):
        _pair = pairs[i]
        if pos0 == _pair[2] or pos1 == _pair[2] or pos0 == _pair[3] or pos1 == _pair[3]:
            pairs.pop(i)

    for i in range(len(d_pairs)):
        heapq.heappush(pairs, d_pairs[i])

    cur_potential = 0.0
    for i in range(len(pairs)):
        cur_potential -= pairs[i][0]

    lost = state.score.potential - cur_potential
    value = realized - lost
    score = Score(state.score.potential, realized, lost, value)
    return DAState(csd, variables, op_codes, pairs, score, state.kernel)


@njit(cache=True)
def get_top_n_pairs(state: DAState, n: int):
    return state.pairs[:n]
    _pairs = state.pairs.copy()
    return [heapq.heappop(_pairs) for _ in range(min(n, len(_pairs)))]


@njit
def cmvm_cse(state: DAState, progress=None, beams: int = 1, dc=None):
    assert len(state.pairs) > 0, f'{len(state.pairs)}'
    top_pairs = get_top_n_pairs(state, beams)
    states_0 = [update_state(state, top_pairs[i], dc) for i in range(len(top_pairs))]

    next_score = np.full((beams, beams), -1, dtype=np.float64)
    use_n = np.empty(beams, dtype=np.int64)
    while True:
        use_n[:] = 0
        next_score[:] = -1
        for i in range(len(states_0)):
            if len(states_0[i].pairs) > 0 and states_0[i].pairs[0][0] < 0:
                break
        else:
            break

        for i in range(len(states_0)):
            # pass
            _state = states_0[i]
            next_score[i, 0] = _state.score.realized
            next_score[i, 1:] = -1
            top_pairs = get_top_n_pairs(_state, beams)
            for j in range(len(top_pairs)):
                next_score[i, j] = -top_pairs[j][0] + _state.score.realized

        order = np.argsort(next_score.ravel())[::-1]
        for i in range(beams):
            i_st, i_pair = np.divmod(order[i], next_score.shape[0])
            if next_score[i_st, i_pair] > 0:
                use_n[i_st] += 1

        states_1 = []
        for i in range(beams):
            if use_n[i] == 0:
                continue

            _state = states_0[i]
            if len(_state.pairs) == 0 or _state.pairs[0][0] > 0:
                states_1.append(_state)
                continue

            top_pairs = get_top_n_pairs(_state, use_n[i])
            for j in range(use_n[i]):
                states_1.append(update_state(_state, top_pairs[j], dc))

        states_0 = states_1

    _max = states_0[0].score.realized
    _idx = 0
    for i in range(len(states_0)):
        _state = states_0[i]
        if _state.score.realized > _max:
            _max = _state.score.realized
            _idx = i
    return states_0[_idx]


@njit
def compile_kernel_mono(
    kernel: np.ndarray,
    signs: tuple[bool, ...],
    bits: tuple[int, ...],
    int_bits: tuple[int, ...],
    symmetrics: tuple[bool, ...],
    depths: tuple[int, ...],
    n_beams: int = 1,
    dc: int | None = None,
):
    state = init_state(kernel, signs, bits, int_bits, symmetrics, depths)
    _state = cmvm_cse(state, beams=n_beams, dc=dc)
    return _state


def compile_kernel(
    kernel: np.ndarray,
    signs: Sequence[bool],
    bits: Sequence[int],
    int_bits: Sequence[int],
    symmetrics: Sequence[bool],
    depths: Sequence[int],
    n_beams: int = 1,
    dc: int | None = None,
    n_inp_max: int = -1,
    n_out_max: int = -1,
) -> list[list[DAState]]:
    d_inp, d_out = kernel.shape
    n_inp_part = 1
    if n_inp_max > 0 and n_inp_max < d_inp:
        n_inp_part = ceil(d_inp / n_inp_max)
    n_out_part = 1
    if n_out_max > 0 and n_out_max < d_out:
        n_out_part = ceil(d_out / n_out_max)

    inp_chunk_size = ceil(d_inp / n_inp_part)
    out_chunk_size = ceil(d_out / n_out_part)

    inp_part_locs = np.arange(0, n_inp_part + 1) * inp_chunk_size
    out_part_locs = np.arange(0, n_out_part + 1) * out_chunk_size

    states = [[None for _ in range(n_out_part)] for _ in range(n_inp_part)]
    for idx in range(n_inp_part * n_out_part):
        j, i = np.divmod(idx, n_inp_part)
        inp_start, inp_end = inp_part_locs[i], min(inp_part_locs[i + 1], d_inp)  # type: ignore
        out_start, out_end = out_part_locs[j], min(out_part_locs[j + 1], d_out)  # type: ignore
        _kernel = kernel[inp_start:inp_end, out_start:out_end]
        _signs = signs[inp_start:inp_end]
        _bits = bits[inp_start:inp_end]
        _int_bits = int_bits[inp_start:inp_end]
        _symmetrics = symmetrics[inp_start:inp_end]
        _depths = depths[inp_start:inp_end]

        # unify input type to prevent recompilation
        _kernel = np.ascontiguousarray(_kernel)
        _signs = tuple(bool(v) for v in _signs)
        _bits = tuple(int(v) for v in _bits)
        _int_bits = tuple(int(v) for v in _int_bits)
        _symmetrics = tuple(bool(v) for v in _symmetrics)
        _depths = tuple(int(v) for v in _depths)
        try:
            states[i][j] = compile_kernel_mono(_kernel, _signs, _bits, _int_bits, _symmetrics, _depths, n_beams, dc)
        except AssertionError:
            states[i][j] = init_state(_kernel, _signs, _bits, _int_bits, _symmetrics, _depths)

    return states  # type: ignore
