from math import ceil

import numpy as np

from da4ml.types import CombLogic, Op, Pipeline

from .passes import dead_code_elimin


def _index_remap(op: Op, idx_map: dict[int, int]) -> Op:
    if op.opcode == -1:
        return op
    id0 = op.id0
    id1 = op.id1
    id0 = idx_map.get(id0, id0)
    id1 = idx_map.get(id1, id1)
    if op.opcode == 6:  # msb_mux
        id_c = op.data & 0xFFFFFFFF
        shift = (op.data >> 32) & 0xFFFFFFFF
        id_c = idx_map.get(id_c, id_c)
        data = id_c + (shift << 32)
    else:
        data = op.data
    return Op(id0, id1, op.opcode, data, op.qint, op.latency, op.cost)


def to_pipeline(comb: CombLogic, n_stages: int | None = None, latency_cutoff: float | None = None, verbose=True) -> Pipeline:
    """Split the record into multiple stages based on the latency of the operations.
    Only useful for HDL generation.
    Exactly one of n_stages and latency_cutoff must be specified.

    Parameters
    ----------
    sol : CombLogic
        The combinational logic to be pipelined into multiple stages.
    n_stages : int|None
        The number of stages to split the operations into. If None, it will be determined by the latency_cutoff. Default is None.
    latency_cutoff : float
        The latency cutoff for splitting the operations. If None, it will be determined by the n_stages. Default is None.
    verbose : bool
        Whether to print the actual cutoff used for splitting. Only used if rebalance is True.
        Default is True.

    Returns
    -------
    Pipeline
        The cascaded solution with multiple stages.
    """

    assert (n_stages is not None) + (latency_cutoff is not None) == 1, (
        'Exactly one of n_stages and latency_cutoff must be specified.'
    )
    latencies = [op.latency for op in comb.ops]
    assert np.all(np.diff(latencies) >= 0), 'Operations must be sorted by latency in descending order.'
    _latency = latencies[-1]
    n_stages = n_stages if n_stages is not None else max(ceil(_latency / latency_cutoff), 1)  # type: ignore
    assert n_stages > 0, 'Number of stages must be greater than 0.'
    lat_cutoffs = np.linspace(0, _latency, n_stages + 1)[1:]
    split_idxs = [0] + np.searchsorted(latencies, lat_cutoffs, side='right').tolist()
    if verbose:
        print(f'Latency cutoffs for splitting: {lat_cutoffs.tolist()}')

    staged_ops = [comb.ops[i:j] for i, j in zip(split_idxs[:-1], split_idxs[1:])]

    ext_req_idx: list[list[int]] = []
    'all idxs required by this or later stages from earlier stages'
    last_req = set(comb.out_idxs) - {-1}
    for ii in range(len(staged_ops) - 1, -1, -1):
        _req = {i for op in staged_ops[ii] for i in op.input_ids}
        _all_req = last_req.union(_req)
        _ext_req = {i for i in _all_req if i < split_idxs[ii]}
        last_req = _ext_req
        ext_req_idx.append(sorted(_ext_req))
    ext_req_idx = ext_req_idx[::-1] + [comb.out_idxs]

    index_maps: list[dict[int, int]] = []
    staged_ops_remap: list[list[Op]] = []
    for ii, _ops in enumerate(staged_ops):
        index_map: dict[int, int] = {}
        'global idx -> local idx'
        ops: list[Op] = []
        inp_idx = 0
        for i, gi in enumerate(ext_req_idx[ii]):
            _op = comb.ops[gi]
            index_map[gi] = i
            if _op.opcode != 5:
                ops.append(Op(inp_idx, -1, -1, 0, _op.qint, _op.latency, 0))
                inp_idx += 1
            else:
                ops.append(_op)  # const copy
        index_maps.append(index_map)

        global_base_idx = split_idxs[ii]
        local_base_idx = len(ext_req_idx[ii])
        for i, op in enumerate(_ops):
            ops.append(_index_remap(op, index_map))
            index_map[global_base_idx + i] = local_base_idx + i
        staged_ops_remap.append(ops)

    ext_req_idx = [[i for i in req if i < 0 or comb.ops[i].opcode != 5] for req in ext_req_idx]
    # const vars copied to individual stages

    combs: list[CombLogic] = []
    for ii, ops in enumerate(staged_ops_remap):
        if ii == 0:
            inp_shifts = comb.inp_shifts
            n_in = comb.shape[0]
        else:
            n_in = len(ext_req_idx[ii])
            inp_shifts = [0] * n_in
        index_map = index_maps[ii]
        if ii == n_stages - 1:
            out_idxs = [index_map[i] for i in comb.out_idxs]
            n_out = len(out_idxs)
            out_shifts = comb.out_shifts
            out_negs = comb.out_negs
        else:
            out_idxs = [index_map[i] for i in ext_req_idx[ii + 1]]
            n_out = len(out_idxs)
            out_shifts = [0] * n_out
            out_negs = [False] * n_out

        _comb = CombLogic(
            shape=(n_in, n_out),
            inp_shifts=inp_shifts,
            out_idxs=out_idxs,
            out_shifts=out_shifts,
            out_negs=out_negs,
            ops=ops,
            carry_size=comb.carry_size,
            adder_size=comb.adder_size,
            lookup_tables=comb.lookup_tables,
        )
        combs.append(dead_code_elimin(_comb))

    return Pipeline(tuple(combs))
