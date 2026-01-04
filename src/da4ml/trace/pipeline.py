from math import ceil, floor

from ..cmvm.types import CombLogic, Op, Pipeline
from .fixed_variable import FixedVariable, HWConfig
from .tracer import comb_trace


def retime_pipeline(csol: Pipeline, verbose=True):
    n_stages = len(csol[0])
    cutoff_high = ceil(max(max(sol.out_latency) / (i + 1) for i, sol in enumerate(csol[0])))
    cutoff_low = 0
    adder_size, carry_size = csol[0][0].adder_size, csol[0][0].carry_size
    best = csol
    while cutoff_high - cutoff_low > 1:
        cutoff = (cutoff_high + cutoff_low) // 2
        _hwconf = HWConfig(adder_size, carry_size, cutoff)
        inp = [FixedVariable(*qint, hwconf=_hwconf) for qint in csol.inp_qint]
        try:
            out = list(csol(inp))
        except AssertionError:
            cutoff_low = cutoff
            continue
        _sol = to_pipeline(comb_trace(inp, out), cutoff, retiming=False)
        if len(_sol[0]) > n_stages:
            cutoff_low = cutoff
        else:
            cutoff_high = cutoff
            best = _sol
    if verbose:
        print(f'actual cutoff: {cutoff_high}')
    return best


def _get_new_idx(
    idx: int,
    locator: list[dict[int, int]],
    opd: dict[int, list[Op]],
    out_idxd: dict[int, list[int]],
    ops: list[Op],
    stage: int,
    latency_cutoff: float,
):
    if idx < 0:
        return idx
    p0_stages = locator[idx].keys()
    if stage not in p0_stages:
        # Need to copy parent to later states
        p0_stage = max(p0_stages)
        p0_idx = locator[idx][p0_stage]
        for j in range(p0_stage, stage):
            op0 = ops[idx]
            latency = float(latency_cutoff * (j + 1))
            out_idxd.setdefault(j, []).append(locator[idx][j])
            _copy_op = Op(len(out_idxd[j]) - 1, -1, -1, 0, op0.qint, latency, 0.0)
            opd.setdefault(j + 1, []).append(_copy_op)
            p0_idx = len(opd[j + 1]) - 1
            locator[idx][j + 1] = p0_idx
    else:
        p0_idx = locator[idx][stage]
    return p0_idx


def to_pipeline(comb: CombLogic, latency_cutoff: float, retiming=True, verbose=True) -> Pipeline:
    """Split the record into multiple stages based on the latency of the operations.
    Only useful for HDL generation.

    Parameters
    ----------
    sol : CombLogic
        The combinational logic to be pipelined into multiple stages.
    latency_cutoff : float
        The latency cutoff for splitting the operations.
    retiming : bool
        Whether to retime the solution after splitting. Default is True.
        If False, new stages are created when the propagation latency exceeds the cutoff.
        If True, after the first round of splitting, the solution is retimed balance the delay within each stage.
    verbose : bool
        Whether to print the actual cutoff used for splitting. Only used if rebalance is True.
        Default is True.

    Returns
    -------
    CascadedSolution
        The cascaded solution with multiple stages.
    """
    assert len(comb.ops) > 0, 'No operations in the record'
    for i, op in enumerate(comb.ops):
        if op.id1 != -1:
            break

    def get_stage(op: Op):
        return floor(op.latency / (latency_cutoff + 1e-9)) if latency_cutoff > 0 else 0

    opd: dict[int, list[Op]] = {}
    out_idxd: dict[int, list[int]] = {}

    locator: list[dict[int, int]] = []

    ops = comb.ops.copy()
    lat = max(ops[i].latency for i in comb.out_idxs)
    for i in comb.out_idxs:
        op_out = ops[i]
        ops.append(Op(i, -1001, -1001, 0, op_out.qint, lat, 0.0))

    for i, op in enumerate(ops):
        stage = get_stage(op)
        if op.opcode == -1:
            # Copy from external buffer
            opd.setdefault(stage, []).append(op)
            locator.append({stage: len(opd[stage]) - 1})
            continue

        p0_idx = _get_new_idx(op.id0, locator, opd, out_idxd, ops, stage, latency_cutoff)
        p1_idx = _get_new_idx(op.id1, locator, opd, out_idxd, ops, stage, latency_cutoff)
        if op.opcode in (6, -6):
            k = op.data & 0xFFFFFFFF
            _shift = (op.data >> 32) & 0xFFFFFFFF
            k = _get_new_idx(k, locator, opd, out_idxd, ops, stage, latency_cutoff)
            data = _shift << 32 | k
        else:
            data = op.data

        if p1_idx == -1001:
            # Output to external buffer
            out_idxd.setdefault(stage, []).append(p0_idx)
        else:
            _Op = Op(p0_idx, p1_idx, op.opcode, data, op.qint, op.latency, op.cost)
            opd.setdefault(stage, []).append(_Op)
            locator.append({stage: len(opd[stage]) - 1})
    sols = []
    max_stage = max(opd.keys())
    n_in = comb.shape[0]
    for i, stage in enumerate(opd.keys()):
        _ops = opd[stage]
        _out_idx = out_idxd[stage]
        n_out = len(_out_idx)

        if i == max_stage:
            out_shifts = comb.out_shifts
            out_negs = comb.out_negs
        else:
            out_shifts = [0] * len(_out_idx)
            out_negs = [False] * len(_out_idx)

        if comb.lookup_tables is not None:
            _ops, lookup_tables = remap_table_idxs(comb, _ops)
        else:
            lookup_tables = None
        _sol = CombLogic(
            shape=(n_in, n_out),
            inp_shifts=[0] * n_in,
            out_idxs=_out_idx,
            out_shifts=out_shifts,
            out_negs=out_negs,
            ops=_ops,
            carry_size=comb.carry_size,
            adder_size=comb.adder_size,
            lookup_tables=lookup_tables,
        )
        sols.append(_sol)

        n_in = n_out
    csol = Pipeline(tuple(sols))

    if retiming:
        csol = retime_pipeline(csol, verbose=verbose)
    return csol


def remap_table_idxs(comb: CombLogic, _ops):
    assert comb.lookup_tables is not None
    table_idxs = sorted(list({op.data for op in _ops if op.opcode == 8}))
    remap = {j: i for i, j in enumerate(table_idxs)}
    _ops_remap = []
    for op in _ops:
        if op.opcode == 8:
            op = Op(op.id0, op.id1, op.opcode, remap[op.data], op.qint, op.latency, op.cost)
        _ops_remap.append(op)
    _ops = _ops_remap
    lookup_tables = tuple(comb.lookup_tables[i] for i in table_idxs)
    return _ops, lookup_tables
