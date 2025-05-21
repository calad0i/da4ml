from math import ceil, floor

from ..cmvm.types import CascadedSolution, Op, Solution
from .fixed_variable import FixedVariable, HWConfig
from .tracer import comb_trace


def retime_pipeline(csol: CascadedSolution, verbose=True):
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


def to_pipeline(sol: Solution, latency_cutoff: int, retiming=True, verbose=True) -> CascadedSolution:
    """Split the record into multiple stages based on the latency of the operations.
    Only useful for HDL generation.

    Parameters
    ----------
    sol : Solution
        The solution to be split into multiple stages.
    latency_cutoff : int
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
    assert len(sol.ops) > 0, 'No operations in the record'
    for i, op in enumerate(sol.ops):
        if op.id1 != -1:
            break

    def get_stage(op: Op):
        return floor(op.latency / (latency_cutoff + 1e-9)) if latency_cutoff > 0 else 0

    opd: dict[int, list[Op]] = {}
    out_idxd: dict[int, list[int]] = {}

    locator: list[dict[int, int]] = []

    ops = sol.ops.copy()
    lat = max(ops[i].latency for i in sol.out_idxs)
    for i in sol.out_idxs:
        op_out = ops[i]
        ops.append(Op(i, -1001, -1001, 0, op_out.qint, lat, 0.0))

    for i, op in enumerate(ops):
        stage = get_stage(op)
        if op.opcode == -1:
            # Copy from external buffer
            opd.setdefault(stage, []).append(op)
            locator.append({stage: len(opd[stage]) - 1})
            continue
        p0_stages = locator[op.id0].keys()
        if stage not in p0_stages:
            # Need to copy parent to later states
            p0_stage = max(p0_stages)
            p0_idx = locator[op.id0][p0_stage]
            for j in range(p0_stage, stage):
                op0 = ops[op.id0]
                latency = float(latency_cutoff * (j + 1))
                out_idxd.setdefault(j, []).append(locator[op.id0][j])
                _copy_op = Op(len(out_idxd[j]) - 1, -1, -1, 0, op0.qint, latency, 0.0)
                opd.setdefault(j + 1, []).append(_copy_op)
                p0_idx = len(opd[j + 1]) - 1
                locator[op.id0][j + 1] = p0_idx
        else:
            p0_idx = locator[op.id0][stage]

        if op.opcode in (0, 1):
            p1_stages = locator[op.id1].keys()
            if stage not in p1_stages:
                # Need to copy parent to later states
                p1_stage = max(p1_stages)
                p1_idx = locator[op.id1][p1_stage]
                for j in range(p1_stage, stage):
                    op1 = ops[op.id1]
                    latency = float(latency_cutoff * (j + 1))
                    out_idxd.setdefault(j, []).append(locator[op.id1][j])
                    _copy_op = Op(len(out_idxd[j]) - 1, -1, -1, 0, op1.qint, latency, 0.0)
                    opd.setdefault(j + 1, []).append(_copy_op)
                    p1_idx = len(opd[j + 1]) - 1
                    locator[op.id1][j + 1] = p1_idx
            else:
                p1_idx = locator[op.id1][stage]
        else:
            p1_idx = op.id1

        if p1_idx == -1001:
            # Output to external buffer
            out_idxd.setdefault(stage, []).append(p0_idx)
        else:
            _Op = Op(p0_idx, p1_idx, op.opcode, op.data, op.qint, op.latency, op.cost)
            opd.setdefault(stage, []).append(_Op)
            locator.append({stage: len(opd[stage]) - 1})
    sols = []
    max_stage = max(opd.keys())
    for i, stage in enumerate(opd.keys()):
        _ops = opd[stage]
        _out_idx = out_idxd[stage]
        n_in = sum(op.opcode == -1 for op in _ops)
        n_out = len(_out_idx)

        if i == max_stage:
            out_shifts = sol.out_shifts
            out_negs = sol.out_negs
        else:
            out_shifts = [0] * len(_out_idx)
            out_negs = [False] * len(_out_idx)

        _sol = Solution(
            shape=(n_in, n_out),
            inp_shift=[0] * n_in,
            out_idxs=_out_idx,
            out_shifts=out_shifts,
            out_negs=out_negs,
            ops=_ops,
            carry_size=sol.carry_size,
            adder_size=sol.adder_size,
        )
        sols.append(_sol)
    csol = CascadedSolution(tuple(sols))

    if retiming:
        csol = retime_pipeline(csol, verbose=verbose)
    return csol
