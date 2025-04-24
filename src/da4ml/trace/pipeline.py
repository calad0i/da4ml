from math import floor

from ..cmvm.types import CascadedSolution, Op, Solution


def pipelining(sol: Solution, latency_cutoff: int) -> CascadedSolution:
    """Split the record into multiple stages based on the latency of the operations.
    Only useful for HDL generation.

    Parameters
    ----------
    trace : Trace
        The trace record to be split. latency_cutoff is read from the hwconf.

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
        ops.append(Op(i, -1001, False, 0, op_out.qint, lat, 0.0))

    for i, op in enumerate(ops):
        stage = get_stage(op)
        if op.id1 in (-1, -4):
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
                _copy_op = Op(len(out_idxd[j]) - 1, -1, False, 0, op0.qint, latency, 0.0)
                opd.setdefault(j + 1, []).append(_copy_op)
                p0_idx = len(opd[j + 1]) - 1
                locator[op.id0][j + 1] = p0_idx
        else:
            p0_idx = locator[op.id0][stage]

        if op.id1 >= 0:
            p1_stages = locator[op.id1].keys()
            if stage not in p1_stages:
                # Need to copy parent to later states
                p1_stage = max(p1_stages)
                p1_idx = locator[op.id1][p1_stage]
                for j in range(p1_stage, stage):
                    op1 = ops[op.id1]
                    latency = float(latency_cutoff * (j + 1))
                    out_idxd.setdefault(j, []).append(locator[op.id1][j])
                    _copy_op = Op(len(out_idxd[j]) - 1, -1, False, 0, op1.qint, latency, 0.0)
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
            _Op = Op(p0_idx, p1_idx, op.sub, op.shift, op.qint, op.latency, op.cost)
            opd.setdefault(stage, []).append(_Op)
            locator.append({stage: len(opd[stage]) - 1})
    sols = []
    max_stage = max(opd.keys())
    for i, stage in enumerate(opd.keys()):
        _ops = opd[stage]
        _out_idx = out_idxd[stage]
        n_in = sum(op.id1 == -1 for op in _ops)
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
    return CascadedSolution(tuple(sols))
