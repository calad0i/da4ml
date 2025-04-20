from math import floor

from ..cmvm.types import CascadedSolution, Op
from .tracer import Trace, trace_to_solution


def pipelining(trace: Trace) -> CascadedSolution:
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
    assert len(trace.ops) > 0, 'No operations in the record'
    for i, op in enumerate(trace.ops):
        if op.id1 != -1:
            break

    latency_cutoff = trace.hwconf.latency_cutoff

    def get_stage(op: Op):
        return floor(op.latency / (latency_cutoff + 1e-9)) if latency_cutoff > 0 else 0

    opd: dict[int, list[Op]] = {}
    out_idxd: dict[int, list[int]] = {}

    locator: list[dict[int, int]] = []

    ops = trace.ops.copy()
    lat = max(ops[i].latency for i in trace.out_idx)
    for i in trace.out_idx:
        op_out = ops[i]
        ops.append(Op(i, -4, False, 0, op_out.qint, lat, 0.0))

    for i, op in enumerate(ops):
        stage = get_stage(op)
        if op.id1 == -1:
            # Copy from external buffer
            opd.setdefault(stage, []).append(op)
            locator.append({stage: i})
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

        if p1_idx == -4:
            # Output to external buffer
            out_idxd.setdefault(stage, []).append(p0_idx)
        else:
            _Op = Op(p0_idx, p1_idx, op.sub, op.shift, op.qint, op.latency, op.cost)
            opd.setdefault(stage, []).append(_Op)
            locator.append({stage: len(opd[stage]) - 1})
    outputs = []
    max_stage = max(opd.keys())
    for i, stage in enumerate(opd.keys()):
        _ops = opd[stage]
        _out_idx = out_idxd[stage]
        if i == max_stage:
            out_factors = trace.out_factors
        else:
            out_factors = [1.0] * len(_out_idx)
        rec = Trace(
            trace.hwconf,
            ops=_ops,
            out_idx=_out_idx,
            out_factors=out_factors,
        )
        outputs.append(trace_to_solution(rec))
    return CascadedSolution(tuple(outputs))
