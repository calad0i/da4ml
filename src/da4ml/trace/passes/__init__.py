from ...types import CombLogic
from .canon import canonicalize
from .cse import common_subexpr_elimin
from .dce import dead_code_elimin
from .null_op import null_quant_elimin
from .order import order_ops
from .retrace import _retrace
from .surrogate import add_surrogate


def optimize(
    comb: CombLogic,
    keep_dead_inputs: bool = False,
    surrogate=True,
    retrace=True,
) -> CombLogic:
    counter = 0
    while True:
        comb = canonicalize(comb)
        comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
        comb = common_subexpr_elimin(comb)
        comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
        comb = null_quant_elimin(comb)
        comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
        comb0 = order_ops(comb)
        if retrace:
            comb = _retrace(comb)
            comb = optimize(comb, retrace=False, keep_dead_inputs=keep_dead_inputs, surrogate=False)
        if not retrace or comb == comb0:
            break
        if counter > 2:
            raise RuntimeError('Optimization did not converge after 3 iterations')
        counter += 1
    if surrogate:
        comb = add_surrogate(comb)
    comb = order_ops(comb)
    return comb
