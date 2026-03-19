from ...types import CombLogic
from .canon import canonicalize
from .cse import common_subexpr_elimin
from .dce import dead_code_elimin
from .null_op import null_quant_elimin
from .order import order_ops
from .surrogate import add_surrogate


def optimize(
    comb: CombLogic,
    keep_dead_inputs: bool = False,
) -> CombLogic:
    comb = canonicalize(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = common_subexpr_elimin(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = null_quant_elimin(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = add_surrogate(comb)
    comb = order_ops(comb)
    return comb
