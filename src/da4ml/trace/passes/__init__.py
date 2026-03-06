from ...types import CombLogic
from .affine import affine_range_recomp
from .cse import common_subexpr_elimin
from .dce import dead_code_elimin
from .null_quant import null_quant_elimin
from .order import order_by_latency
from .surrogate import add_surrogate


def optimize(
    comb: CombLogic,
    keep_dead_inputs: bool = False,
) -> CombLogic:
    comb = common_subexpr_elimin(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = affine_range_recomp(comb)
    comb = common_subexpr_elimin(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = null_quant_elimin(comb)
    comb = dead_code_elimin(comb, keep_dead_inputs=keep_dead_inputs)
    comb = add_surrogate(comb)
    comb = order_by_latency(comb)
    return comb
