from ...types import CombLogic


def _retrace(comb: CombLogic) -> CombLogic:
    from ..fixed_variable_array import FixedVariableArray
    from ..tracer import comb_trace

    inp = FixedVariableArray.from_kif(*comb.inp_kifs).as_new()
    out = comb(inp, quantize=False)
    return comb_trace(inp, out, optimize=False)
