import numpy as np

from .balanced_reduction import balanced_reduction
from .fixed_variable import FixedVariable
from .nb_fixed_precision import NBFixedPrecision
from .utils import DAState, OpCode


def var_from_nb_var(op_codes: OpCode, nb_variables: list[NBFixedPrecision], bias_idx: int):
    variables: list[FixedVariable] = []

    i = 0
    while i < len(op_codes) and op_codes[i].pos1 < 0:
        op_code = op_codes[i]
        pos0, pos1 = op_code.pos0, op_code.pos1
        shift = op_code.shift0
        if pos1 >= 0:
            break
        v = FixedVariable.from_nb_precision(nb_variables[pos0], name=f'inp[{pos0+bias_idx}]')
        v._factor = 2.0**shift
        variables.append(v)
        i += 1

    while i < len(op_codes):
        op_code = op_codes[i]
        pos0, pos1 = op_code.pos0, op_code.pos1
        shift0, shift1 = op_code.shift0, op_code.shift1
        sign0, sign1 = op_code.sign0, op_code.sign1
        v0, v1 = variables[pos0], variables[pos1]
        v = sign0 * (v0 << shift0) + sign1 * (v1 << shift1)
        variables.append(v)
        i += 1

    return variables


def gather_output_var_cumlist(state: DAState, variables: list[FixedVariable]):
    n_out = state.kernel.shape[1]
    cumlist = [[] for _ in range(n_out)]
    csd = np.array(state.csd)
    for di, do, shift in zip(*np.where(csd != 0)):
        sign = csd[di, do, shift]
        cumlist[do].append(sign * (variables[di] << shift))
    return cumlist


def graph_compile_states(states: list[list[DAState]], signed_balanced_reduction=True):
    n_split_in = len(states)
    n_split_out = len(states[0])
    assert all(len(states[i]) == n_split_out for i in range(n_split_in))
    kernel_shapes = np.empty((n_split_in, n_split_out, 2), dtype=np.int64)
    for i, j in np.ndindex(n_split_in, n_split_out):
        state = states[i][j]
        kernel_shapes[i, j] = state.kernel.shape
    assert np.all(np.std(kernel_shapes[:, :, 0], axis=1) == 0), 'Input kernel shapes must be the same'
    assert np.all(np.std(kernel_shapes[:, :, 1], axis=0) == 0), 'Output kernel shapes must be the same'
    n_in = kernel_shapes[:, 0, 0]
    n_out = kernel_shapes[0, :, 1]
    idx_in_biases = np.cumsum([0] + list(n_in[:-1]))
    idx_out_biases = np.cumsum([0] + list(n_out[:-1]))

    input_variables = []
    idx_in_bias = 0
    for i in range(n_split_in):
        _state = states[i][0]
        _n_in = n_in[i]
        _vars = var_from_nb_var(_state.op_codes[:_n_in], _state.variables[:_n_in], idx_in_bias)
        input_variables.append(_vars)
        idx_in_bias += _state.kernel.shape[0]

    output_variables = [[] for _ in range(np.sum(n_out))]
    for i, j in np.ndindex(n_split_in, n_split_out):
        state = states[i][j]
        idx_in_bias = idx_in_biases[i]
        idx_out_bias = idx_out_biases[j]
        variables = var_from_nb_var(state.op_codes, state.variables, idx_in_bias)
        _cumlist = gather_output_var_cumlist(state, variables)
        for k, buf in enumerate(_cumlist):
            output_variables[idx_out_bias + k].extend(buf)

    _output_variables: list[FixedVariable] = [
        balanced_reduction(buf, signed=signed_balanced_reduction) for buf in output_variables
    ]  # type: ignore
    input_variables = [v for vs in input_variables for v in vs]
    return input_variables, _output_variables
