from math import ceil, log2

import numpy as np

from da4ml.cmvm.types import Op, QInterval, Solution, _minimal_kif


def ssa_gen(ops: list[Op], print_latency: bool = False):
    kifs = list(map(_minimal_kif, (op.qint for op in ops)))
    widths = list(map(sum, kifs))
    inp_kifs = [_minimal_kif(op.qint) for op in ops if op.opcode == -1]
    inp_widths = list(map(sum, inp_kifs))
    _inp_widths = np.cumsum([0] + inp_widths)
    inp_idxs = np.stack([_inp_widths[1:] - 1, _inp_widths[:-1]], axis=1)

    lines = []

    for i, op in enumerate(ops):
        bw = widths[i]
        v = f'v{i}[{bw-1}:0]'
        _def = f'wire [{bw-1}:0] v{i};'

        match op.opcode:
            case -1:  # Input marker
                i0, i1 = inp_idxs[op.id0]
                line = f'{_def} assign {v} = inp[{i0}:{i1}];'
            case 2 | -2:  # ReLU
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias

                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]

                if op.opcode == -2:
                    _min, _max, step = ops[op.id0].qint
                    bw_neg = max(sum(_minimal_kif(QInterval(-_max, -_min, step))), bw0)
                    lines.append(
                        f'wire [{bw_neg-1}:0] v{op.id0}_neg; assign v{op.id0}_neg[{bw_neg-1}:0] = -{v0_name}[{bw0-1}:0];'
                    )
                    v0_name = f'v{op.id0}_neg'
                if ops[op.id0].qint.min < 0:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}] & {{{bw}{{~{v0_name}[{bw0-1}]}}}};'
                else:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'
            case 3 | -3:  # Explicit quantization
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]

                if op.opcode == -3:
                    _min, _max, step = ops[op.id0].qint
                    bw_neg = max(sum(_minimal_kif(QInterval(-_max, -_min, step))), bw0)
                    lines.append(
                        f'wire [{bw_neg-1}:0] v{op.id0}_neg; assign v{op.id0}_neg[{bw_neg-1}:0] = -{v0_name}[{bw0-1}:0];'
                    )
                    v0_name = f'v{op.id0}_neg'

                line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'
            case 4:  # constant addition
                num = op.data
                sign, mag = int(num < 0), abs(num)
                line = f"{_def} assign {v} = '{bin(mag)[1:]};"
                bw1 = ceil(log2(mag + 1))
                bw0 = widths[op.id0]
                s0 = int(kifs[op.id0][0])
                v0 = f'v{op.id0}[{bw0-1}:0]'
                v1 = f"'{bin(mag)[1:]}"
                shift = int(log2(op.qint.step / ops[op.id0].qint.step))
                line = f'{_def} shift_adder #({bw0}, {bw1}, {s0}, 0, {bw}, {shift}, {sign}) op_{i} ({v0}, {v1}, {v});'
            case 5:  # constant
                num = op.data
                if num < 0:
                    num = 2**bw + num
                line = f"{_def} assign {v} = '{bin(num)[1:]};"

            case 0 | 1:  # Common a+/-b<<shift oprs
                p0, p1 = kifs[op.id0], kifs[op.id1]  # precision -> keep_neg, integers (no sign), fractional

                bw0, bw1 = widths[op.id0], widths[op.id1]  # width
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = op.data + f0 - f1
                v0, v1 = f'v{op.id0}[{bw0-1}:0]', f'v{op.id1}[{bw1-1}:0]'

                line = f'{_def} shift_adder #({bw0}, {bw1}, {s0}, {s1}, {bw}, {shift}, {op.opcode}) op_{i} ({v0}, {v1}, {v});'
            case _:
                raise ValueError(f'Unknown opcode {op.opcode} for operation {i} ({op})')

        if print_latency:
            line += f' // {op.latency}'
        lines.append(line)
    return lines


def output_gen(sol: Solution):
    lines = []
    widths = list(map(sum, map(_minimal_kif, sol.out_qint)))
    _widths = np.cumsum([0] + widths)
    out_idxs = np.stack([_widths[1:] - 1, _widths[:-1]], axis=1)
    for i, idx in enumerate(sol.out_idxs):
        if idx < 0:
            continue
        i0, i1 = out_idxs[i]
        bw = widths[i]
        bw0 = sum(_minimal_kif(sol.ops[idx].qint))
        if sol.out_negs[i]:
            lines.append(f'wire [{bw-1}:0] out_neg{i}; assign out_neg{i} = -v{idx}[{bw0-1}:0];')
            lines.append(f'assign out[{i0}:{i1}] = out_neg{i}[{bw-1}:0];')
        else:
            lines.append(f'assign out[{i0}:{i1}] = v{idx}[{bw-1}:0];')
    return lines


def comb_logic_gen(sol: Solution, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    inp_bits = sum(map(sum, map(_minimal_kif, sol.inp_qint)))
    out_bits = sum(map(sum, map(_minimal_kif, sol.out_qint)))

    fn_signature = [
        f'module {fn_name} (',
        f'    input [{inp_bits-1}:0] inp,',
        f'    output [{out_bits-1}:0] out',
        ');',
    ]

    ssa_lines = ssa_gen(sol.ops, print_latency=print_latency)
    output_lines = output_gen(sol)

    indent = '    '
    base_indent = '\n'
    body_indent = base_indent + indent
    code = f"""{base_indent[1:]}{base_indent.join(fn_signature)}

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    {body_indent.join(ssa_lines)}

    // verilator lint_on UNUSEDSIGNAL

    {body_indent.join(output_lines)}

    endmodule
"""
    if timescale is not None:
        code = f'{timescale}\n\n{code}'
    return code
