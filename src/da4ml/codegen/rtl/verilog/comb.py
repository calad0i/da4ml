from hashlib import sha256
from math import ceil, log2
from uuid import UUID

import numpy as np

from ....cmvm.types import CombLogic, Op, QInterval, _minimal_kif


def make_neg(lines: list[str], idx: int, qint: QInterval, v0_name: str, neg_repo: dict[int, tuple[int, str]]):
    if idx == 21568:
        pass
    if idx in neg_repo:
        return neg_repo[idx]
    _min, _max, step = qint
    bw0 = sum(_minimal_kif(qint))
    bw_neg = sum(_minimal_kif(QInterval(-_max, -_min, step)))
    was_signed = int(_min < 0)
    lines.append(
        f'wire [{bw_neg - 1}:0] v{idx}_neg; negative #({bw0}, {bw_neg}, {was_signed}) op_neg_{idx} ({v0_name}, v{idx}_neg);'
    )
    bw0 = bw_neg
    v0_name = f'v{idx}_neg'
    neg_repo[idx] = (bw0, v0_name)
    return bw0, v0_name


def gen_mem_file(sol: CombLogic, op: Op) -> str:
    assert op.opcode == 8
    assert sol.lookup_tables is not None
    table = sol.lookup_tables[op.data]
    width = sum(table.spec.out_kif)
    ndigits = ceil(width / 4)
    data = table.padded_table(sol.ops[op.id0].qint)
    mem_lines = [f'{hex(value)[2:].upper().zfill(ndigits)}' for value in data & ((1 << width) - 1)]
    return '\n'.join(mem_lines)


def get_table_name(sol: CombLogic, op: Op) -> str:
    memfile = gen_mem_file(sol, op)
    hash_obj = sha256(memfile.encode('utf-8'))
    _int = int(hash_obj.hexdigest()[:32], 16)
    uuid = UUID(int=_int, version=4)
    return f'table_{str(uuid)}.mem'


def ssa_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]], print_latency: bool = False) -> list[str]:
    ops = sol.ops
    kifs = list(map(_minimal_kif, (op.qint for op in ops)))
    widths: list[int] = list(map(sum, kifs))
    inp_kifs = [_minimal_kif(qint) for qint in sol.inp_qint]
    inp_widths = list(map(sum, inp_kifs))
    _inp_widths = np.cumsum([0] + inp_widths)
    inp_idxs = np.stack([_inp_widths[1:] - 1, _inp_widths[:-1]], axis=1)

    lines: list[str] = []
    ref_count = sol.ref_count

    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            continue

        bw = widths[i]
        v = f'v{i}[{bw - 1}:0]'
        _def = f'wire [{bw - 1}:0] v{i};'
        if bw == 0:
            continue

        match op.opcode:
            case -1:  # Input marker
                i0, i1 = inp_idxs[op.id0]
                line = f'{_def} assign {v} = model_inp[{i0}:{i1}];'

            case 0 | 1:  # Common a+/-b<<shift oprs
                p0, p1 = kifs[op.id0], kifs[op.id1]  # precision -> keep_neg, integers (no sign), fractional

                bw0, bw1 = widths[op.id0], widths[op.id1]  # width
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = op.data + f0 - f1
                v0, v1 = f'v{op.id0}[{bw0 - 1}:0]', f'v{op.id1}[{bw1 - 1}:0]'

                line = f'{_def} shift_adder #({bw0}, {bw1}, {s0}, {s1}, {bw}, {shift}, {op.opcode}) op_{i} ({v0}, {v1}, {v});'

            case 2 | -2:  # ReLU
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias

                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]

                if op.opcode == -2:
                    bw0, v0_name = make_neg(lines, op.id0, ops[op.id0].qint, v0_name, neg_repo)
                if ops[op.id0].qint.min < 0:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}] & {{{bw}{{~{v0_name}[{bw0 - 1}]}}}};'
                else:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'

            case 3 | -3:  # Explicit quantization
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]

                if op.opcode == -3:
                    bw0, v0_name = make_neg(lines, op.id0, ops[op.id0].qint, v0_name, neg_repo)

                if i0 >= bw0:
                    if op.opcode == 3:
                        assert ops[op.id0].qint.min < 0, f'{i}, {op.id0}'
                    else:
                        assert ops[op.id0].qint.max > 0, f'{i}, {op.id0}'

                    if i1 >= bw0:
                        v0_name = f'{{{i0 - i1 + 1}{{{v0_name}[{bw0 - 1}]}}}}'
                    else:
                        v0_name = f'{{{{{i0 - bw0 + 1}{{{v0_name}[{bw0 - 1}]}}}}, {v0_name}[{bw0 - 1}:{i1}]}}'
                    line = f'{_def} assign {v} = {v0_name};'
                else:
                    line = f'{_def} assign {v} = {v0_name}[{i0}:{i1}];'

            case 4:  # constant addition
                num = op.data
                sign, mag = int(num < 0), abs(num)
                bw1 = ceil(log2(mag + 1))
                bw0 = widths[op.id0]
                s0 = int(kifs[op.id0][0])
                v0 = f'v{op.id0}[{bw0 - 1}:0]'
                v1 = f"{bw1}'{bin(mag)[1:]}"
                shift = kifs[op.id0][2] - kifs[i][2]

                line = f'{_def} shift_adder #({bw0}, {bw1}, {s0}, 0, {bw}, {shift}, {sign}) op_{i} ({v0}, {v1}, {v});'

            case 5:  # constant
                num = op.data
                if num < 0:
                    num = 2**bw + num
                line = f"{_def} assign {v} = '{bin(num)[1:]};"

            case 6 | -6:  # MSB Muxing
                k, a, b = op.data & 0xFFFFFFFF, op.id0, op.id1
                p0, p1 = kifs[a], kifs[b]
                inv = '1' if op.opcode == -6 else '0'
                bwk, bw0, bw1 = widths[k], widths[a], widths[b]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                _shift = (op.data >> 32) & 0xFFFFFFFF
                _shift = _shift if _shift < 0x80000000 else _shift - 0x100000000
                shift = f0 - f1 + _shift
                vk, v0, v1 = f'v{k}[{bwk - 1}]', f'v{a}[{bw0 - 1}:0]', f'v{b}[{bw1 - 1}:0]'
                if bw0 == 0:
                    v0, bw0 = "1'b0", 1
                if bw1 == 0:
                    v1, bw1 = "1'b0", 1

                line = f'{_def} mux #({bw0}, {bw1}, {s0}, {s1}, {bw}, {shift}, {inv}) op_{i} ({vk}, {v0}, {v1}, {v});'

            case 7:  # Multiplication
                bw0, bw1 = widths[op.id0], widths[op.id1]  # width
                s0, s1 = int(kifs[op.id0][0]), int(kifs[op.id1][0])
                v0, v1 = f'v{op.id0}[{bw0 - 1}:0]', f'v{op.id1}[{bw1 - 1}:0]'

                line = f'{_def} multiplier #({bw0}, {bw1}, {s0}, {s1}, {bw}) op_{i} ({v0}, {v1}, {v});'

            case 8:  # Lookup Table
                name = get_table_name(sol, op)
                bw0 = widths[op.id0]

                line = f'{_def} lookup_table #({bw0}, {bw}, "{name}") op_{i} (v{op.id0}, {v});'

            case _:
                raise ValueError(f'Unknown opcode {op.opcode} for operation {i} ({op})')

        if print_latency:
            line += f' // {op.latency}'
        lines.append(line)
    return lines


def output_gen(sol: CombLogic, neg_repo: dict[int, tuple[int, str]]) -> list[str]:
    lines = []
    widths = list(map(sum, map(_minimal_kif, sol.out_qint)))
    _widths = np.cumsum([0] + widths)
    out_idxs = np.stack([_widths[1:] - 1, _widths[:-1]], axis=1)
    for i, idx in enumerate(sol.out_idxs):
        if idx < 0:
            continue
        i0, i1 = out_idxs[i]
        if i0 == i1 - 1:
            continue
        bw = widths[i]
        if sol.out_negs[i]:
            _, name = make_neg(lines, idx, sol.ops[idx].qint, f'v{idx}', neg_repo)
            lines.append(f'assign model_out[{i0}:{i1}] = {name}[{bw - 1}:0];')

        else:
            lines.append(f'assign model_out[{i0}:{i1}] = v{idx}[{bw - 1}:0];')
    return lines


def comb_logic_gen(sol: CombLogic, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    inp_bits = sum(map(sum, map(_minimal_kif, sol.inp_qint)))
    out_bits = sum(map(sum, map(_minimal_kif, sol.out_qint)))

    fn_signature = [
        f'module {fn_name} (',
        f'    input [{inp_bits - 1}:0] model_inp,',
        f'    output [{out_bits - 1}:0] model_out',
        ');',
    ]

    neg_repo: dict[int, tuple[int, str]] = {}
    ssa_lines = ssa_gen(sol, neg_repo=neg_repo, print_latency=print_latency)
    output_lines = output_gen(sol, neg_repo)

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


def table_mem_gen(sol: CombLogic) -> dict[str, str]:
    if not sol.lookup_tables:
        return {}
    mem_files = {get_table_name(sol, op): gen_mem_file(sol, op) for op in sol.ops if op.opcode == 8}
    return mem_files
