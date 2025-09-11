from math import ceil, log2

import numpy as np

from ....cmvm.types import Op, QInterval, Solution, _minimal_kif


def make_neg(
    signals: list[str],
    assigns: list[str],
    op: Op,
    ops: list[Op],
    bw0: int,
    v0_name: str,
):
    _min, _max, step = ops[op.id0].qint
    bw_neg = max(sum(_minimal_kif(QInterval(-_max, -_min, step))), bw0)
    was_signed = int(_min < 0)
    signals.append(f'signal v{op.id0}_neg : std_logic_vector({bw_neg-1} downto {0});')
    assigns.append(
        f'op_neg_{op.id0} : entity work.negative generic map (BW_IN => {bw0}, BW_OUT => {bw_neg}, IN_SIGNED => {was_signed}) port map (neg_in => {v0_name}, neg_out => v{op.id0}_neg);'
    )
    bw0 = bw_neg
    v0_name = f'v{op.id0}_neg'
    return bw0, v0_name


def ssa_gen(sol: Solution, neg_defined: set[int], print_latency: bool = False):
    ops = sol.ops
    kifs = list(map(_minimal_kif, (op.qint for op in ops)))
    widths = list(map(sum, kifs))
    inp_kifs = [_minimal_kif(qint) for qint in sol.inp_qint]
    inp_widths = list(map(sum, inp_kifs))
    _inp_widths = np.cumsum([0] + inp_widths)
    inp_idxs = np.stack([_inp_widths[1:] - 1, _inp_widths[:-1]], axis=1)

    signals = []
    assigns = []
    ref_count = sol.ref_count

    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            continue

        bw = widths[i]
        if bw == 0:
            continue

        match op.opcode:
            case -1:  # Input marker
                i0, i1 = inp_idxs[op.id0]
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                line = f'v{i} <= model_inp({i0} downto {i1});'

            case 0 | 1:  # Common a+/-b<<shift oprs
                p0, p1 = kifs[op.id0], kifs[op.id1]
                bw0, bw1 = widths[op.id0], widths[op.id1]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = op.data + f0 - f1
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                line = f'op_{i}:entity work.shift_adder generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw},SHIFT1=>{shift},IS_SUB=>{op.opcode}) port map(in0=>v{op.id0},in1=>v{op.id1},result=>v{i});'

            case 2 | -2:  # ReLU
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]
                if op.opcode == -2 and op.id0 not in neg_defined:
                    neg_defined.add(op.id0)
                    bw0, v0_name = make_neg(signals, assigns, op, ops, bw0, v0_name)
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                if ops[op.id0].qint.min < 0:
                    if bw > 1:
                        line = f'v{i} <= {v0_name}({i0} downto {i1}) and ({bw - 1} downto 0 => not {v0_name}({bw0-1}));'
                    else:
                        line = f'v{i}(0) <= {v0_name}(0) and (not {v0_name}({bw0-1}));'
                else:
                    line = f'v{i} <= {v0_name}({i0} downto {i1});'

            case 3 | -3:  # Explicit quantization
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0, i1 = bw + lsb_bias - 1, lsb_bias
                v0_name = f'v{op.id0}'
                bw0 = widths[op.id0]
                if op.opcode == -3 and op.id0 not in neg_defined:
                    neg_defined.add(op.id0)
                    bw0, v0_name = make_neg(signals, assigns, op, ops, bw0, v0_name)
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                line = f'v{i} <= {v0_name}({i0} downto {i1});'

            case 4:  # constant addition
                num = op.data
                sign, mag = int(num < 0), abs(num)
                bw1 = ceil(log2(mag + 1)) if mag > 0 else 1
                bw0 = widths[op.id0]
                s0 = int(kifs[op.id0][0])
                shift = kifs[op.id0][2] - kifs[i][2]
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                bin_val = format(mag, f'0{bw1}b')
                line = f'op_{i}:entity work.shift_adder generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>0,BW_OUT=>{bw},SHIFT1=>{shift},IS_SUB=>{sign}) port map(in0=>v{op.id0},in1=>"{bin_val}",result=>v{i});'
            case 5:  # constant
                num = op.data
                if num < 0:
                    num = 2**bw + num
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                bin_val = format(num, f'0{bw}b')
                line = f'v{i} <= "{bin_val}";'

            case 6 | -6:  # MSB Muxing
                k, a, b = op.data & 0xFFFFFFFF, op.id0, op.id1
                p0, p1 = kifs[a], kifs[b]
                inv = '1' if op.opcode == -6 else '0'
                bwk, bw0, bw1 = widths[k], widths[a], widths[b]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                _shift = (op.data >> 32) & 0xFFFFFFFF
                _shift = _shift if _shift < 0x80000000 else _shift - 0x100000000
                shift = f0 - f1 + _shift
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                line = f'op_{i}:entity work.mux generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw},SHIFT1=>{shift},INVERT1=>{inv}) port map(key=>v{k}({bwk-1}),in0=>v{a},in1=>v{b},result=>v{i});'
            case 7:  # Multiplication
                bw0, bw1 = widths[op.id0], widths[op.id1]
                s0, s1 = int(kifs[op.id0][0]), int(kifs[op.id1][0])
                signals.append(f'signal v{i}:std_logic_vector({bw-1} downto {0});')
                line = f'op_{i}:entity work.multiplier generic map(BW_INPUT0=>{bw0},BW_INPUT1=>{bw1},SIGNED0=>{s0},SIGNED1=>{s1},BW_OUT=>{bw}) port map(in0=>v{op.id0},in1=>v{op.id1},result=>v{i});'

            case _:
                raise ValueError(f'Unknown opcode {op.opcode} for operation {i} ({op})')

        if print_latency:
            line += f' -- {op.latency}'
        assigns.append(line)
    return signals, assigns


def output_gen(sol: Solution, neg_defined: set[int]):
    assigns = []
    signals = []
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
            if idx not in neg_defined:
                neg_defined.add(idx)
                bw0 = sum(_minimal_kif(sol.ops[idx].qint))
                was_signed = int(_minimal_kif(sol.ops[idx].qint)[0])
                signals.append(f'signal v{idx}_neg:std_logic_vector({bw-1} downto {0});')
                assigns.append(
                    f'op_neg_{idx}:entity work.negative generic map(BW_IN=>{bw0},BW_OUT=>{bw},IN_SIGNED=>{was_signed}) port map(neg_in=>v{idx},neg_out=>v{idx}_neg);'
                )
            assigns.append(f'model_out({i0} downto {i1}) <= v{idx}_neg({bw-1} downto {0});')
        else:
            assigns.append(f'model_out({i0} downto {i1}) <= v{idx}({bw-1} downto {0});')
    return signals, assigns


def comb_logic_gen(sol: Solution, fn_name: str, print_latency: bool = False, timescale: str | None = None):
    inp_bits = sum(map(sum, map(_minimal_kif, sol.inp_qint)))
    out_bits = sum(map(sum, map(_minimal_kif, sol.out_qint)))

    neg_defined = set()
    ssa_signals, ssa_assigns = ssa_gen(sol, neg_defined=neg_defined, print_latency=print_latency)
    output_signals, output_assigns = output_gen(sol, neg_defined)
    blk = '\n    '

    code = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {fn_name} is port(
    model_inp:in std_logic_vector({inp_bits-1} downto {0});
    model_out:out std_logic_vector({out_bits-1} downto {0})
);
end entity {fn_name};

architecture rtl of {fn_name} is
    {blk.join(ssa_signals + output_signals)}


begin
    {blk.join(ssa_assigns + output_assigns)}

end architecture rtl;

"""
    return code
