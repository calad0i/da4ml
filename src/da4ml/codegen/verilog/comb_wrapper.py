from itertools import accumulate

from ...cmvm.types import QInterval, Solution, _minimal_kif


def hetero_io_map(qints: list[QInterval], merge: bool = False):
    N = len(qints)
    ks, _is, fs = zip(*map(_minimal_kif, qints))
    Is = [_i + _k for _i, _k in zip(_is, ks)]
    max_I, max_f = max(Is), max(fs)
    max_bw = max_I + max_f
    width_regular, width_packed = max_bw * N, sum(Is) + sum(fs)

    regular: list[tuple[int, int]] = []
    pads: list[tuple[int, int, int]] = []

    bws = [I + f for I, f in zip(Is, fs)]
    _bw = list(accumulate([0] + bws))
    hetero = [(i - 1, j) for i, j in zip(_bw[1:], _bw[:-1])]

    for i in range(N):
        base = max_bw * i
        bias_low = max_f - fs[i]
        bias_high = max_I - Is[i]
        low = base + bias_low
        high = (base + max_bw - 1) - bias_high
        regular.append((high, low))

        if bias_low != 0:
            pads.append((base + bias_low - 1, base, -1))
        if bias_high != 0:
            copy_from = hetero[i][0] if ks[i] else -1
            pads.append((base + max_bw - 1, base + max_bw - bias_high, copy_from))

    if not merge:
        return regular, hetero, pads, (width_regular, width_packed)

    # Merging consecutive intervals when possible
    for i in range(N - 2, -1, -1):
        this_high = regular[i][0]
        next_low = regular[i + 1][1]
        if next_low - this_high != 1:
            continue
        regular[i] = (regular[i + 1][0], regular[i][1])
        regular.pop(i + 1)
        hetero[i] = (hetero[i + 1][0], hetero[i][1])
        hetero.pop(i + 1)

    for i in range(len(pads) - 2, -1, -1):
        if pads[i + 1][1] - pads[i][0] == 1 and pads[i][2] == pads[i + 1][2]:
            pads[i] = (pads[i + 1][0], pads[i][1], pads[i][2])
            pads.pop(i + 1)

    return regular, hetero, pads, (width_regular, width_packed)


def generate_io_wrapper(module_name: str, sol: Solution):
    reg_in, het_in, _, shape_in = hetero_io_map(sol.inp_qint, merge=True)
    reg_out, het_out, pad_out, shape_out = hetero_io_map(sol.out_qint, merge=True)

    w_reg_in, w_het_in = shape_in
    w_reg_out, w_het_out = shape_out

    inp_assignment = [f'assign packed_inp[{ih}:{jh}] = inp[{ir}:{jr}];' for (ih, jh), (ir, jr) in zip(het_in, reg_in)]
    _out_assignment: list[tuple[int, str]] = []

    for i, ((ih, jh), (ir, jr)) in enumerate(zip(het_out, reg_out)):
        _out_assignment.append((ih, f'assign out[{ir}:{jr}] = packed_out[{ih}:{jh}];'))

    for i, (i, j, copy_from) in enumerate(pad_out):
        n_bit = i - j + 1
        pad = f"{n_bit}'b0" if copy_from == -1 else f'{{{n_bit}{{packed_out[{copy_from}]}}}}'
        _out_assignment.append((i, f'assign out[{i}:{j}] = {pad};'))
    _out_assignment.sort(key=lambda x: x[0])
    out_assignment = [v for _, v in _out_assignment]

    inp_assignment_str = '\n    '.join(inp_assignment)
    out_assignment_str = '\n    '.join(out_assignment)

    return f"""`timescale 1 ns / 1 ps

module {module_name}_wrapper (
    input [{w_reg_in-1}:0] inp,
    output [{w_reg_out-1}:0] out
);
    wire [{w_het_in-1}:0] packed_inp;
    wire [{w_het_out-1}:0] packed_out;

    {inp_assignment_str}

    {module_name} op (
        .inp(packed_inp),
        .out(packed_out)
    );

    {out_assignment_str}

endmodule
"""


def binder_gen(module_name: str, sol: Solution):
    max_inp_bw = sum(map(max, zip(*map(_minimal_kif, sol.inp_qint))))
    max_out_bw = sum(map(max, zip(*map(_minimal_kif, sol.out_qint))))

    n_in, n_out = sol.shape
    return f"""#include "V{module_name}.h"
    #include "ioutils.hh"
    #include <iostream>
    #include <verilated.h>

    constexpr size_t N_in = {n_in};
    constexpr size_t N_out = {n_out};
    constexpr size_t max_inp_bw = {max_inp_bw};
    constexpr size_t max_out_bw = {max_out_bw};

    extern "C" {{
    void test(int32_t *c_inp, int32_t *c_out) {{
        V{module_name} *dut = new V{module_name};

        write_input<N_in, max_inp_bw>(dut->inp, c_inp);

        dut->eval();

        std::vector<int32_t> output = read_output<N_out, max_out_bw>(dut->out);
        for (size_t i = 0; i < output.size(); ++i) {{
            c_out[i] = output[i];
        }}
        // Clean up

        dut->final();
        delete dut;
    }}
    }}"""
