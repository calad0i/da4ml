from itertools import accumulate

from ...cmvm.types import CascadedSolution, QInterval, Solution, _minimal_kif


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


def generate_io_wrapper(sol: Solution | CascadedSolution, module_name: str, pipelined: bool = False):
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

    clk_and_rst_inp, clk_and_rst_bind = '', ''
    if pipelined:
        clk_and_rst_inp = '\n   input clk,'
        clk_and_rst_bind = '\n        .clk(clk),'

    return f"""`timescale 1 ns / 1 ps

module {module_name}_wrapper ({clk_and_rst_inp}
    // verilator lint_off UNUSEDSIGNAL
    input [{w_reg_in-1}:0] inp,
    // verilator lint_on UNUSEDSIGNAL
    output [{w_reg_out-1}:0] out
);
    wire [{w_het_in-1}:0] packed_inp;
    wire [{w_het_out-1}:0] packed_out;

    {inp_assignment_str}

    {module_name} op ({clk_and_rst_bind}
        .inp(packed_inp),
        .out(packed_out)
    );

    {out_assignment_str}

endmodule
"""


def comb_binder_gen(sol: Solution, module_name: str):
    k_in, i_in, f_in = zip(*map(_minimal_kif, sol.inp_qint))
    k_out, i_out, f_out = zip(*map(_minimal_kif, sol.out_qint))
    max_inp_bw = max(k + i for k, i in zip(k_in, i_in)) + max(f_in)
    max_out_bw = max(k + i for k, i in zip(k_out, i_out)) + max(f_out)

    n_in, n_out = sol.shape
    return f"""#include "V{module_name}.h"
#include "ioutils.hh"
#include <verilated.h>

#ifdef _OPENMP
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

constexpr size_t N_inp = {n_in};
constexpr size_t N_out = {n_out};
constexpr size_t max_inp_bw = {max_inp_bw};
constexpr size_t max_out_bw = {max_out_bw};
typedef V{module_name} dut_t;

extern "C" {{

bool openmp_enabled() {{
    return _openmp;
}}

void _inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {{
    dut_t *dut = new dut_t;

    for (size_t i = 0; i < n_samples; ++i) {{
        write_input<N_inp, max_inp_bw>(dut->inp, &c_inp[i * N_inp]);
        dut->eval();
        read_output<N_out, max_out_bw>(dut->out, &c_out[i * N_out]);
    }}

    dut->final();
    delete dut;
}}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {{
    size_t n_max_threads = omp_get_max_threads();
    size_t n_samples_per_thread = std::max<size_t>(n_samples / n_max_threads, 32);
    size_t n_thread = n_samples / n_samples_per_thread;
    n_thread += (n_samples % n_samples_per_thread) ? 1 : 0;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < n_thread; ++i) {{
        size_t start = i * n_samples_per_thread;
        size_t end = std::min<size_t>(start + n_samples_per_thread, n_samples);
        size_t n_samples_this_thread = end - start;

        _inference(&c_inp[start * N_inp], &c_out[start * N_out], n_samples_this_thread);
    }}
    #else
    _inference(c_inp, c_out, n_samples);
    #endif
}}
}}"""


def pipeline_binder_gen(csol: CascadedSolution, module_name: str, II: int = 1, latency_multiplier: int = 1):
    k_in, i_in, f_in = zip(*map(_minimal_kif, csol.inp_qint))
    k_out, i_out, f_out = zip(*map(_minimal_kif, csol.out_qint))
    max_inp_bw = max(k + i for k, i in zip(k_in, i_in)) + max(f_in)
    max_out_bw = max(k + i for k, i in zip(k_out, i_out)) + max(f_out)

    latency = len(csol.solutions) * latency_multiplier

    n_in, n_out = csol.shape
    return f"""#include "V{module_name}.h"
#include "ioutils.hh"
#include <verilated.h>

#ifdef _OPENMP
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

constexpr size_t N_inp = {n_in};
constexpr size_t N_out = {n_out};
constexpr size_t max_inp_bw = {max_inp_bw};
constexpr size_t max_out_bw = {max_out_bw};
constexpr size_t II = {II};
constexpr size_t latency = {latency};
typedef V{module_name} dut_t;

extern "C" {{

bool openmp_enabled() {{
    return _openmp;
}}

void _inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {{
    dut_t *dut = new dut_t;

    size_t clk_req = n_samples * II + latency + 1;

    for (size_t t_inp = 0; t_inp < clk_req; ++t_inp) {{
        size_t t_out = t_inp - latency - 1;

        if (t_inp < n_samples * II && t_inp % II == 0) {{
            write_input<N_inp, max_inp_bw>(dut->inp, &c_inp[t_inp / II * N_inp]);
        }}

        dut->clk = 0;
        dut->eval();

        if (t_inp > latency && t_out % II == 0) {{
            read_output<N_out, max_out_bw>(dut->out, &c_out[t_out / II * N_out]);
        }}

        dut->clk = 1;
        dut->eval();
    }}

    dut->final();
    delete dut;
}}

void inference(int32_t *c_inp, int32_t *c_out, size_t n_samples) {{
#ifdef _OPENMP
    size_t n_max_threads = omp_get_max_threads();
    size_t n_samples_per_thread = std::max<size_t>(n_samples / n_max_threads, 32);
    size_t n_thread = n_samples / n_samples_per_thread;
    n_thread += (n_samples % n_samples_per_thread) ? 1 : 0;

    #pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < n_thread; ++i) {{
        size_t start = i * n_samples_per_thread;
        size_t end = std::min<size_t>(start + n_samples_per_thread, n_samples);
        size_t n_samples_this_thread = end - start;

        _inference(&c_inp[start * N_inp], &c_out[start * N_out], n_samples_this_thread);
    }}
#else
    _inference(c_inp, c_out, n_samples);
#endif
}}

}}"""
