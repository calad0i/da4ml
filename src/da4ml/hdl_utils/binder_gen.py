from ..cmvm.fixed_variable import FixedVariable


def binder_gen(module_name: str, inp: list[FixedVariable], out: list[FixedVariable]):
    max_inp_bw = max(_inp.k for _inp in inp) + max(_inp.i for _inp in inp) + max(_inp.shift for _inp in inp)
    max_out_bw = max(_out.k for _out in out) + max(_out.i for _out in out) + max(_out.shift for _out in out)
    return f"""#include "V{module_name}.h"
    #include "ioutils.hh"
    #include <iostream>
    #include <verilated.h>

    constexpr size_t N_in = {len(inp)};
    constexpr size_t N_out = {len(out)};
    constexpr size_t n_clks = 0;
    constexpr size_t max_inp_bw = {max_inp_bw};
    constexpr size_t max_out_bw = {max_out_bw};

    extern "C" {{
    void test(int32_t *c_inp, int32_t *c_out) {{
        V{module_name} *dut = new V{module_name};

        write_input<N_in, max_inp_bw>(dut->in, c_inp);

        for (size_t i = 0; i < n_clks; ++i) {{
            dut->clk = 0;
            dut->eval();
            dut->clk = 1;
            dut->eval();
            dut->clk = 0;
        }}
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
