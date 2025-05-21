from ...cmvm.types import CascadedSolution, _minimal_kif
from .comb import comb_logic_gen


def pipeline_logic_gen(
    csol: CascadedSolution, name: str, print_latency=False, timescale: str | None = '`timescale 1 ns / 1 ps', reset_high=True
):
    N = len(csol.solutions)
    inp_bits = [sum(map(sum, map(_minimal_kif, sol.inp_qint))) for sol in csol.solutions]
    out_bits = inp_bits[1:] + [sum(map(sum, map(_minimal_kif, csol.out_qint)))]

    registers = [f'reg [{width}-1:0] stage{i}_inp;' for i, width in enumerate(inp_bits)]
    wires = [f'wire [{width}-1:0] stage{i}_out;' for i, width in enumerate(out_bits)]

    comb_logic = [f'{name}_stage{i} stage{i} (.inp(stage{i}_inp), .out(stage{i}_out));' for i in range(N)]

    serial_logic = ['stage0_inp <= inp;']
    serial_logic += [f'stage{i}_inp <= stage{i-1}_out;' for i in range(1, N)]
    serial_logic += [f'out <= stage{N-1}_out;']

    sep0 = '\n    '
    sep1 = '\n        '

    module = f"""module {name} (
    input clk,
    input [{inp_bits[0]-1}:0] inp,
    output reg [{out_bits[-1]-1}:0] out
);

    {sep0.join(registers)}
    {sep0.join(wires)}

    {sep0.join(comb_logic)}

    always @(posedge clk) begin
        {sep1.join(serial_logic)}
    end
endmodule
"""

    if timescale:
        module = f'{timescale}\n\n{module}'

    ret: dict[str, str] = {}
    for i, s in enumerate(csol.solutions):
        stage_name = f'{name}_stage{i}'
        ret[stage_name] = comb_logic_gen(s, stage_name, print_latency=print_latency, timescale=timescale)
    ret[name] = module
    return ret
