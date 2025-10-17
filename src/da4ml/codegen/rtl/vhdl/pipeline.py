from ....cmvm.types import Pipeline, _minimal_kif
from .comb import comb_logic_gen


def pipeline_logic_gen(
    csol: Pipeline,
    name: str,
    print_latency=False,
    timescale: str | None = None,
    register_layers: int = 1,
):
    N = len(csol.solutions)
    inp_bits = [sum(map(sum, map(_minimal_kif, sol.inp_qint))) for sol in csol.solutions]
    out_bits = inp_bits[1:] + [sum(map(sum, map(_minimal_kif, csol.out_qint)))]

    registers = [f'signal stage{i}_inp:std_logic_vector({width - 1} downto 0);' for i, width in enumerate(inp_bits)]
    for i in range(0, register_layers - 1):
        registers += [f'signal stage{j}_inp_copy{i}:std_logic_vector({width - 1} downto 0);' for j, width in enumerate(inp_bits)]
    wires = [f'signal stage{i}_out:std_logic_vector({width - 1} downto 0);' for i, width in enumerate(out_bits)]

    comb_logic = [
        f'stage{i}:entity work.{name}_stage{i} port map(model_inp=>stage{i}_inp,model_out=>stage{i}_out);' for i in range(N)
    ]

    if register_layers == 1:
        serial_logic = ['stage0_inp <= model_inp;']
        serial_logic += [f'stage{i}_inp <= stage{i - 1}_out;' for i in range(1, N)]
    else:
        serial_logic = ['stage0_inp_copy0 <= model_inp;']
        for j in range(1, register_layers - 1):
            serial_logic.append(f'stage0_inp_copy{j} <= stage0_inp_copy{j - 1};')
        serial_logic.append(f'stage0_inp <= stage0_inp_copy{register_layers - 2};')
        for i in range(1, N):
            serial_logic.append(f'stage{i}_inp_copy0 <= stage{i - 1}_out;')
            for j in range(1, register_layers - 1):
                serial_logic.append(f'stage{i}_inp_copy{j} <= stage{i}_inp_copy{j - 1};')
            serial_logic.append(f'stage{i}_inp <= stage{i}_inp_copy{register_layers - 2};')

    serial_logic += [f'model_out <= stage{N - 1}_out;']

    blk = '\n    '

    module = f"""library ieee;
use ieee.std_logic_1164.all;
entity {name} is port(
    clk:in std_logic;
    model_inp:in std_logic_vector({inp_bits[0] - 1} downto 0);
    model_out:out std_logic_vector({out_bits[-1] - 1} downto 0));
end entity {name};

architecture rtl of {name} is
    {blk.join(registers)}
    {blk.join(wires)}

begin
    {blk.join(comb_logic)}

    process(clk) begin
        if rising_edge(clk) then
            {f'{blk}        '.join(serial_logic)}
        end if;
    end process;
end architecture rtl;
"""

    ret: dict[str, str] = {}
    for i, s in enumerate(csol.solutions):
        stage_name = f'{name}_stage{i}'
        ret[stage_name] = comb_logic_gen(s, stage_name, print_latency=print_latency, timescale=timescale)
    ret[name] = module
    return ret
