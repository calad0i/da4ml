from ....types import Pipeline, minimal_kif
from .comb import comb_logic_gen


def pipeline_logic_gen(
    csol: Pipeline,
    name: str,
    print_latency=False,
    timescale: str | None = None,
    comb_logic_gen_fn=None,
    no_shreg: bool = False,
):
    comb_logic_gen_fn = comb_logic_gen_fn or comb_logic_gen
    N = len(csol.solutions)
    inp_bits = [sum(map(sum, map(minimal_kif, sol.inp_qint))) for sol in csol.solutions]
    out_bits = inp_bits[1:] + [sum(map(sum, map(minimal_kif, csol.out_qint)))]

    registers = [f'signal stage{i}_inp:std_logic_vector({width - 1} downto 0);' for i, width in enumerate(inp_bits)]
    wires = [f'signal stage{i}_out:std_logic_vector({width - 1} downto 0);' for i, width in enumerate(out_bits)]

    shreg_attr = ()
    if no_shreg:
        shreg_attr = ['attribute shreg_extract:string;']
        shreg_attr += [f'attribute shreg_extract of stage{i}_inp:signal is "no";' for i in range(N)]

    comb_logic = [
        f'stage{i}:entity work.{name}_stage{i} port map(model_inp=>stage{i}_inp,model_out=>stage{i}_out);' for i in range(N)
    ]

    serial_logic = ['stage0_inp <= model_inp;']
    serial_logic += [f'stage{i}_inp <= stage{i - 1}_out;' for i in range(1, N)]

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
    {blk.join(shreg_attr) if shreg_attr else ''}
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
        ret[stage_name] = comb_logic_gen_fn(s, stage_name, print_latency=print_latency, timescale=timescale)
    ret[name] = module
    return ret
