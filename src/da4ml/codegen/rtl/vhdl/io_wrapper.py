from itertools import accumulate

from ....cmvm.types import CombLogic, Pipeline, QInterval, _minimal_kif


def _loc(i: int, j: int):
    return f'({i} downto {j})' if i != j else f'({i})'


def hetero_io_map(qints: list[QInterval], merge: bool = False):
    N = len(qints)
    ks, _is, fs = zip(*map(_minimal_kif, qints))
    Is = [_i + _k for _i, _k in zip(_is, ks)]
    max_I, max_f = max(_is) + max(ks), max(fs)
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

    mask = list(high < low for high, low in hetero)
    regular = [r for r, m in zip(regular, mask) if not m]
    hetero = [h for h, m in zip(hetero, mask) if not m]

    if not merge:
        return regular, hetero, pads, (width_regular, width_packed)

    # Merging consecutive intervals when possible
    NN = len(regular) - 2
    for i in range(NN, -1, -1):
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


def generate_io_wrapper(sol: CombLogic | Pipeline, module_name: str, pipelined: bool = False):
    reg_in, het_in, _, shape_in = hetero_io_map(sol.inp_qint, merge=True)
    reg_out, het_out, pad_out, shape_out = hetero_io_map(sol.out_qint, merge=True)

    w_reg_in, w_het_in = shape_in
    w_reg_out, w_het_out = shape_out

    inp_assignment = [f'packed_inp{_loc(ih, jh)} <= model_inp{_loc(ir, jr)};' for (ih, jh), (ir, jr) in zip(het_in, reg_in)]
    _out_assignment: list[tuple[int, str]] = []

    for i, ((ih, jh), (ir, jr)) in enumerate(zip(het_out, reg_out)):
        if ih == jh - 1:
            continue
        _out_assignment.append((ih, f'model_out{_loc(ir, jr)} <= packed_out{_loc(ih, jh)};'))

    for i, (i, j, copy_from) in enumerate(pad_out):
        n_bit = i - j + 1
        value = "'0'" if copy_from == -1 else f'packed_out({copy_from})'
        pad = f'(others => {value})' if n_bit > 1 else value
        _out_assignment.append((i, f'model_out{_loc(i, j)} <= {pad};'))
    _out_assignment.sort(key=lambda x: x[0])
    out_assignment = [v for _, v in _out_assignment]

    inp_assignment_str = '\n    '.join(inp_assignment)
    out_assignment_str = '\n    '.join(out_assignment)

    clk_and_rst_inp, clk_and_rst_bind = '', ''
    if pipelined:
        clk_and_rst_inp = '\n    clk:in std_logic;'
        clk_and_rst_bind = '\n        clk=>clk,'

    return f"""library ieee;
use ieee.std_logic_1164.all;
entity {module_name}_wrapper is port({clk_and_rst_inp}
    model_inp:in std_logic_vector({w_reg_in - 1} downto {0});
    model_out:out std_logic_vector({w_reg_out - 1} downto {0})
);
end entity {module_name}_wrapper;

architecture rtl of {module_name}_wrapper is
    signal packed_inp:std_logic_vector({w_het_in - 1} downto {0});
    signal packed_out:std_logic_vector({w_het_out - 1} downto {0});

begin
    {inp_assignment_str}

    op:entity work.{module_name} port map({clk_and_rst_bind}
        model_inp=>packed_inp,
        model_out=>packed_out
    );

    {out_assignment_str}

end architecture rtl;
"""
