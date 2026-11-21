"""Opt-in ternary-row checks: bit-cell recurrence and small formal miters (4x4 for 4-bit weights, 2x2 for
8-bit weights).

Requires da4ml installed. Yosys is required for exact bit-level RTL equality;
the local arithmetic/row truth-table checks run without a hardware mapper.
"""
from itertools import product
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from da4ml.codegen.rtl.verilog import comb_logic_gen
from da4ml.codegen.rtl.verilog.ternary_rows import _init, select_fusions
from da4ml.trace import FixedVariableArray, HWConfig, comb_trace
from da4ml.types import minimal_kif


def _bitcell_eval(values, widths, signed, shifts, signs, out_width):
    init = _init(signs)
    qprev = int(sum(s < 0 for s in signs) == 2)
    cin = int(any(s < 0 for s in signs))
    acc = 0
    for at in range(out_width):
        pins = []
        for v, w, is_signed, shift in zip(values, widths, signed, shifts):
            raw = v & ((1 << w) - 1)
            index = at - shift
            if index < 0:
                pins.append(0)
            elif index >= w:
                pins.append((raw >> (w - 1)) & 1 if is_signed else 0)
            else:
                pins.append((raw >> index) & 1)
        lut_addr = pins[0] | (pins[1] << 1) | (pins[2] << 2) | (qprev << 3)
        majority = (init >> lut_addr) & 1
        propagate = (init >> (32 + lut_addr)) & 1
        outbit = propagate ^ cin
        cin = cin if propagate else qprev
        acc |= outbit << at
        qprev = majority
    return acc


@pytest.mark.parametrize('signs', [(1, 1, 1), (1, 1, -1), (1, -1, -1), (-1, 1, 1), (-1, -1, 1)])
@pytest.mark.parametrize('signed', [(False, False, False), (True, True, True), (True, False, True)])
def test_row_bit_cells_exhaustive(signs, signed):
    widths = (2, 2, 2)
    shifts = (0, 1, 2)
    w = 8
    choices = [range(-2, 2) if sg else range(4) for sg in signed]
    for v in product(*choices):
        got = _bitcell_eval(v, widths, signed, shifts, signs, w)
        want = sum(s * (x << k) for s, x, k in zip(signs, v, shifts)) % (1 << w)
        assert got == want


def _random_solution(size, bitwidth, signed, seed):
    rng = np.random.default_rng(seed)
    if bitwidth == 8:
        A = rng.integers(129, 256, size=(size, size), dtype=np.int64)
    else:
        A = rng.integers(-8, 8, size=(size, size), dtype=np.int64)
    k = np.full(size, int(signed), dtype=np.int64)
    i = np.full(size, 7 if signed else 8, dtype=np.int64)
    f = np.zeros(size, dtype=np.int64)
    inp = FixedVariableArray.from_kif(
        k, i, f, hwconf=HWConfig(1, -1, -1), solver_options={'hard_dc': 0}
    )
    y = inp @ A.T.astype(np.float32)
    return comb_trace(inp, y)


@pytest.mark.parametrize('signed', [True, False])
@pytest.mark.parametrize('bitwidth', [4, 8])
@pytest.mark.parametrize('seed', [41, 97, 421])
def test_random_matrices_formal_miter(signed, bitwidth, seed, tmp_path):
    yosys = shutil.which('yosys')
    if yosys is None:
        pytest.skip('Yosys required for formal Verilog miter')
    # 8-bit weights: 2x2, since a 4x4 miter of 8-bit weights can exceed the SAT time limit
    sol = _random_solution(4 if bitwidth == 4 else 2, bitwidth, signed, seed)
    fused, _ = select_fusions(sol, [minimal_kif(op.qint) for op in sol.ops])
    if not fused:
        pytest.skip('da4ml generated no eligible single-use adder pair')
    (tmp_path / 'reference.v').write_text(comb_logic_gen(sol, 'reference'))
    (tmp_path / 'candidate.v').write_text(comb_logic_gen(sol, 'candidate', ternary_rows=True))
    width_in = sum(map(sum, map(minimal_kif, sol.inp_qint)))
    width_out = sum(map(sum, map(minimal_kif, sol.out_qint)))
    (tmp_path / 'miter.v').write_text(f"""module miter(
        input [{width_in - 1}:0] model_inp, output mismatch
    );
        wire [{width_out - 1}:0] r, c;
        reference u0 (.model_inp(model_inp), .model_out(r));
        candidate u1 (.model_inp(model_inp), .model_out(c));
        assign mismatch = |(r ^ c);
    endmodule
    """)
    src = Path(__file__).resolve().parents[1] / 'src/da4ml/codegen/rtl/verilog/source'
    command = (
        f'read_verilog -sv +/xilinx/cells_sim.v; '
        f'read_verilog -sv {src / "shift_adder.v"} {src / "negative.v"} '
        f'{tmp_path / "reference.v"} {tmp_path / "candidate.v"} {tmp_path / "miter.v"}; '
        'prep -top miter -flatten; '
        'sat -timeout 90 -verify -prove mismatch 0'
    )
    p = subprocess.run([yosys, '-Q', '-T', '-p', command], capture_output=True, text=True, timeout=110)
    assert p.returncode == 0, (p.stdout + '\n' + p.stderr)[-5000:]
    assert 'SUCCESS' in p.stdout, p.stdout[-3000:]
