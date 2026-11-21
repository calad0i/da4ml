"""Opt-in UltraScale+ three-operand carry rows for da4ml Verilog.

These rows merge a single-use pair of shift_adder operations.  Each row uses
one LUT6_2 per output bit and CARRY8 blocks, and spans the full output width,
so it uses more carry bits than the pair it replaces.  Default: disabled.
benchmarks/ternary_rows_yosys.py reports LUTs, carry cells and path lengths.

The transformation is intentionally conservative: it only fuses non-negative
integer shifts and leaves all other operations unchanged.  Numeric widths
and signed extensions follow the original precision data in CombLogic.
"""
from math import ceil


def _shift(op, kifs):
    return op.data + kifs[op.id0][2] - kifs[op.id1][2]


def select_fusions(sol, kifs):
    """Return {parent: [(source, shift, coefficient_sign), ...]}, consumed."""
    ops = sol.ops
    ref = sol.ref_count
    consumed = set()
    fused = {}
    for i in reversed(range(len(ops))):
        op = ops[i]
        if i in consumed or op.opcode not in (0, 1) or not ref[i]:
            continue
        parent_shift = _shift(op, kifs)
        if parent_shift < 0:
            continue
        for child_id in (op.id0, op.id1):
            if child_id < 0 or child_id in consumed or ref[child_id] != 1:
                continue
            child = ops[child_id]
            if child.opcode not in (0, 1):
                continue
            child_shift = _shift(child, kifs)
            if child_shift < 0:
                continue
            parent_sign = -1 if op.opcode == 1 else 1
            child_sign = -1 if child.opcode == 1 else 1
            if child_id == op.id0:
                terms = [(child.id0, 0, 1),
                         (child.id1, child_shift, child_sign),
                         (op.id1, parent_shift, parent_sign)]
            else:
                terms = [(op.id0, 0, 1),
                         (child.id0, parent_shift, parent_sign),
                         (child.id1, parent_shift + child_shift,
                          parent_sign * child_sign)]
            fused[i] = terms
            consumed.add(child_id)
            break
    return fused, consumed


def _init(signs):
    """INIT for LUT6_2: O5=majority(3), O6=xor(3) xor q_prev."""
    low = high = 0
    for pin in range(32):
        bits = [((pin >> j) & 1) ^ int(signs[j] < 0) for j in range(3)]
        q = (pin >> 3) & 1
        low |= int(sum(bits) >= 2) << pin
        high |= (bits[0] ^ bits[1] ^ bits[2] ^ q) << pin
    return (high << 32) | low


def _bit(src, shift, at, widths, kifs):
    j = at - shift
    if j < 0:
        return "1'b0"
    w = widths[src]
    if j >= w:
        return f'v{src}[{w - 1}]' if kifs[src][0] else "1'b0"
    return f'v{src}[{j}]'


def emit_ternary_row(node, out_width, terms, kifs, widths):
    """Return complete Verilog lines for one LUT6_2/CARRY8 ternary row.

    Adds terms modulo 2**out_width, as does the original shift_adder cascade.
    UltraScale+ primitive definitions are required for simulation/synthesis.
    """
    if len(terms) != 3 or out_width <= 0 or any(shift < 0 for _, shift, _ in terms):
        raise ValueError('invalid ternary-row parameters')
    signs = [sign for _, _, sign in terms]
    nneg = sum(sign < 0 for sign in signs)
    if nneg > 2:
        raise ValueError('three negative operands need separate handling')
    name = f'trn_{node}'
    result = [
        f'wire [{out_width - 1}:0] {name}_s, {name}_d, {name}_q, {name}_o;',
    ]
    init = _init(signs)
    q0 = 1 if nneg == 2 else 0
    ci = 1 if nneg else 0
    for bit in range(out_width):
        inputs = [_bit(s, k, bit, widths, kifs) for s, k, _ in terms]
        qprev = f'{name}_q[{bit - 1}]' if bit else f"1'b{q0}"
        result.append(f'assign {name}_d[{bit}] = {qprev};')
        result.append(
            f'LUT6_2 #(.INIT(64\'h{init:016X})) {name}_lut{bit} ('
            f'.O6({name}_s[{bit}]), .O5({name}_q[{bit}]), '
            f'.I0({inputs[0]}), .I1({inputs[1]}), .I2({inputs[2]}), '
            f'.I3({qprev}), .I4(1\'b0), .I5(1\'b1));'
        )
    for row in range(ceil(out_width / 8)):
        lo = row * 8
        n = min(8, out_width - lo)
        pad = 8 - n
        def port(vec):
            bits = [f'{name}_{vec}[{lo + x}]' for x in range(n)]
            return '{' + ', '.join(["1'b0"] * pad + list(reversed(bits))) + '}'
        cin = f"1'b{ci}" if row == 0 else f'{name}_co{row - 1}[7]'
        result.append(f'wire [7:0] {name}_co{row}, {name}_out{row};')
        result.append(
            f'CARRY8 #(.CARRY_TYPE("SINGLE_CY8")) {name}_cy{row} ('
            f'.CO({name}_co{row}), .O({name}_out{row}), '
            f'.CI({cin}), .CI_TOP(1\'b0), '
            f'.DI({port("d")}), .S({port("s")}));'
        )
        result.append(f'assign {name}_o[{lo + n - 1}:{lo}] = {name}_out{row}[{n - 1}:0];')
    result.append(f'assign v{node} = {name}_o;')
    return result
