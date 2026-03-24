from math import ceil, log2

import numpy as np
from xls import FunctionBuilder, Package, Value
from xls.c_api import optimize_ir, parse_ir_package
from xls.ir_builder import BuilderBase, BValue, Function

from ...types import CombLogic, minimal_kif


def _extend(bb: BuilderBase, val: BValue, old_bw: int, new_bw: int, is_signed: bool) -> BValue:
    if new_bw <= old_bw:
        return val
    if is_signed:
        return bb.add_sign_extend(val, new_bw)
    else:
        return bb.add_zero_extend(val, new_bw)


def _literal(bb: BuilderBase, bw: int, val: int) -> BValue:
    return bb.add_literal(Value.make_ubits(bw, val & ((1 << bw) - 1)))


def _shift_pad(bb: BuilderBase, val: BValue, shift: int, old_bw: int, new_bw: int, is_signed: bool):
    assert shift >= 0
    if shift > 0:
        zero_pad = _literal(bb, shift, 0)
        val = bb.add_concat([val, zero_pad])
    return _extend(bb, val, old_bw + shift, new_bw, is_signed)


def shift_adder(
    bb: BuilderBase,
    v0: BValue,
    v1: BValue,
    bw0: int,
    bw1: int,
    s0: int,
    s1: int,
    bw_out: int,
    drop_lsbs: int,
    shift: int,
    is_sub: int,
):
    in0_need = bw0 - shift if shift < 0 else bw0
    in1_need = bw1 + shift if shift > 0 else bw1
    extra_pad = (is_sub + 1) if (s0 != s1) else (is_sub + 0)
    bw_add = max(in0_need, in1_need) + extra_pad + 1

    shift0, shift1 = max(0, -shift), max(0, shift)
    v0 = _shift_pad(bb, v0, shift0, bw0, bw_add, bool(s0))
    v1 = _shift_pad(bb, v1, shift1, bw1, bw_add, bool(s1))

    if is_sub:
        accum = bb.add_sub(v0, v1)
    else:
        accum = bb.add_add(v0, v1)

    return bb.add_bit_slice(accum, drop_lsbs, bw_out)


def negate(bb: BuilderBase, val: BValue, bw_in: int, bw_out: int, in_signed: bool):
    if bw_in < bw_out:
        val_ext = _extend(bb, val, bw_in, bw_out, in_signed)
        return bb.add_negate(val_ext)
    else:
        neg = bb.add_negate(val)
        return bb.add_bit_slice(neg, 0, bw_out)


def _build_core_ops(bb, pkg, sol, model_inp, inp_starts, inp_widths, kifs, widths):
    """Build the core computation ops from a flat-bits input BValue.

    Returns a list of (BValue, bit_width) for each output.
    """
    ops = sol.ops
    ref_count = sol.ref_count
    buf: list[BValue] = [None] * len(ops)  # type: ignore

    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            continue

        bw = widths[i]
        if bw == 0:
            continue

        match op.opcode:
            case -2:  # Negation
                bw0 = widths[op.id0]
                is_signed = ops[op.id0].qint.min < 0
                buf[i] = negate(bb, buf[op.id0], bw0, bw, is_signed)

            case -1:  # Input
                start = inp_starts[op.id0]
                buf[i] = bb.add_bit_slice(model_inp, start, bw)

            case 0 | 1:  # Add/Sub with shift
                p0, p1 = kifs[op.id0], kifs[op.id1]
                bw0, bw1 = widths[op.id0], widths[op.id1]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                shift = op.data + f0 - f1
                dlsbs = max(f0, f1 - op.data) - kifs[i][2]

                buf[i] = shift_adder(
                    bb,
                    buf[op.id0],
                    buf[op.id1],
                    bw0,
                    bw1,
                    s0,
                    s1,
                    bw,
                    dlsbs,
                    shift,
                    op.opcode,
                )

            case 2:  # ReLU
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                v0 = buf[op.id0]
                bw0 = widths[op.id0]

                if ops[op.id0].qint.min < 0:
                    # Signed: zero if negative (MSB=1)
                    msb = bb.add_bit_slice(v0, bw0 - 1, 1)
                    sliced = bb.add_bit_slice(v0, lsb_bias, bw)
                    zero = _literal(bb, bw, 0)
                    buf[i] = bb.add_select(msb, [sliced, zero])
                else:
                    buf[i] = bb.add_bit_slice(v0, lsb_bias, bw)

            case 3:  # Quantize (bit slice with possible sign extension)
                lsb_bias = kifs[op.id0][2] - kifs[i][2]
                i0 = bw + lsb_bias - 1
                v0 = buf[op.id0]
                bw0 = widths[op.id0]

                if i0 >= bw0:
                    assert ops[op.id0].qint.min < 0
                    v0_ext = bb.add_sign_extend(v0, i0 + 1)
                    buf[i] = bb.add_bit_slice(v0_ext, lsb_bias, bw)
                else:
                    buf[i] = bb.add_bit_slice(v0, lsb_bias, bw)

            case 4:  # Const addition
                val = ((op.data & 0xFFFFFFFF) + 0x80000000) % 0x100000000 - 0x80000000
                f1 = (((op.data >> 32) & 0xFFFFFFFF) + 0x80000000) % 0x100000000 - 0x80000000
                bw0 = widths[op.id0]
                bw1 = max(1, int(ceil(log2(abs(val) + 1))))
                s0 = int(kifs[op.id0][0])
                f0 = kifs[op.id0][2]
                s1 = int(val < 0)
                shift = f0 - f1
                dlsbs = max(f0, f1) - kifs[i][2]

                abs_val = abs(val)
                const_lit = _literal(bb, bw1, abs_val)

                buf[i] = shift_adder(
                    bb,
                    buf[op.id0],
                    const_lit,
                    bw0,
                    bw1,
                    s0,
                    0,
                    bw,
                    dlsbs,
                    shift,
                    s1,
                )

            case 5:  # Const definition
                num = op.data
                if num < 0:
                    num = (1 << bw) + num
                buf[i] = _literal(bb, bw, num)

            case 6:  # MSB Mux
                id_c = op.data & 0xFFFFFFFF
                a_idx, b_idx = op.id0, op.id1
                p0, p1 = kifs[a_idx], kifs[b_idx]
                bwk, bw0, bw1 = widths[id_c], widths[a_idx], widths[b_idx]
                s0, f0, s1, f1 = int(p0[0]), p0[2], int(p1[0]), p1[2]
                fo = kifs[i][2]
                _shift_raw = (op.data >> 32) & 0xFFFFFFFF
                _shift_raw = _shift_raw if _shift_raw < 0x80000000 else _shift_raw - 0x100000000
                shift1 = fo - f1 + _shift_raw
                shift0 = fo - f0
                assert shift0 == 0 or shift1 == 0
                shift = shift1 * (shift1 > 0) - shift0 * (shift0 > 0)
                inv = 0

                in0_need = bw0 - shift if shift < 0 else bw0
                in1_need = bw1 + shift if shift > 0 else bw1
                extra_pad = (inv + 1) if (s0 != s1) else (inv + 0)
                bw_buf = max(in0_need, in1_need) + extra_pad

                va = buf[a_idx] if bw0 > 0 else _literal(bb, 1, 0)
                vb = buf[b_idx] if bw1 > 0 else _literal(bb, 1, 0)
                _bw0 = bw0 if bw0 > 0 else 1
                _bw1 = bw1 if bw1 > 0 else 1

                if shift < 0:
                    pad_r = -shift
                    zero_pad = _literal(bb, pad_r, 0)
                    va = bb.add_concat([va, zero_pad])
                    _bw0 += pad_r
                va_ext = _extend(bb, va, _bw0, bw_buf, bool(s0))

                if shift > 0:
                    pad_r = shift
                    zero_pad = _literal(bb, pad_r, 0)
                    vb = bb.add_concat([vb, zero_pad])
                    _bw1 += pad_r
                vb_ext = _extend(bb, vb, _bw1, bw_buf, bool(s1))

                msb = bb.add_bit_slice(buf[id_c], bwk - 1, 1)
                va_out = bb.add_bit_slice(va_ext, 0, bw)
                vb_out = bb.add_bit_slice(vb_ext, 0, bw)
                buf[i] = bb.add_select(msb, [vb_out, va_out])

            case 7:  # Multiplication
                bw0, bw1 = widths[op.id0], widths[op.id1]
                s0, s1 = int(kifs[op.id0][0]), int(kifs[op.id1][0])
                bw_prod = bw0 + bw1

                v0, v1 = buf[op.id0], buf[op.id1]

                if s0 and s1:
                    v0_ext = bb.add_sign_extend(v0, bw_prod)
                    v1_ext = bb.add_sign_extend(v1, bw_prod)
                    prod = bb.add_smul(v0_ext, v1_ext)
                elif s0 and not s1:
                    v0_ext = bb.add_sign_extend(v0, bw_prod)
                    v1_ext = bb.add_zero_extend(v1, bw_prod)
                    prod = bb.add_smul(v0_ext, v1_ext)
                elif not s0 and s1:
                    v0_ext = bb.add_zero_extend(v0, bw_prod)
                    v1_ext = bb.add_sign_extend(v1, bw_prod)
                    prod = bb.add_smul(v0_ext, v1_ext)
                else:
                    v0_ext = bb.add_zero_extend(v0, bw_prod)
                    v1_ext = bb.add_zero_extend(v1, bw_prod)
                    prod = bb.add_umul(v0_ext, v1_ext)

                buf[i] = bb.add_bit_slice(prod, 0, bw)

            case 8:  # Lookup table
                assert sol.lookup_tables is not None
                table = sol.lookup_tables[op.data]
                out_bw = bw
                padded = table.padded_table(ops[op.id0].qint)

                elem_type = pkg.get_bits_type(out_bw)
                elements = []
                for v in padded:
                    if np.isnan(v):
                        elements.append(_literal(bb, out_bw, 0))
                    else:
                        elements.append(_literal(bb, out_bw, int(v) & ((1 << out_bw) - 1)))

                arr = bb.add_array(elem_type, elements)
                buf[i] = bb.add_array_index(arr, [buf[op.id0]])

            case 9:  # Unary bitwise
                v0 = buf[op.id0]
                match op.data:
                    case 0:
                        buf[i] = bb.add_not(v0)
                    case 1:
                        buf[i] = bb.add_or_reduce(v0)
                    case 2:
                        buf[i] = bb.add_and_reduce(v0)
                    case _:
                        raise ValueError(f'Unknown unary bitwise op {op.data}')

            case 10:  # Binary bitwise
                data = op.data
                subop = (data >> 56) & 0xFF
                _shift = data & 0xFFFFFFFF
                shift = (_shift + 0x80000000) % 0x100000000 - 0x80000000 + kifs[op.id0][2] - kifs[op.id1][2]

                bw0, bw1 = widths[op.id0], widths[op.id1]
                s0 = int(ops[op.id0].qint.min < 0)
                s1 = int(ops[op.id1].qint.min < 0)

                in0_need = bw0 - shift if shift < 0 else bw0
                in1_need = bw1 + shift if shift > 0 else bw1
                extra_pad = 1 if (s0 != s1) else 0
                bw_tmp = max(in0_need, in1_need) + extra_pad

                v0 = buf[op.id0]
                v1 = buf[op.id1]

                if shift < 0:
                    pad_r = -shift
                    zero_pad = _literal(bb, pad_r, 0)
                    v0 = bb.add_concat([v0, zero_pad])
                    bw0_eff = bw0 + pad_r
                else:
                    bw0_eff = bw0
                v0_ext = _extend(bb, v0, bw0_eff, bw_tmp, bool(s0))

                if shift > 0:
                    pad_r = shift
                    zero_pad = _literal(bb, pad_r, 0)
                    v1 = bb.add_concat([v1, zero_pad])
                    bw1_eff = bw1 + pad_r
                else:
                    bw1_eff = bw1
                v1_ext = _extend(bb, v1, bw1_eff, bw_tmp, bool(s1))

                match subop:
                    case 0:
                        result = bb.add_and(v0_ext, v1_ext)
                    case 1:
                        result = bb.add_or(v0_ext, v1_ext)
                    case 2:
                        result = bb.add_xor(v0_ext, v1_ext)
                    case _:
                        raise ValueError(f'Unknown binary bitwise subop {subop}')

                buf[i] = bb.add_bit_slice(result, 0, bw)

            case _:
                raise ValueError(f'Unknown opcode {op.opcode}')

    return buf


def _sol_io_params(sol: CombLogic):
    """Compute I/O parameters from a CombLogic solution."""
    ops = sol.ops
    kifs = list(map(minimal_kif, (op.qint for op in ops)))
    widths: list[int] = list(map(sum, kifs))

    inp_kifs = [minimal_kif(qint) for qint in sol.inp_qint]
    inp_widths = list(map(sum, inp_kifs))
    _inp_widths = np.cumsum([0] + inp_widths)
    inp_starts = _inp_widths[:-1].tolist()
    total_inp_bits = int(_inp_widths[-1])

    out_kifs = [minimal_kif(qint) for qint in sol.out_qint]
    out_widths = [sum(k) for k in out_kifs]
    total_out_bits = sum(out_widths)

    return kifs, widths, inp_kifs, inp_widths, inp_starts, total_inp_bits, out_kifs, out_widths, total_out_bits


def build_xls_function(sol: CombLogic, fn_name: str = 'dais_fn') -> tuple[Package, Function]:
    kifs, widths, inp_kifs, inp_widths, inp_starts, total_inp_bits, out_kifs, out_widths, total_out_bits = _sol_io_params(sol)

    if total_inp_bits == 0 or total_out_bits == 0:
        raise ValueError('Cannot build XLS function with zero-width I/O')

    pkg = Package.create(fn_name)
    fb = FunctionBuilder.create(fn_name, pkg)
    bb = fb.as_builder_base()

    inp_type = pkg.get_bits_type(total_inp_bits)
    model_inp = fb.add_parameter('model_inp', inp_type)

    buf = _build_core_ops(bb, pkg, sol, model_inp, inp_starts, inp_widths, kifs, widths)

    out_parts = []
    for idx_i, out_idx in enumerate(sol.out_idxs):
        obw = out_widths[idx_i]
        if obw == 0:
            continue
        out_parts.append(buf[out_idx])

    if len(out_parts) == 0:
        raise ValueError('No output parts')

    if len(out_parts) == 1:
        ret_val = bb.add_identity(out_parts[0], name='model_out')
    else:
        ret_val = bb.add_concat(out_parts[::-1], name='model_out')

    _ = fb.build_with_return_value(ret_val)
    ir_str = pkg.to_string()
    pkg_opt = parse_ir_package(optimize_ir(ir_str, top=fn_name))
    fn_opt = pkg_opt.get_function(fn_name)
    return pkg_opt, fn_opt


def build_xls_io_wrapper(sol: CombLogic, fn_name: str = 'dais_fn') -> tuple[Package, Function]:
    """Build an XLS function with unpacked array I/O.

    Input:  bits[max_inp_bw][N_inp]  — array of uniform-width input elements
    Output: bits[max_out_bw][N_out]  — array of uniform-width output elements

    Uses RTL-style uniform bit layout: max_bw = max(k) + max(i) + max(f).
    Each element is LSB-aligned at bit offset max(f) - f_j within its slot.
    """
    kifs, widths, inp_kifs, inp_widths, inp_starts, total_inp_bits, out_kifs, out_widths, total_out_bits = _sol_io_params(sol)
    inp_size, out_size = sol.shape

    if total_inp_bits == 0 or total_out_bits == 0:
        raise ValueError('Cannot build XLS function with zero-width I/O')

    # RTL-style uniform bit widths: max(k) + max(i) + max(f)
    inp_ks, inp_is, inp_fs = zip(*inp_kifs)
    max_inp_bw = max(inp_ks) + max(inp_is) + max(inp_fs)
    max_f_inp = max(inp_fs)

    out_ks, out_is, out_fs = zip(*out_kifs)
    max_out_bw = max(out_ks) + max(out_is) + max(out_fs)
    max_f_out = max(out_fs)

    pkg = Package.create(fn_name)
    fb = FunctionBuilder.create(fn_name, pkg)
    bb = fb.as_builder_base()

    # Input: array of bits[max_inp_bw], length N_inp
    inp_elem_type = pkg.get_bits_type(max_inp_bw)
    inp_arr_type = pkg.get_array_type(inp_elem_type, inp_size)
    inp_arr = fb.add_parameter('model_inp', inp_arr_type)

    # Extract each element from input array at the correct LSB offset
    idx_bw = max(1, int(ceil(log2(max(inp_size, 2)))))
    inp_sliced = []
    for j in range(inp_size):
        idx_lit = _literal(bb, idx_bw, j)
        elem = bb.add_array_index(inp_arr, [idx_lit])
        ibw = inp_widths[j]
        if ibw == 0:
            continue
        bias_low = max_f_inp - inp_fs[j]
        elem = bb.add_bit_slice(elem, bias_low, ibw)
        inp_sliced.append(elem)

    if len(inp_sliced) == 1:
        flat_inp = inp_sliced[0]
    else:
        flat_inp = bb.add_concat(inp_sliced[::-1])

    # Run core computation on the flat bits
    buf = _build_core_ops(bb, pkg, sol, flat_inp, inp_starts, inp_widths, kifs, widths)

    # Assemble output array: place each output at the correct LSB offset
    out_parts_raw = []
    for idx_i, out_idx in enumerate(sol.out_idxs):
        obw = out_widths[idx_i]
        out_parts_raw.append((buf[out_idx], obw, out_kifs[idx_i]))

    if len(out_parts_raw) == 0:
        raise ValueError('No output parts')

    out_elem_type = pkg.get_bits_type(max_out_bw)
    out_elems = []
    for v, obw, kif in out_parts_raw:
        if obw == 0:
            out_elems.append(_literal(bb, max_out_bw, 0))
            continue
        k_j, _, f_j = kif
        bias_low = max_f_out - f_j
        is_signed = bool(k_j)
        # Pad low bits for fractional alignment
        if bias_low > 0:
            v = bb.add_concat([v, _literal(bb, bias_low, 0)])
            obw += bias_low
        # Sign/zero extend to max_out_bw
        if obw < max_out_bw:
            v = _extend(bb, v, obw, max_out_bw, is_signed)
        out_elems.append(v)

    ret_val = bb.add_array(out_elem_type, out_elems, name='model_out')

    _ = fb.build_with_return_value(ret_val)
    ir_str = pkg.to_string()
    pkg_opt = parse_ir_package(optimize_ir(ir_str, top=fn_name))
    fn_opt = pkg_opt.get_function(fn_name)
    return pkg_opt, fn_opt
