from collections.abc import Callable

from ...cmvm.types import Op, QInterval, Solution, _minimal_kif
from ...trace.fixed_variable import _const_f


def kif_to_vitis_type(k: bool | int = 1, i: int = 0, f: int = 0):
    if k == i == f == 0:
        f = 1
    return f'ap_{"" if k else "u"}fixed<{k+i+f},{k+i}>'


def kif_to_hlslib_type(k: bool | int = 1, i: int = 0, f: int = 0):
    if k == i == f == 0:
        f = 1
    return f'ac_fixed<{int(k)},{k+i+f},{k+i}>'


def get_typestr_fn(flavor: str):
    match flavor.lower():
        case 'vitis':
            typestr_fn = kif_to_vitis_type
        case 'hlslib':
            typestr_fn = kif_to_hlslib_type
        case _:
            raise ValueError(f'Unsupported flavor: {flavor}')
    return typestr_fn


def ssa_gen(ops: list[Op], print_latency: bool, typestr_fn: Callable[[bool | int, int, int], str]):
    all_kifs = map(_minimal_kif, (op.qint for op in ops))
    all_types = list(map(lambda x: typestr_fn(*x), all_kifs))

    lines = []

    for i, op in enumerate(ops):
        _type = all_types[i]

        ref0 = f'v{op.id0}'

        match op.opcode:
            case -1:
                # Input marker
                val = f'inp[{ops[op.id0].id0}]'

            case 0 | 1:
                # Common a+/-b<<shift op
                ref1 = f'bit_shift<{op.data}>(v{op.id1})' if op.data != 0 else f'v{op.id1}'
                val = f'{ref0} {"-" if op.opcode == 1 else "+"} {ref1}'

            case 2 | -2:
                if op.opcode == 2:  # relu(inp)
                    if ops[op.id0].qint.min < 0:
                        val = f'{ref0} > 0 ? {_type}({ref0}) : {_type}(0)'
                    else:
                        val = ref0
                else:  # relu(-inp)
                    if ops[op.id0].qint.max > 0:
                        val = f'{ref0} > 0 ? {_type}(0) : {_type}(-{ref0})'
                    else:
                        val = f'-{ref0}'

            case 3 | -3:
                # Explicit quantization op, done implicitly via assignment
                val = ref0 if op.opcode == 3 else f'-{ref0}'

            case 4:
                # Constant addition
                _number = op.data * op.qint.step
                sign, mag = ('-' if _number < 0 else '+'), abs(_number)
                f = _const_f(mag)
                const_type_str = typestr_fn(*_minimal_kif(QInterval(mag, mag, 2.0**-f)))
                val = f'{ref0} {sign} {const_type_str}({mag})'

            case 5:
                _number = op.data * op.qint.step
                val = f'{_number}'

            case _:
                raise ValueError(f'Unsupported opcode: {op.opcode}')

        line = f'{_type} v{i} = {val};'

        if print_latency:
            line += f' // {op.latency}'
        lines.append(line)
    return lines


def output_gen(sol: Solution, typestr_fn: Callable[[bool | int, int, int], str]):
    lines = []
    for i, idx in enumerate(sol.out_idxs):
        if idx < 0:
            lines.append(f'out[{i}] = 0;')
            continue
        _type = typestr_fn(*_minimal_kif(sol.out_qint[i]))
        shift = sol.out_shifts[i]
        neg_str = '-' if sol.out_negs[i] else ''
        if shift == 0:
            lines.append(f'out[{i}] = {_type}({neg_str}v{idx});')
        else:
            lines.append(f'out[{i}] = {_type}({neg_str}bit_shift<{shift}>(v{idx}));')
    return lines


def cpp_logic_and_bridge_gen(
    sol: Solution,
    fn_name: str,
    flavor: str,
    pragmas: list[str] | None = None,
    n_indent: int = 4,
    n_base_indent: int = 0,
    print_latency: bool = False,
):
    typestr_fn = get_typestr_fn(flavor)
    in_kif = map(max, zip(*map(_minimal_kif, sol.inp_qint)))
    inp_type = typestr_fn(*in_kif)
    out_kif = map(max, zip(*map(_minimal_kif, sol.out_qint)))
    out_type = typestr_fn(*out_kif)

    n_in, n_out = sol.shape
    template_def = 'template <typename inp_t, typename out_t>'
    fn_signature = f'void {fn_name}(inp_t inp[{n_in}], out_t out[{n_out}])'
    pragmas = pragmas or []

    ssa_lines = ssa_gen(sol.ops, print_latency=print_latency, typestr_fn=typestr_fn)
    output_lines = output_gen(sol, typestr_fn=typestr_fn)

    indent = ' ' * n_indent
    base_indent = indent * n_base_indent
    body_indent = '\n' + base_indent + indent
    code = f"""{base_indent}{template_def}
{base_indent}{fn_signature} {{ // {inp_type} -> {out_type}
{body_indent}{body_indent.join(pragmas)}
{body_indent}{body_indent.join(ssa_lines)}
{body_indent}{body_indent.join(output_lines)}
{base_indent}}}
"""
    bridge = f"""#include "bridge.h"
#include "fn.h"

extern "C" {{
void bridge(double *inp, double *out, int size) {{
    auto fn = {fn_name}<{inp_type}, {out_type}>;
    vitis_bridge<{inp_type}, {out_type}, {n_in}, {n_out}>(fn, inp, out, size);
}}
}}"""
    return code, bridge
