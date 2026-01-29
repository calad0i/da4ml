from collections.abc import Callable

from ...cmvm.types import CombLogic, Op, QInterval, _minimal_kif
from ...trace.fixed_variable import _const_f, interpret_as
from ..rtl.verilog.comb import get_table_name as get_table_name_rtl


def get_table_name(sol: CombLogic, op: Op) -> str:
    return get_table_name_rtl(sol, op)[:-4].replace('-', '_')


def gen_mem_line(sol: CombLogic, op: Op, typestr_fn: Callable[[bool | int, int, int], str]) -> tuple[str, str]:
    assert op.opcode == 8
    assert sol.lookup_tables is not None
    table = sol.lookup_tables[op.data]
    data = table.padded_table(sol.ops[op.id0].qint)
    data = interpret_as(data, *table.spec.out_kif)
    values = ','.join(map(str, data))
    type_str = typestr_fn(*table.spec.out_kif)
    table_name = get_table_name(sol, op)
    line = f'static const {type_str} {table_name}[] = {{{values}}};'
    return table_name, line


def gen_mem_def(sol: CombLogic, typestr_fn: Callable[[bool | int, int, int], str]):
    lines: dict[str, str] = {}
    for op in sol.ops:
        if op.opcode == 8:
            table_name, line = gen_mem_line(sol, op, typestr_fn)
            lines[table_name] = line
    return list(lines.values())


def kif_to_vitis_type(k: bool | int = 1, i: int = 0, f: int = 0):
    if k == i == f == 0:
        f = 1
    return f'ap_{"" if k else "u"}fixed<{k + i + f},{k + i}>'


def kif_to_hlslib_type(k: bool | int = 1, i: int = 0, f: int = 0):
    if k == i == f == 0:
        f = 1
    return f'ac_fixed<{int(k)},{k + i + f},{k + i}>'


def kif_to_oneapi_type(k: bool | int = 1, i: int = 0, f: int = 0):
    # OneAPI requires at least 2 bits for all ac_fixed as of 2025.1
    return f'ac_fixed<{int(k)},{max(k + i + f, 2)},{k + i}>'


def get_typestr_fn(flavor: str):
    match flavor.lower():
        case 'vitis':
            typestr_fn = kif_to_vitis_type
        case 'hlslib':
            typestr_fn = kif_to_hlslib_type
        case 'oneapi':
            typestr_fn = kif_to_oneapi_type
        case _:
            raise ValueError(f'Unsupported flavor: {flavor}')
    return typestr_fn


def ssa_gen(comb: CombLogic, print_latency: bool, typestr_fn: Callable[[bool | int, int, int], str]):
    ops = comb.ops
    all_kifs = list(map(_minimal_kif, (op.qint for op in ops)))
    all_types = list(map(lambda x: typestr_fn(*x), all_kifs))

    lines = []
    ref_count = comb.ref_count
    for i, op in enumerate(ops):
        if ref_count[i] == 0:
            # Skip unused ops
            continue

        _type = all_types[i]

        ref0 = f'v{op.id0}'

        match op.opcode:
            case -1:
                # Input marker
                val = f'model_inp[{op.id0}]'
            case 0 | 1:
                # Common a+/-b<<shift op
                ref1 = f'bit_shift<{op.data}>(v{op.id1})' if op.data != 0 else f'v{op.id1}'
                val = f'{ref0} {"-" if op.opcode == 1 else "+"} {ref1}'
            case 2 | -2:
                if op.opcode == 2:  # relu(model_inp)
                    if ops[op.id0].qint.min < 0:
                        val = f'{ref0} > 0 ? {_type}({ref0}) : {_type}(0)'
                    else:
                        val = ref0
                else:  # relu(-model_inp)
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
                # Define constant
                _number = op.data * op.qint.step
                val = f'{_number}'
            case 6 | -6:
                # MSB Mux
                id_c = op.data & 0xFFFFFFFF
                bw_k = sum(all_kifs[id_c])
                shift = (op.data >> 32) & 0xFFFFFFFF
                shift = shift if shift < 0x80000000 else shift - 0x100000000
                ref_k = f'v{id_c}[{bw_k - 1}]'
                sign = '-' if op.opcode == -6 else ''
                ref1 = f'v{op.id1}' if shift == 0 else f'bit_shift<{shift}>(v{op.id1})'
                bw0, bw1 = sum(all_kifs[op.id0]), sum(all_kifs[op.id1])
                if bw0 == 0:
                    ref0 = '0'
                if bw1 == 0:
                    ref1 = '0'
                val = f'{ref_k} ? {_type}({ref0}) : {_type}({sign}{ref1})'
            case 7:
                # Multiplication
                ref1 = f'v{op.id1}'
                val = f'{ref0} * {ref1}'
            case 8:
                # Look-up
                table_name = get_table_name(comb, op)
                ref0 = f'v{op.id0}'
                val = f'{table_name}[{ref0}.range()]'
            case _:
                raise ValueError(f'Unsupported opcode: {op.opcode}')

        line = f'{_type} v{i} = {val};'

        if print_latency:
            line += f' // {op.latency}'
        lines.append(line)

    mem_def_lines = gen_mem_def(comb, typestr_fn)
    if mem_def_lines:
        mem_def_lines.extend(['', ''])

    return mem_def_lines + lines


def output_gen(sol: CombLogic, typestr_fn: Callable[[bool | int, int, int], str]):
    lines = []
    for i, idx in enumerate(sol.out_idxs):
        if idx < 0:
            lines.append(f'model_out[{i}] = 0;')
            continue
        _type = typestr_fn(*_minimal_kif(sol.out_qint[i]))
        shift = sol.out_shifts[i]
        neg_str = '-' if sol.out_negs[i] else ''
        if shift == 0:
            lines.append(f'model_out[{i}] = {_type}({neg_str}v{idx});')
        else:
            lines.append(f'model_out[{i}] = {_type}({neg_str}bit_shift<{shift}>(v{idx}));')
    return lines


def get_io_types(sol: CombLogic, flavor: str):
    typestr_fn = get_typestr_fn(flavor)
    in_kif = map(max, zip(*map(_minimal_kif, sol.inp_qint)))
    inp_type = typestr_fn(*in_kif)
    out_kif = map(max, zip(*map(_minimal_kif, sol.out_qint)))
    out_type = typestr_fn(*out_kif)
    return inp_type, out_type


def hls_logic_and_bridge_gen(
    sol: CombLogic,
    fn_name: str,
    flavor: str,
    pragmas: list[str] | None = None,
    n_indent: int = 4,
    n_base_indent: int = 0,
    print_latency: bool = False,
):
    typestr_fn = get_typestr_fn(flavor)
    inp_t, out_t = get_io_types(sol, flavor)

    n_in, n_out = sol.shape
    template_def = 'template <typename inp_t, typename out_t>'
    fn_signature = f'void {fn_name}(inp_t model_inp[{n_in}], out_t model_out[{n_out}])'
    pragmas = pragmas or []

    ssa_lines = ssa_gen(sol, print_latency=print_latency, typestr_fn=typestr_fn)
    output_lines = output_gen(sol, typestr_fn=typestr_fn)

    indent = ' ' * n_indent
    base_indent = indent * n_base_indent
    body_indent = '\n' + base_indent + indent
    code = f"""{base_indent}{template_def}
{base_indent}{fn_signature} {{ // {inp_t} -> {out_t}
{base_indent + indent}{body_indent.join(pragmas)}
{body_indent}{body_indent.join(ssa_lines)}
{body_indent}{body_indent.join(output_lines)}
{base_indent}}}
"""
    bridge = f"""#include "binder_util.hh"
#include "{fn_name}.hh"

struct {fn_name}_config {{
    static const size_t N_inp = {n_in};
    static const size_t N_out = {n_out};
    typedef {inp_t} inp_t;
    typedef {out_t} out_t;
    constexpr static auto f = {fn_name}<inp_t, out_t>;
}};

extern "C" {{

bool openmp_enabled() {{
    return _openmp;
}}

void inference_f64(double *model_inp, double *model_out, size_t size, size_t n_threads) {{
    batch_inference<{fn_name}_config, double>(model_inp, model_out, size, n_threads);
}}

void inference_f32(float *model_inp, float *model_out, size_t size, size_t n_threads) {{
    batch_inference<{fn_name}_config, float>(model_inp, model_out, size, n_threads);
}}
}}"""
    return code, bridge
