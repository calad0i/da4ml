from da4ml.cmvm.types import Op, Solution, _minimal_kif


def kif_to_vitis_type(k: bool | int, i: int, f: int):
    return f'ap_{"" if k else "u"}fixed<{k+i+f},{k+i}>'


def kif_to_hlslib_type(k: bool | int, i: int, f: int):
    return f'ac_fixed<{int(k)},{k+i+f},{k+i}>'


class CppCodeGen:
    def type(self, k: bool | int, i: int, f: int):
        return self._type_fn(k, i, f)

    def __init__(self, flavor: str):
        match flavor.lower():
            case 'vitis':
                self._type_fn = kif_to_vitis_type
            case 'hlslib':
                self._type_fn = kif_to_hlslib_type
            case _:
                raise ValueError(f'Unsupported flavor: {flavor}')
        self.flavor = flavor

    def ssa_gen(self, ops: list[Op], print_latency: bool = False):
        all_kifs = map(_minimal_kif, (op.qint for op in ops))
        all_types = list(map(lambda x: self.type(*x), all_kifs))

        lines = []

        for i, op in enumerate(ops):
            _type = all_types[i]

            ref0 = f'v{op.id0}'

            if op.id1 >= 0:
                # Common a+/-b<<shift op
                ref1 = f'bit_shift<{op.shift}>(v{op.id1})' if op.shift != 0 else f'v{op.id1}'
                val = f'{ref0} {"-" if op.sub else "+"} {ref1}'

            elif op.id1 == -1:
                # Input marker
                val = f'inp[{ops[op.id0].id0}]'

            elif op.id1 == -2:
                if not op.sub:  # relu(inp)
                    if ops[op.id0].qint.min < 0:
                        val = f'{ref0} > 0 ? {_type}({ref0}) : {_type}(0)'
                    else:
                        val = ref0
                else:  # relu(-inp)
                    if ops[op.id0].qint.max > 0:
                        val = f'{ref0} > 0 ? {_type}(0) : {_type}(-{ref0})'
                    else:
                        val = f'-{ref0}'

            elif op.id1 == -3:
                # Explicit quantization op, done implicitly via assignment
                val = ref0

            else:
                raise ValueError(f'Invalid id1: {op.id1}')

            line = f'{_type} v{i} = {val};'

            if print_latency:
                line += f' // {op.latency}'
            lines.append(line)
        return lines

    def output_gen(self, sol: Solution):
        lines = []
        for i, idx in enumerate(sol.out_idxs):
            if idx < 0:
                lines.append(f'out[{i}] = 0;')
                continue
            _type = self.type(*_minimal_kif(sol.out_qint[i]))
            shift = sol.out_shifts[i]
            neg_str = '-' if sol.out_neg[i] else ''
            if shift == 0:
                lines.append(f'out[{i}] = {_type}({neg_str}v{idx});')
            else:
                lines.append(f'out[{i}] = {_type}({neg_str}bit_shift<{shift}>(v{idx}));')
        return lines

    def __call__(self, sol: Solution, fn_name: str, n_indent: int = 4, n_base_indent: int = 0, print_latency: bool = False):
        in_kif = map(max, zip(*map(_minimal_kif, sol.inp_qint)))
        inp_type = self.type(*in_kif)
        out_kif = map(max, zip(*map(_minimal_kif, sol.out_qint)))
        out_type = self.type(*out_kif)

        n_in, n_out = sol.shape
        fn_signature = f'void {fn_name}({inp_type} inp[{n_in}], {out_type} out[{n_out}])'
        pragma = ''
        if self.flavor == 'vitis':
            pragma = '#pragma HLS INLINE'

        ssa_lines = self.ssa_gen(sol.ops, print_latency=print_latency)
        output_lines = self.output_gen(sol)

        indent = ' ' * n_indent
        base_indent = indent * n_base_indent
        body_indent = '\n' + base_indent + indent
        code = f"""{base_indent}{fn_signature} {{
{body_indent}{pragma}
{body_indent}{body_indent.join(ssa_lines)}
{body_indent}{body_indent.join(output_lines)}
{base_indent}}}
"""
        bridge = f"""#include "bridge.h"
#include "fn.h"

extern "C" {{
void bridge(double *inp, double *out, int size) {{
    vitis_bridge<{inp_type}, {out_type}, {n_in}, {n_out}>({fn_name}, inp, out, size);
}}
}}"""
        return code, bridge
