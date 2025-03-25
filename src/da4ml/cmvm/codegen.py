import types
from collections.abc import Sequence
from math import log2

from .fixed_variable import FixedVariable, Namer


class PyCodegenBackend:
    _comment = '#'

    def __init__(self, namer=Namer(), fn_name: str = 'placeholder', **kwargs):
        self._namer = namer
        self._attrs = {'fn_name': fn_name, **kwargs}

    def reference_code(self, v: FixedVariable):
        """How the variable should be referenced in the code:
        ```
        ... (+/- a * 2**shift) ...
        ```
        """
        if v.int_min == v.int_max:
            return f'{v.min}'

        neg = v._factor < 0
        shift = log2(abs(v._factor))
        assert shift % 1 == 0
        shift = int(shift)
        s_sign = '-' if neg else ''
        s_shift = f' * {2.**shift}' if shift != 0 else ''
        return f'{s_sign}{v.name}{s_shift}'

    def def_code(self, v: FixedVariable):
        """How the variable should be defined in the code:
        ```
        # a is not defined before
        a = b + c * 2**shift
        ```
        """
        if v.int_min == v.int_max:
            raise ValueError('Constant variable should not be defined')
        assert v._from is not None, 'Variable not derived from other variables cannot be defined in runtime'
        v1_str = self.reference_code(v._from[0])
        v2_str = self.reference_code(v._from[1])
        if v2_str[0] == '-':
            return f'{v.name} = {v1_str} - {v2_str[1:]}'
        return f'{v.name} = {v1_str} + {v2_str}'

    def _resolve_variable(self, v: FixedVariable, _recorded: dict[str, FixedVariable]):
        if v.name in _recorded:
            return

        if v.int_min == v.int_max:
            _recorded[v.name] = v
            return

        if v._from is None:
            raise ValueError('Variable not derived from other variables cannot be defined in runtime')

        self._resolve_variable(v._from[0], _recorded)
        self._resolve_variable(v._from[1], _recorded)
        _recorded[v.name] = v

    def resolve_all_variables(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        _recorded = {v.name: v for v in inputs}
        for v in outputs:
            self._resolve_variable(v, _recorded)
        return _recorded

    def gen_lines(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        variables = self.resolve_all_variables(inputs, outputs)
        keys = list(variables.keys())
        keys = sorted(keys, key=lambda x: variables[x]._depth)
        codes = list(self.gen_code_prefixs(inputs))
        cur_depth = -1
        s_inputs = set(inputs)
        for key in keys:
            v = variables[key]
            if v.int_min == v.int_max or v in s_inputs:
                continue
            if cur_depth != v._depth:
                cur_depth = v._depth
                codes.append(f'{self._comment} ========================== Latency: {cur_depth} ==========================')
            codes.append(self.def_code(v))
        codes.extend(self.gen_code_suffixs(outputs))
        return codes

    def gen_code_prefixs(self, inputs) -> Sequence[str]:
        return ()

    def gen_code_suffixs(self, outputs) -> Sequence[str]:
        return [f'out[{i}] = {self.reference_code(out)}' for i, out in enumerate(outputs)]

    def gen_fn(self, inputs: list[FixedVariable], outputs: list[FixedVariable], **kwargs):
        fn_name = kwargs.get('fn_name', self._attrs['fn_name'])
        code = self.gen_lines(inputs, outputs)
        code_str = '\n    '.join(code)
        fn_str = f"""def {fn_name}(inp: list[float]):
    out = [0.]*{len(outputs)}
    {code_str}
    return out
"""
        fn_obj = compile(fn_str, '<string>', 'exec')
        fn = types.FunctionType(fn_obj.co_consts[1], globals())
        return fn, fn_str

    def __call__(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        return self.gen_fn(inputs, outputs)


class VitisCodegenBackend(PyCodegenBackend):
    _comment = '//'

    def __init__(self, namer=Namer(), fn_name: str = 'placeholder', **kwargs):
        self._namer = namer
        self._attrs = {'fn_name': fn_name, **kwargs}

    def reference_code(self, v: FixedVariable):
        """How the variable should be referenced in the code"""
        if v.int_min == v.int_max:
            k, b, i = v.k, v.b, v.i
            u = '' if k else 'u'
            type_str = f'ap_{u}fixed<{max(b+k,1)}, {i+k}>'
            return f'{type_str}({v.min})'

        neg = v._factor < 0
        shift = log2(abs(v._factor))
        assert shift % 1 == 0
        shift = int(shift)
        s_sign = '-' if neg else ''
        if shift == 0:
            return f'{s_sign}{v.name}'
        return f'{s_sign}bit_shift<{shift}>({v.name})'

    def gen_code_suffixs(self, outputs):
        return [f'out[{i}] = {self.reference_code(out)};' for i, out in enumerate(outputs)]

    def def_code(self, v: FixedVariable):
        """How the variable should be defined in the code"""
        if v.int_min == v.int_max:
            raise ValueError('Constant variable should not be defined')
        assert v._from is not None, 'Variable not derived from other variables cannot be defined in runtime'
        v1_str = self.reference_code(v._from[0])
        v2_str = self.reference_code(v._from[1])
        vv = v * (1 / v._factor)
        k, b, i = vv.k, vv.b, vv.i
        b, i = b + k, i + k  # b and i did not include sign bit
        u = '' if k else 'u'
        type_str = f'ap_{u}fixed<{b}, {i}>'
        if v2_str[0] == '-':
            return f'{type_str} {v.name} = {v1_str} - {v2_str[1:]};'
        return f'{type_str} {v.name} = {v1_str} + {v2_str};'

    def gen_fn(self, inputs: list[FixedVariable], outputs: list[FixedVariable], **kwargs):
        attrs = {**self._attrs, **kwargs}
        fn_name = attrs['fn_name']
        code = self.gen_lines(inputs, outputs)
        code_str = '\n    '.join(code)

        fn_str = f"""template <typename inp_t, typename out_t>
void {fn_name}(inp_t inp[{len(inputs)}], out_t out[{len(outputs)}]) {{
    {code_str}
}}
"""
        self._comment = '#'
        fn, _ = PyCodegenBackend().gen_fn(inputs, outputs, fn_name=fn_name)
        self._comment = '//'
        return fn, fn_str

    def __call__(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        return self.gen_fn(inputs, outputs)


class VerilogCodegenBackend(PyCodegenBackend):
    _comment = '//'

    def __init__(self, namer=Namer(), module_name: str = 'placeholder', **kwargs):
        self._namer = namer
        self._attrs = {'fn_name': module_name, **kwargs}

    def def_code(self, v: FixedVariable):
        """How the variable should be defined in the code"""
        if v.int_min == v.int_max:
            raise ValueError('Constant variable should not be defined')
        assert v._from is not None, 'Variable not derived from other variables cannot be defined in runtime'

        vv = v * (1 / v._factor)
        v_in0, v_in1 = vv._from  # type: ignore

        f_in0, f_in1 = v_in0.shift, v_in1.shift
        shift = f_in0 - f_in1
        v_in0, v_in1 = v_in0 * (1 / v_in0._factor), v_in1 * (1 / v_in1._factor)
        B_out = vv.b + vv.k
        declare = f'wire [{B_out-1}:0] {v.name};'

        k_in0, k_in1 = v_in0.k, v_in1.k
        b_in0, b_in1 = v_in0.b, v_in1.b
        B_in0, B_in1 = b_in0 + k_in0, b_in1 + k_in1

        ref_in0 = f'{v._from[0].name.replace("[","").replace("]","")}'
        ref_in1 = f'{v._from[1].name.replace("[","").replace("]","")}'
        signed0, signed1 = v._from[0].k, v._from[1].k
        assert not v._from[0]._factor < 0, 'operand 0 must have a positive factor'
        is_sub = int(v._from[1]._factor < 0)

        cmd = f'shift_adder #({B_in0}, {B_in1}, {signed0}, {signed1}, {B_out}, {shift}, {is_sub}) op_{v.name} ({ref_in0}[{B_in0-1}:0], {ref_in1}[{B_in1-1}:0], {v.name}[{B_out-1}:0]);'
        return f'{declare} {cmd}'

    def gen_code_prefixs(self, inputs: list[FixedVariable]) -> Sequence[str]:
        code = []
        max_f = max([v.shift for v in inputs])
        max_I = max([v.i for v in inputs]) + max([v.k for v in inputs])
        max_bw = max_f + max_I
        for i, inp in enumerate(inputs):
            name = inp.name.replace('[', '').replace(']', '')
            bw = inp.b + inp.k
            bias = i * max_bw + max_f - inp.shift
            code.append(f'wire [{bw-1}:0] {name};')
            code.append(f'assign {name}[{bw-1}:0] = in[{bias + bw - 1}:{bias}];')
        return code

    def gen_code_suffixs(self, outputs: list[FixedVariable]) -> Sequence[str]:
        code = []
        max_f = max([v.shift for v in outputs])
        max_bw = max([v.i for v in outputs]) + max([v.k for v in outputs]) + max_f
        for i, out in enumerate(outputs):
            name = out.name.replace('[', '').replace(']', '')
            bw = out.b + out.k
            if out._factor < 0:
                code.append(f'wire [{bw-1}:0] {name}_neg;')
                code.append(f'assign {name}_neg = -{name};')
                name = f'{name}_neg'
            n_pad = max_bw - bw
            n_pad_right = max_f - out.shift
            n_pad_left = n_pad - n_pad_right
            bias = i * max_bw + n_pad_right
            if n_pad_right > 0:
                code.append(f"assign out[{bias - 1}:{bias - n_pad_right}] = {{{n_pad_right}{{1'b0}}}};")
            code.append(f'assign out[{bias + bw - 1}:{bias}] = {name};')
            if n_pad_left > 0:
                v_pad_left = f'{name}[{bw-1}]' if out.k else "1'b0"
                code.append(f'assign out[{bias + bw - 1 + n_pad_left}:{bias + bw}] = {{{n_pad_left}{{{v_pad_left}}}}};')
        return code

    def gen_fn(self, inputs: list[FixedVariable], outputs: list[FixedVariable], **kwargs):
        attrs = {**self._attrs, **kwargs}
        fn_name = attrs['fn_name']
        code = self.gen_lines(inputs, outputs)
        code_str = '\n    '.join(code)

        max_inp_bw = max([v.i for v in inputs]) + max([v.k for v in inputs]) + max([v.shift for v in inputs])
        max_out_bw = max([v.i for v in outputs]) + max([v.k for v in outputs]) + max([v.shift for v in outputs])
        padded_inp_bits = max_inp_bw * len(inputs)
        padded_out_bits = max_out_bw * len(outputs)
        fn_str = f"""`timescale 1ns / 1ps

module {fn_name} (
    // verilator lint_off UNUSEDSIGNAL
    input  [{padded_inp_bits-1}:0] in,
    input  clk,
    output [{padded_out_bits-1}:0] out
    // verilator lint_on UNUSEDSIGNAL
);

    {code_str}
endmodule

"""
        self._comment = '#'
        fn, _ = PyCodegenBackend().gen_fn(inputs, outputs, fn_name=fn_name)
        self._comment = '//'
        return fn, fn_str

    def __call__(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        return self.gen_fn(inputs, outputs)
