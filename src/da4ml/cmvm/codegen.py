import types
from math import log2

from .fixed_variable import FixedVariable, Namer


class PyCodegenBackend:
    _comment = '#'

    def __init__(self, namer=Namer(), fn_name: str = 'placeholder', **kwargs):
        self._namer = namer
        self._attrs = {'fn_name': fn_name, **kwargs}

    def reference_code(self, v: FixedVariable):
        """How the variable should be referenced in the code"""
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
        """How the variable should be defined in the code"""
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
        codes = []
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
        for i, out in enumerate(outputs):
            codes.append(f'out[{i}] = {self.reference_code(out)}')
        return codes

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

    def gen_lines(self, inputs: list[FixedVariable], outputs: list[FixedVariable]):
        codes = super().gen_lines(inputs, outputs)
        n = len(outputs)
        for i, out in enumerate(outputs):
            codes[-n + i] = f'out[{i}] = {self.reference_code(out)};'
        return codes

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
