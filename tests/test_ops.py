import os

import numpy as np
import pytest

from da4ml.cmvm.types import CombLogic
from da4ml.codegen import HLSModel, RTLModel
from da4ml.trace import FixedVariableArray, FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import quantize, relu


class OperationTest:
    def test_op(self, op_func, test_data: np.ndarray, comb: CombLogic, inp: FixedVariableArray, n_samples: int):
        traced_out = comb.predict(test_data, n_threads=1)
        expected_out = quantize(op_func(quantize(test_data, *inp.kif)).reshape(n_samples, -1), 1, 12, 12)
        np.testing.assert_equal(traced_out, expected_out)

        symbolic_out = []
        for x in test_data[:100]:
            x = list(map(float, x))
            r = comb(x, quantize=True)
            symbolic_out.append(r)
        symbolic_out = np.array(symbolic_out, dtype=np.float64)
        np.testing.assert_equal(symbolic_out, traced_out[:100])

    @pytest.fixture()
    def comb(self, op_func, inp: FixedVariableArray):
        out = quantize(op_func(inp), 1, 12, 12)
        comb = comb_trace(inp, out)
        return comb

    @pytest.fixture()
    def n_samples(self) -> int:
        return 10000

    @pytest.fixture()
    def inp(self) -> FixedVariableArray:
        b = np.random.randint(0, 9, size=8)
        i = np.random.randint(-8, 8, size=8)
        k = np.random.randint(0, 2, size=8)
        inp = FixedVariableArray.from_kif(k, i, b - i)
        return inp

    @pytest.fixture(autouse=True)
    def test_data(self, inp: FixedVariableArray, n_samples: int):
        shape = inp.shape
        data = np.random.randn(n_samples, *shape) * 32
        return data

    def test_retrace(self, comb: CombLogic, inp: FixedVariableArray, test_data: np.ndarray):
        inp2 = FixedVariableArrayInput(inp.shape).quantize(*inp.kif).as_new()
        out2 = comb(inp2, debug=True, quantize=True)  # type: ignore
        comb2 = comb_trace(inp2, out2)
        r1, r2 = comb.predict(test_data), comb2.predict(test_data)
        np.testing.assert_equal(r1, r2)

    def test_serialization(self, comb: CombLogic, temp_directory: str):
        comb.save(f'{temp_directory}/comb.json')
        comb2 = CombLogic.load(f'{temp_directory}/comb.json')
        assert comb == comb2


class OperationTestSynth(OperationTest):
    @pytest.mark.parametrize('flavor', ('verilog', 'vhdl'))
    @pytest.mark.parametrize('latency_cutoff', (-1, 0.5, 1))
    def test_rtl_gen(self, comb: CombLogic, flavor: str, latency_cutoff, temp_directory: str, test_data: np.ndarray):
        rtl_model = RTLModel(comb, 'test', temp_directory, flavor=flavor, latency_cutoff=latency_cutoff)
        before = rtl_model.__repr__()
        if flavor == 'verilog' and os.system('verilator --version') != 0:
            rtl_model.write()
            os.system(f'rm -rf {temp_directory}')
            pytest.skip('verilator not found')
        if flavor == 'vhdl' and os.system('ghdl --version') != 0:
            rtl_model.write()
            os.system(f'rm -rf {temp_directory}')
            pytest.skip('ghdl not found')
        rtl_model.compile(nproc=1)
        after = rtl_model.__repr__()
        assert before != after

        rtl_pred = rtl_model.predict(test_data, n_threads=1)
        comb_pred = comb.predict(test_data, n_threads=1)
        np.testing.assert_equal(rtl_pred, comb_pred)
        os.system(f'rm -rf {temp_directory}')

    @pytest.mark.parametrize('flavor', ('vitis',))
    def test_hls_gen(self, comb: CombLogic, flavor: str, temp_directory: str, test_data: np.ndarray):
        hls_model = HLSModel(comb, 'test', temp_directory, flavor=flavor)
        # if flavor != 'vitis':
        #     hls_model.write()
        #     os.system(f'rm -rf {temp_directory}')
        #     pytest.skip('hlslib and oneapi functional simulation not implemented yet')

        before = hls_model.__repr__()
        hls_model.compile()
        after = hls_model.__repr__()
        assert before != after

        hls_pred = hls_model.predict(test_data, n_threads=1)
        comb_pred = comb.predict(test_data, n_threads=1)
        np.testing.assert_equal(hls_pred, comb_pred)
        os.system(f'rm -rf {temp_directory}')


class TestQuantize(OperationTestSynth):
    @pytest.fixture()
    def op_func(self, overflow_mode: str, round_mode: str):
        return lambda x: quantize(x, 1, 3, 3, overflow_mode, round_mode)

    @pytest.fixture(params=['WRAP', 'SAT', 'SAT_SYM'])
    def overflow_mode(self, request) -> str:
        return request.param

    @pytest.fixture(params=['TRN', 'RND'])
    def round_mode(self, request) -> str:
        return request.param


class TestShiftAdd(OperationTestSynth):
    @pytest.fixture()
    def op_func(self, s: tuple[float, float]):
        return lambda x: x[..., :4] * s[0] + x[..., 4:] * s[1]

    @pytest.fixture(params=[(0.5, 0.5), (1.0, -2.0), (-3.5, 0.125), (-2.0, -2.0)])
    def s(self, request) -> tuple[float, float]:
        return request.param


class TestLookup(OperationTestSynth):
    @pytest.fixture()
    def op_func(self, fn):
        return lambda x: quantize(fn(x), 1, 3, 3, 'SAT', 'RND_CONV')

    @pytest.fixture(params=['sin', 'tanh', 'sin-and-tanh'])
    def fn(self, request):
        if request.param == 'sin':
            return np.sin
        elif request.param == 'tanh':
            return np.tanh
        elif request.param == 'sin-and-tanh':
            return lambda x: np.tanh(np.sin(x))
        else:
            raise ValueError()


class TestReLU(OperationTestSynth):
    @pytest.fixture()
    def op_func(self):
        return lambda x: relu(x * 2 * (np.arange(8) % 2) - 1 + np.arange(-8, 8, 2))


class TestBranching(OperationTestSynth):
    @pytest.fixture(params=['abs', 'max', 'min', 'mux', 'cmp', 'mux2'])
    def op_func(self, request):
        if request.param == 'abs':
            return np.abs
        if request.param == 'max':
            return lambda x: np.max(x, axis=-1)
        if request.param == 'min':
            return lambda x: np.min(x, axis=-1)
        elif request.param == 'mux':
            return lambda x: np.where(x[..., :1] < x[..., 1:], x[..., :7], x[..., 1:])
        elif request.param == 'cmp':
            return lambda x: x[..., :4] >= x[..., 4:]
        elif request.param == 'mux2':
            return lambda x: np.where(x[..., :4] <= x[..., 4:], x[..., 4:] * -2, x[..., :4] * 7)
        else:
            raise ValueError()


class TestMul(OperationTestSynth):
    @pytest.fixture()
    def op_func(self):
        return lambda x: x[..., 0:4] * x[..., 4:8]


class TestBinaryBitOps(OperationTestSynth):
    @pytest.fixture(params=['and', 'or', 'xor'])
    def op_func(self, request):
        w0 = np.arange(8) - 4
        w1 = ((np.arange(8) % 2) * 2 - 1) * np.arange(1, 9)
        sf = 2**16

        def func(x):
            x0, x1 = x * w0, x[..., ::-1] * w1
            if isinstance(x, np.ndarray):
                x0, x1 = (x0 * sf).astype(np.int64), (x1 * sf).astype(np.int64)
            if request.param == 'and':
                x = x0 & x1
            elif request.param == 'or':
                x = x0 | x1
            elif request.param == 'xor':
                x = x0 ^ x1
            else:
                raise ValueError()

            if isinstance(x, np.ndarray):
                x = x / sf

            return x + 3.75

        return func


class TestBitReduction(OperationTestSynth):
    @pytest.fixture(params=[0, 1])
    def signed(self, request) -> bool:
        return bool(request.param)

    @pytest.fixture()
    def inp(self, signed) -> FixedVariableArray:
        k = np.ones(8, dtype=np.int64) * signed
        i = np.full(8, 4, dtype=np.int64)
        f = np.zeros(8, dtype=np.int64)
        inp = FixedVariableArray.from_kif(k, i, f)
        return inp

    @pytest.fixture(params=['all', 'any'])
    def op_func(self, request, signed):
        def func(x):
            if request.param == 'any':
                return x != 0
            else:
                if isinstance(x, np.ndarray):
                    return x == -1 if signed else x == 15
                else:
                    return x.to_bool('all')

        return func


class TestBitNot(OperationTestSynth):
    @pytest.fixture(params=[0, 1])
    def signed(self, request) -> bool:
        return bool(request.param)

    @pytest.fixture()
    def inp(self, signed) -> FixedVariableArray:
        k = np.ones(8, dtype=np.int64) * signed
        i = np.full(8, 8 - signed, dtype=np.int64)
        f = np.zeros(8, dtype=np.int64)
        inp = FixedVariableArray.from_kif(k, i, f)
        return inp

    @pytest.fixture(params=['not'])
    def op_func(self, request, signed):
        def func(x):
            if request.param == 'not':
                if isinstance(x, np.ndarray):
                    x = x.astype(np.int8) if signed else x.astype(np.uint8)
                x = ~x
            else:
                raise ValueError(f'Unknown unary bit op {request.param}')

            return x + 3.75

        return func
