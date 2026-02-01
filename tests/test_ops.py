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
        expected_out = op_func(quantize(test_data, *inp.kif)).reshape(n_samples, -1)
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
        out = op_func(inp)
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
        inp2 = FixedVariableArrayInput(inp.shape).quantize(*inp.kif)
        out2 = comb(inp2)  # type: ignore
        comb2 = comb_trace(inp2, out2)
        r1, r2 = comb.predict(test_data), comb2.predict(test_data)
        np.testing.assert_equal(r1, r2)

    def test_serialization(self, comb: CombLogic, temp_directory: str):
        comb.save(f'{temp_directory}/comb.json')
        comb2 = CombLogic.load(f'{temp_directory}/comb.json')
        assert comb == comb2


class OperationTestSynth(OperationTest):
    @pytest.mark.parametrize('flavor', ('verilog', 'vhdl'))
    @pytest.mark.parametrize('latency_cutoff', (-1, 1))
    def test_rtl_gen(self, comb: CombLogic, flavor: str, latency_cutoff, temp_directory: str, test_data: np.ndarray):
        rtl_model = RTLModel(comb, 'test', temp_directory, flavor=flavor, latency_cutoff=latency_cutoff)
        rtl_model.compile(nproc=1)

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

        hls_model.compile()

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
        return lambda x: relu(x)


class TestBranching(OperationTestSynth):
    @pytest.fixture(params=['abs', 'max', 'min', 'mux'])
    def op_func(self, request):
        if request.param == 'abs':
            return np.abs
        if request.param == 'max':
            return lambda x: np.max(x, axis=-1)
        if request.param == 'min':
            return lambda x: np.min(x, axis=-1)
        elif request.param == 'mux':
            return lambda x: np.where(x[..., :1] < x[..., 1:], x[..., :7], x[..., 1:])
        else:
            raise ValueError()


class TestMul(OperationTestSynth):
    @pytest.fixture()
    def op_func(self):
        return lambda x: x[..., 0:4] * x[..., 4:8]
