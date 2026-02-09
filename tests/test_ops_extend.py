import numpy as np
import pytest

from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import quantize, relu

from .test_ops import OperationTest


@pytest.fixture(autouse=True)
def w8x8():
    return (np.random.randn(8, 8).astype(np.float32) * 32).round() / 32


functions = {
    'einsum0': lambda x, w: np.einsum('...i,...i->...i', x[..., :4], x[..., 4:]),
    'einsum1': lambda x, w: np.einsum('...ij,...jk->...ik', x.reshape(-1, 4, 2), x.reshape(-1, 2, 4)),
    'power': lambda x, w: x**2,
    'cmvm0': lambda x, w: np.einsum('...i,ij->...j', x, w),
    'cmvm1': lambda x, w: np.einsum('...i,ij->...', x, w),
    'cmvm2': lambda x, w: x @ w,
    'cmvm3': lambda x, w: np.einsum('ij,...j->...i', w, x),
    'cmvm_collapsed_left': lambda x, w: np.einsum('ij,...j->...i', w, x * 0 + 1),
    'cmvm_collapsed_right': lambda x, w: (x * 0 + 2) @ w,
    'mvm_collapsed_left': lambda x, w: np.einsum('...i,...i->...i', x * 0 + 3, x),
    'mvm_collapsed_right': lambda x, w: np.einsum('...i,...i->...i', x, x * 0 + 4),
    'mvm_collapsed_all': lambda x, w: np.einsum('...i,...i->...i', x * 0 + 5, x * 0 + 6),
    'maximum': lambda x, w: np.maximum(x[..., None, :], w),
    'minimum': lambda x, w: np.minimum(x[..., None, :], w),
    'amax': lambda x, w: np.amax(x, axis=-1, keepdims=True),
    'amin': lambda x, w: np.amin(x, axis=-1, keepdims=True),
    'relu0': lambda x, w: relu(x),
    'relu1': lambda x, w: relu(x, i=np.array(1)),
    'relu2': lambda x, w: relu(x, f=np.array(1), round_mode='RND'),
    'multi_cadd': lambda x, w: x + 2 + 3.75,
    'mux0': lambda x, w: np.where(x[..., None] > w, x[..., None], w),
    'lut': lambda x, w: (
        quantize(np.cos(np.sin(x)), 1, 2, 3) if isinstance(x, np.ndarray) else quantize(x.apply(np.sin).apply(np.cos), 1, 2, 3)
    ),
    'prod': lambda x, w: np.prod(x[..., :3], axis=-1, keepdims=True),
    'mean': lambda x, w: np.mean(x, axis=-1, keepdims=True),
    'sum': lambda x, w: np.sum(x, axis=-1, keepdims=True),
    'clip0': lambda x, w: np.clip(x, -1.0, 2.0),
    'clip1': lambda x, w: np.clip(x[..., :4], x[..., 4:8], 1.5),
    'dot0': lambda x, w: np.dot(x, w),
    'dot1': lambda x, w: np.dot(np.mean(x, axis=-1, keepdims=True), np.array(1.25)),
    'where1': lambda x, w: np.where(x - 3 == 0, x * 2, x / 2),
    'where2': lambda x, w: np.where(x != 0, x, -1),
    'where3': lambda x, w: np.where(x >= 1.375, -1, x),
    'where4': lambda x, w: np.where(x[..., :4] <= x[..., 4:], x[..., 4:] + 1, x[..., 4:] - 1),
}


class TestOperations(OperationTest):
    @pytest.fixture(params=list(functions.keys()))
    def op_func(self, request, w8x8: np.ndarray, test_data, inp):
        np.save('/tmp/wtf1.npy', w8x8)
        np.save('/tmp/wtf2.npy', test_data)
        np.save('/tmp/wtf3.npy', inp.kif)
        return lambda x: functions[request.param](x, w8x8)


@pytest.mark.parametrize('thres', [0.0, 0.5, 1.0])
def test_offload(thres):
    w = (np.random.randn(8, 8).astype(np.float32) * 10).round() / 10

    def offload_fn(weights, vector):
        return np.random.rand(*np.shape(weights)) > thres

    inp = FixedVariableArrayInput((2, 8), solver_options={'offload_fn': offload_fn}).quantize(1, 4, 3)
    out = inp @ w
    comb = comb_trace(inp, out)

    data_in = np.random.rand(10000, 2, 8).astype(np.float32) * 64 - 32
    traced_out = comb.predict(data_in, n_threads=1)
    expected_out = (quantize(data_in, *inp.kif) @ w).reshape(10000, -1)
    np.testing.assert_equal(traced_out, expected_out)
