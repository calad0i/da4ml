import numpy as np
import pytest

from da4ml.trace.ops import relu

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
    'maximum': lambda x, w: np.maximum(x[..., None, :], w),
    'minimum': lambda x, w: np.minimum(x[..., None, :], w),
    'amax': lambda x, w: np.amax(x, axis=-1, keepdims=True),
    'amin': lambda x, w: np.amin(x, axis=-1, keepdims=True),
    'relu': lambda x, w: relu(x),
    'mux0': lambda x, w: np.where(x[..., None] > w, x[..., None], w),
}


class TestOperations(OperationTest):
    @pytest.fixture(params=list(functions.keys()))
    def op_func(self, request, w8x8: np.ndarray, test_data, inp):
        np.save('/tmp/wtf1.npy', w8x8)
        np.save('/tmp/wtf2.npy', test_data)
        np.save('/tmp/wtf3.npy', inp.kif)
        return lambda x: functions[request.param](x, w8x8)
