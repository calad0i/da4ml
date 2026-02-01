import numpy as np
import pytest

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
    'relu-transpose': lambda x, w: relu(x),
    'mux0': lambda x, w: np.where(x[..., None] > w, x[..., None], w),
    'lut': lambda x, w: quantize(np.cos(np.sin(x)), 1, 2, 3)
    if isinstance(x, np.ndarray)
    else quantize(x.apply(np.sin).apply(np.cos), 1, 2, 3),
    'prod': lambda x, w: np.prod(x[..., :3], axis=-1, keepdims=True),
    'mean': lambda x, w: np.mean(x, axis=-1, keepdims=True),
    'sum': lambda x, w: np.sum(x, axis=-1, keepdims=True),
    'clip0': lambda x, w: np.clip(x, -1.0, 2.0),
    'clip1': lambda x, w: np.clip(x[..., :4], x[..., 4:8], 1.5),
    'dot0': lambda x, w: np.dot(x, w),
    'dot1': lambda x, w: np.dot(np.mean(x, axis=-1, keepdims=True), np.array(1.25)),
}


class TestOperations(OperationTest):
    @pytest.fixture(params=list(functions.keys()))
    def op_func(self, request, w8x8: np.ndarray, test_data, inp):
        np.save('/tmp/wtf1.npy', w8x8)
        np.save('/tmp/wtf2.npy', test_data)
        np.save('/tmp/wtf3.npy', inp.kif)
        return lambda x: functions[request.param](x, w8x8)
