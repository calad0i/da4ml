from collections.abc import Callable
from math import prod

import keras
import numpy as np
from hgq.layers import QDenseT
from hgq.quantizer.internal import FixedPointQuantizerBase
from keras import ops

from ....converter.hgq2.layers._base import mirror_quantizer
from ....trace import FixedVariableArray
from ....trace.fixed_variable import FixedVariable
from ....trace.ops import _quantize
from ._base import ReplayOperationBase, to_np_arr


def keras_act_to_numpy(act: Callable) -> Callable:
    match act:
        case keras.activations.relu:
            return lambda x: np.maximum(0, x)
        case keras.activations.tanh:
            return np.tanh
        case keras.activations.softmax:
            raise ValueError('Non-local activation must not be used')
        case keras.activations.linear:
            return lambda x: x
        case keras.activations.sigmoid:
            return lambda x: 1 / (1 + np.exp(-x))
        case keras.activations.swish:
            return lambda x: x / (1 + np.exp(-x))
        case keras.activations.gelu:
            return lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        case keras.activations.elu:
            return lambda x: np.where(x > 0, x, np.exp(x) - 1)
        case keras.activations.selu:
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            return lambda x: scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        case keras.activations.softplus:
            return lambda x: np.log1p(np.exp(x))
        case keras.activations.softsign:
            return lambda x: x / (1 + np.abs(x))
        case keras.activations.exponential:
            return lambda x: np.exp(x)
        case keras.activations.hard_silu:
            return lambda x: x * np.minimum(1, np.maximum(0, (x + 1) / 2))
        case _:
            return lambda x: ops.convert_to_numpy(act(ops.convert_to_tensor(x)))


def gather_weights_and_activation(model: keras.Sequential):
    ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    acts: list[Callable[[np.ndarray], np.ndarray]] = []
    for layer in model.layers:
        layer: keras.layers.EinsumDense
        w, b = layer.get_weights()
        act = keras_act_to_numpy(layer.activation)
        if w.ndim == 3:
            w, b = w[..., None], b[..., None]
        ws.append(w)
        bs.append(b)
        acts.append(act)
    return ws, bs, acts


class ReplayDenseTable(ReplayOperationBase):
    handles = (QDenseT,)

    __input_quantizer_handled__ = True

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        op: QDenseT = self.op  # type: ignore

        out = np.broadcast_to(inputs[..., None], (op.n_in, op.n_out))  # type: ignore
        out = mirror_quantizer(op.iq, out)

        l, h, s = out.lhs

        N: np.ndarray = np.round((h - l) / s).astype(np.uint32) + 1

        model = op.module

        ws, bs, acts = gather_weights_and_activation(model)

        out_shape: tuple[int, ...] = tuple(int(x) for x in model.output_shape[1:])  # type: ignore
        r: list[np.ndarray] = [None] * prod(out_shape)  # type: ignore
        n, loc = np.unique(N, return_inverse=True)

        for i in range(n.size):
            mask = loc == i
            _l, _h = l[mask], h[mask]
            inp = np.linspace(_l, _h, n[i])

            _out = inp[..., None]

            idxs = np.where(mask.ravel())[0]
            for w, b, act in zip(ws, bs, acts):
                w = w[mask]
                b = b[mask]
                _out = act(np.einsum('...ni,nij->...nj', _out, w, optimize='optimal') + b)
            _out = _out[..., 0]

            for j, idx in enumerate(idxs):
                r[idx] = _out[..., j]

        assert all(v is not None for v in r)

        toq = op.toq
        toq_internal: FixedPointQuantizerBase = toq.quantizer
        kk, ki, kf = toq_internal.kif

        _shape = (1,) + out.shape
        kk = toq_internal.bw_mapper.bw_to_x(kk, _shape)
        ki = toq_internal.bw_mapper.bw_to_x(ki, _shape)
        kf = toq_internal.bw_mapper.bw_to_x(kf, _shape)

        k, i, f = map(lambda x: to_np_arr(x).astype(np.int32).ravel(), (kk, ki, kf))

        round_mode, overflow_mode = toq_internal.round_mode, toq_internal.overflow_mode
        round_mode = round_mode[2:] if round_mode.startswith('S_') else round_mode
        for arr, _k, _i, _f in zip(r, k, i, f):
            arr[:] = _quantize(arr, _k, _i, _f, overflow_mode, round_mode)

        rr: list[FixedVariable] = [None] * len(r)  # type: ignore
        _vars = out.ravel()._vars
        for i in range(len(r)):
            rr[i] = _vars[i].lookup(r[i])
        out = FixedVariableArray(np.array(rr).reshape(out_shape), solver_options=out.solver_options)
        out = np.sum(out, axis=0)  # type: ignore
        return out


__all__ = ['ReplayDenseTable']
