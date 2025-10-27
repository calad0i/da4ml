import typing
from collections.abc import Sequence
from math import prod
from typing import Any

import hgq
import keras
import numpy as np
from hgq.layers import (
    QAdd,
    QBatchNormalization,
    QBatchNormDense,
    QConv1D,
    QConv2D,
    QConv3D,
    QDense,
    QDot,
    QEinsum,
    QEinsumDense,
    QEinsumDenseBatchnorm,
    QLinformerAttention,
    QMaximum,
    QMeanPow2,
    QMinimum,
    QMultiHeadAttention,
    QMultiply,
    QSoftmax,
    QSubtract,
    QSum,
    QUnaryFunctionLUT,
)
from hgq.layers.core.base import MultipleQuantizers, Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase
from keras.layers import ReLU
from keras.ops import convert_to_numpy
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling
from keras.src.layers.pooling.base_pooling import BasePooling
from keras.src.ops.numpy import (
    Abs,
    Absolute,
    Add,
    Concatenate,
    Divide,
    Dot,
    Einsum,
    GetItem,
    Matmul,
    Max,
    Maximum,
    Min,
    Minimum,
    Moveaxis,
    Multiply,
    Ravel,
    Repeat,
    Reshape,
    Subtract,
    Sum,
    Transpose,
    TrueDivide,
)

from ...trace import FixedVariableArray
from ...trace.ops import conv, einsum, pool, quantize, relu


def to_np_arr(x: Any) -> np.ndarray:
    return np.asarray(convert_to_numpy(x))


def mirror_quantizer(q: Quantizer, v: FixedVariableArray) -> FixedVariableArray:
    q_internal: FixedPointQuantizerBase = q.quantizer
    k, i, f = (to_np_arr(x).astype(np.int8)[0] for x in q_internal.kif)
    round_mode, overflow_mode = q_internal.round_mode, q_internal.overflow_mode
    return quantize(v, k, i, f, overflow_mode=overflow_mode, round_mode=round_mode)


_registry: dict[type, 'type[ReplayOperationBase]'] = {}


class HandlerRegMeta(type):
    """Metaclass for automatic registration of handler classes."""

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, typing.Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == 'ReplayOperationBase':
            return cls

        handles: type | tuple[type, ...] = namespace['handles']
        if not isinstance(handles, tuple):
            handles = (handles,)

        for handle in handles:
            _registry[handle] = cls  # type: ignore
        return cls


class ReplayOperationBase(metaclass=HandlerRegMeta):
    handles: tuple[type, ...] = ()
    __activation_handled__ = False
    __input_quantizer_handled__ = False
    __output_quantizer_handled__ = False

    def __init__(self, layer: 'keras.Operation'):
        assert isinstance(layer, self.handles)
        self.op: Any = layer

    def call(self, *args, **kwargs) -> tuple[FixedVariableArray, ...] | FixedVariableArray: ...

    def __call__(self, *args, **kwargs) -> tuple[FixedVariableArray, ...]:
        assert all(not isinstance(a, FixedVariableArray) for a in kwargs.values())

        if not isinstance(self.op, hgq.layers.QLayerBase):
            r = self.call(*args, **kwargs)
            return r if isinstance(r, tuple) else (r,)

        layer: hgq.layers.QLayerBase = self.op
        assert kwargs.pop('training', False) is False, 'Training mode is not supported in mirror operation'
        assert kwargs.pop('mask', None) is None, 'Masking is not supported in mirror operation'

        if not self.__input_quantizer_handled__:
            assert len(args) == 1
            inputs = args[0]

            if layer.enable_iq:
                if isinstance(inputs, Sequence):
                    assert isinstance(layer.iq, MultipleQuantizers)
                    inputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.iq.quantizers, inputs))
                else:
                    assert isinstance(layer.iq, Quantizer), f'Expected iq to be a Quantizer, got {type(layer.iq)}'
                    inputs = mirror_quantizer(layer.iq, inputs)

            outputs = self.call(inputs, **kwargs)
        else:
            outputs = self.call(*args, **kwargs)

        if not self.__activation_handled__:
            activation = getattr(layer, 'activation', keras.activations.linear)
            if activation is not keras.activations.linear:
                if activation is keras.activations.relu:
                    if isinstance(outputs, tuple):
                        assert len(outputs) == 1, 'ReLU activation is expected to have a single output'
                        outputs = (relu(outputs[0]),)
                    else:
                        outputs = relu(outputs)
                else:
                    raise NotImplementedError(f'Activation {activation} is not supported in mirror operation')

        if layer.enable_oq and not self.__output_quantizer_handled__:
            if isinstance(outputs, tuple):
                assert isinstance(layer.oq, MultipleQuantizers)
                outputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.oq.quantizers, outputs))
            else:
                assert isinstance(layer.oq, Quantizer)
                outputs = mirror_quantizer(layer.oq, outputs)

        if isinstance(outputs, (FixedVariableArray, np.ndarray)):
            outputs = (outputs,)

        return outputs


class ReplayQuantizer(ReplayOperationBase):
    handles = (Quantizer,)

    def __init__(self, op: 'Quantizer'):
        super().__init__(op)
        assert isinstance(op.quantizer, FixedPointQuantizerBase)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return mirror_quantizer(self.op, inputs)


class ReplayQDense(ReplayOperationBase):
    handles = (QDense, QEinsumDense, QEinsumDenseBatchnorm, QBatchNormDense, keras.layers.EinsumDense)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        op = self.op
        if isinstance(op, (QDense, QBatchNormDense)):
            qkernel = op.qkernel
            qbias = op.qbias
            eq = '...c,cC->...C'
        elif isinstance(op, (QEinsumDense, QEinsumDenseBatchnorm)):
            qkernel = op.qkernel
            qbias = op.qbias
            eq = op.equation
        elif isinstance(op, keras.layers.EinsumDense):
            qkernel = op.kernel
            qbias = op.bias
            eq = op.equation
        else:
            raise TypeError(f'Unsupported layer type: {type(op)}')

        qkernel = to_np_arr(qkernel)
        qbias = to_np_arr(qbias) if qbias is not None else None
        return (einsum(eq, inputs[None], qkernel) + qbias)[0]


class ReplayQDot(ReplayOperationBase):
    handles = (QDot, keras.layers.Dot)

    def call(self, inputs: tuple[FixedVariableArray, FixedVariableArray]) -> FixedVariableArray:
        layer: QDot | keras.layers.Dot = self.op
        assert not layer.normalize, 'normalize is not supported in mirror operation'

        axes = layer.axes
        return np.dot(inputs[0][None], inputs[1][None], axes=axes)[0]  # type: ignore


class ReplayQBatchNormalization(ReplayOperationBase):
    handles = (QBatchNormalization,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: QBatchNormalization = self.op
        scale, bias = map(to_np_arr, layer.qscaler_and_qoffset)
        shape = layer._shape[1:]
        return inputs * scale.reshape(shape) + bias.reshape(shape)


class ReplayQConv(ReplayOperationBase):
    handles = (QConv1D, QConv2D, QConv3D)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: QConv1D | QConv2D | QConv3D = self.op
        qkernel = to_np_arr(layer.qkernel)
        qbias = to_np_arr(layer.qbias) if layer.qbias is not None else None
        strides = layer.strides
        padding = layer.padding
        dilation_rate = layer.dilation_rate
        groups = layer.groups

        assert dilation_rate == 1 or all(d == 1 for d in dilation_rate), 'Dilation rate is not supported in mirror operation'
        if layer.data_format == 'channels_first':
            shape = (0,) + tuple(range(2, len(inputs.shape))) + (1,)
            inputs = inputs.transpose(shape)

        outputs = conv(inputs, qkernel, qbias, strides=strides, padding=padding, format=layer.data_format, groups=groups)

        return outputs


class ReplayReLU(ReplayOperationBase):
    handles = (ReLU,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return relu(inputs)


class ReplayReshape(ReplayOperationBase):
    handles = (keras.layers.Reshape, keras.layers.Flatten, Reshape, Ravel)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        if isinstance(self.op, (keras.layers.Flatten, Ravel)):
            return inputs.ravel()
        elif isinstance(self.op, keras.layers.Reshape):
            return inputs.reshape(self.op.target_shape)
        elif isinstance(self.op, Reshape):
            return inputs.reshape(self.op.newshape[1:])
        else:
            raise TypeError(f'Unsupported layer type: {type(self.op)}')


class ReplayMerge(ReplayOperationBase):
    handles = (keras.layers.Add, keras.layers.Concatenate, QAdd)

    def call(self, inputs: tuple[FixedVariableArray, FixedVariableArray]) -> FixedVariableArray:
        op: keras.Operation = self.op
        if isinstance(op, (keras.layers.Add, hgq.layers.QAdd)):
            return inputs[0] + inputs[1]
        elif isinstance(op, keras.layers.Concatenate):
            axis = op.axis
            data = np.concatenate([v._vars for v in inputs], axis=axis)
            return FixedVariableArray(data, inputs[0].solver_options)
        else:
            raise TypeError(f'Unsupported layer type: {type(op)}')


class ReplayPool(ReplayOperationBase):
    handles = (
        hgq.layers.QAvgPool1D,
        hgq.layers.QAvgPool2D,
        hgq.layers.QAvgPool3D,
        hgq.layers.QMaxPool1D,
        hgq.layers.QMaxPool2D,
        hgq.layers.QMaxPool3D,
        hgq.layers.QGlobalAveragePooling1D,
        hgq.layers.QGlobalMaxPooling1D,
        hgq.layers.QGlobalAveragePooling2D,
        hgq.layers.QGlobalMaxPooling2D,
        hgq.layers.QGlobalAveragePooling3D,
        hgq.layers.QGlobalMaxPooling3D,
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalAveragePooling3D,
        keras.layers.GlobalMaxPooling3D,
    )

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        cname = self.op.__class__.__name__
        if 'Max' in cname:
            op = 'max'
        else:
            assert 'Average' in cname, f'Unsupported global pooling layer: {cname}'
            op = 'avg'

        data_format = self.op.data_format
        if data_format == 'channels_first':
            inputs = np.moveaxis(inputs, 1, -1)  # type: ignore

        if isinstance(self.op, BaseGlobalPooling):
            pool_dim = self.op.input_spec.ndim - 2  # type: ignore
            axis = tuple(range(pool_dim))
            keepdims = self.op.keepdims

            if op == 'max':
                out = np.amax(inputs, axis=axis, keepdims=keepdims)  # type: ignore
            elif op == 'avg':
                pool_size = prod(inputs.shape[:-1])
                out = np.sum(inputs, axis=axis, keepdims=keepdims) / pool_size  # type: ignore
        else:
            assert isinstance(self.op, BasePooling), f'Unsupported pooling layer: {type(self.op)}'
            pool_size = self.op.pool_size
            strides = self.op.strides
            padding = self.op.padding
            pool_dim = len(pool_size)
            out = pool(
                inputs,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                pool_type=op,
            )
        if data_format == 'channels_first':
            out = np.moveaxis(out, -1, 1)  # type: ignore

        return out  # type: ignore


class ReplayRepeatVector(ReplayOperationBase):
    handles = (keras.layers.RepeatVector,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: keras.layers.RepeatVector = self.op
        if layer.n == 1:
            return inputs
        # return FixedVariableArray(np.repeat(inputs._vars, layer.n, axis=0), inputs.solver_options)
        return np.repeat(inputs[None], layer.n, axis=0)[0]  # type: ignore


class ReplayGetItem(ReplayOperationBase):
    handles = (GetItem,)

    def call(self, x: FixedVariableArray, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[None][key][0]


class ReplayReduction(ReplayOperationBase):
    handles = (Sum, Max, Min)

    def call(self, x: FixedVariableArray, axis=None, keepdims=False):
        if isinstance(self.op, Sum):
            op = np.sum
        elif isinstance(self.op, Max):
            op = np.amax
        elif isinstance(self.op, Min):
            op = np.amin
        return op(x[None], axis=axis, keepdims=keepdims)[0]  # type: ignore


class ReplayQReduction(ReplayOperationBase):
    handles = (QSum, QMeanPow2)

    def call(self, x: FixedVariableArray):
        layer: QSum = self.op
        axes, scale, keepdims = layer.axes, layer.scale, layer.keepdims
        return np.sum(x[None], axis=axes, keepdims=keepdims)[0] * scale  # type: ignore


class ReplayArithmetic(ReplayOperationBase):
    handles = (Add, Subtract, Multiply, QMultiply, TrueDivide, Divide, QSubtract, QMaximum, QMinimum, Maximum, Minimum)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray):
        name = self.op.__class__.__name__
        if name.startswith('Q'):
            name = name[1:]
        match name:
            case 'Add':
                return x1 + x2
            case 'Subtract':
                return x1 - x2
            case 'Multiply':
                return x1 * x2
            case 'TrueDivide' | 'Divide':
                return x1 / x2
            case 'Maximum':
                return np.maximum(x1, x2)  # type: ignore
            case 'Minimum':
                return np.minimum(x1, x2)  # type: ignore
            case _:
                raise TypeError(f'Unsupported arithmetic operation: {type(self.op)}')


class ReplayConcatenate(ReplayOperationBase):
    handles = (Concatenate,)

    def call(self, xs: Sequence[FixedVariableArray]):
        axis = self.op.axis
        # return backend.numpy.concatenate(xs, axis=self.axis)
        # return FixedVariableArray(np.concatenate([x._vars[None] for x in xs], axis=axis)[0], xs[0].solver_options)
        return np.concatenate([x[None] for x in xs], axis=axis)[0]  # type: ignore


class ReplayRepeat(ReplayOperationBase):
    handles = (Repeat,)

    def call(self, x: FixedVariableArray):
        repeats, axis = self.op.repeats, self.op.axis
        # return FixedVariableArray(np.repeat(x._vars[None], repeats, axis=axis)[0], x.solver_options)
        return np.repeat(x[None], repeats, axis=axis)[0]  # type: ignore


class ReplayTranspose(ReplayOperationBase):
    handles = (Transpose,)

    def call(self, x: FixedVariableArray):
        axes = self.op.axes
        return np.transpose(x, axes)  # type: ignore


class ReplayMoveaxis(ReplayOperationBase):
    handles = (Moveaxis,)

    def call(self, x: FixedVariableArray):
        source, destination = self.op.source, self.op.destination
        return np.moveaxis(x[None], source, destination)[0]  # type: ignore


noop_layers = []
for k, v in keras.layers.__dict__.items():
    name = k.lower()
    if 'dropout' in name or 'random' in name or 'noise' in name:
        noop_layers.append(v)


class ReplayNoOp(ReplayOperationBase):
    handles = tuple(noop_layers)

    def call(self, x: FixedVariableArray, training=False) -> FixedVariableArray:
        assert not training, 'Training mode is not supported in mirror operation'
        return x


class ReplayQEinsum(ReplayOperationBase):
    handles = (QEinsum,)

    def call(self, inputs: tuple[FixedVariableArray, ...]) -> FixedVariableArray:
        layer: QEinsum = self.op
        eq = layer.equation
        return einsum(eq, *inputs)


class ReplayEinsum(ReplayOperationBase):
    handles = (Einsum,)

    def call(self, *operands: FixedVariableArray) -> FixedVariableArray:
        layer: Einsum = self.op
        eq = layer.subscripts
        operands = [operand[None] for operand in operands]  # type: ignore
        return einsum(eq, *operands)[0]


class ReplayMatmul(ReplayOperationBase):
    handles = (Matmul, Dot)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray) -> FixedVariableArray:
        return x1 @ x2


class ReplayAbs(ReplayOperationBase):
    handles = (Absolute, Abs)

    def call(self, x: FixedVariableArray) -> FixedVariableArray:
        return np.abs(x)  # type: ignore


class ReplayQFunctionLUT(ReplayOperationBase):
    __activation_handled__ = True
    handles = (QUnaryFunctionLUT,)

    def call(self, x: FixedVariableArray) -> FixedVariableArray:
        op: QUnaryFunctionLUT = self.op

        def activation(x) -> np.ndarray:
            kx = keras.ops.convert_to_tensor(x[None])
            kx = op.activation(kx)
            return keras.ops.convert_to_numpy(kx[0])  # type: ignore

        return x.apply(activation)


class ReplayQSoftmax(ReplayOperationBase):
    handles = (QSoftmax,)

    def call(self, inputs: FixedVariableArray, mask: None | FixedVariableArray = None) -> FixedVariableArray:
        op: QSoftmax = self.op
        inputs = inputs[None]

        if op.stable:
            inputs = np.amax(inputs, axis=op.axes, keepdims=True) - inputs  # type: ignore

        exp_inp = ReplayQFunctionLUT(op.exp_table)(inputs[0])[0]

        if mask is not None:
            exp_inp = mask[0] * exp_inp

        sums = np.sum(exp_inp[None], axis=op.axes, keepdims=True)[0]  # type: ignore
        divisor = ReplayQFunctionLUT(op.inv_table)(sums)[0]

        return exp_inp * divisor


def _compute_attention_mask(
    query,
    value,
    query_mask=None,
    value_mask=None,
    key_mask=None,
    attention_mask=None,
    use_causal_mask=False,
):
    masks = []
    if query_mask is not None:
        masks.append(np.expand_dims(query_mask, -1))  # [Q, 1]
    if value_mask is not None:
        masks.append(np.expand_dims(value_mask, -2))  # [1, V]
    if key_mask is not None:
        masks.append(np.expand_dims(key_mask, -2))  # [1, V]
    if use_causal_mask:
        q = query.shape[0]
        v = q if value is None else value.shape[0]
        masks.append(np.tril(np.ones((q, v), dtype='uint8')))  # [Q, V]
    masks.append(attention_mask)
    if not masks:
        return None

    if any(isinstance(m, FixedVariableArray) for m in masks):
        return np.prod(np.stack(masks, axis=0), axis=0)
    else:
        return None


def _masked_softmax(op, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # attention_scores = [B, N, T, S]
    if attention_mask is not None:
        # The expand dim happens starting from the `num_heads` dimension,
        # (<batch_dims>, num_heads, <query_attention_dims,
        # key_attention_dims>)
        mask_expansion_axis = -len(op._attention_axes) * 2 - 1
        for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
            attention_mask = np.expand_dims(attention_mask, axis=mask_expansion_axis)
    return ReplayQSoftmax(op._softmax)(attention_scores[0], mask=attention_mask)[0][None]


def _compute_attention(op: QMultiHeadAttention, query, key, value, attention_mask=None, training=None):
    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = einsum(op._dot_product_equation, key, query)

    attention_scores = _masked_softmax(op, attention_scores, attention_mask)

    # `context_layer` = [B, T, N, H]
    attention_output = einsum(op._combine_equation, attention_scores, value)
    return attention_output, attention_scores


class ReplayMHA(ReplayOperationBase):
    handles = (QMultiHeadAttention,)
    __input_quantizer_handled__ = True
    __output_quantizer_handled__ = True

    def call(
        self,
        query: FixedVariableArray,
        value: FixedVariableArray,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        op: QMultiHeadAttention = self.op

        if key is None:
            key = value

        _attention_mask = _compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        query = ReplayQDense(op._query_dense)(query)[0][None]
        key = ReplayQDense(op._key_dense)(key)[0][None]
        value = ReplayQDense(op._value_dense)(value)[0][None]

        attention_output, attention_scores = _compute_attention(op, query, key, value, _attention_mask)
        attention_output = ReplayQDense(op._output_dense)(attention_output[0])[0]

        if op.enable_oq:
            attention_output = mirror_quantizer(op.oq, attention_output)

        if return_attention_scores:
            return attention_output, attention_scores[0]
        return attention_output


class ReplayQLinformerAttention(ReplayMHA):
    handles = (QLinformerAttention,)

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        assert use_causal_mask is False, 'Causal mask is not supported in QLinformerAttention.'
        key = key if key is not None else value
        op: QLinformerAttention = self.op
        key = ReplayQDense(op._lin_k_proj)(key)[0]
        value = ReplayQDense(op._lin_v_proj)(value)[0]
        return super().call(
            query,
            value,
            key,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            return_attention_scores=return_attention_scores,
        )
