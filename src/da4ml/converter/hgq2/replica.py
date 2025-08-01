import typing
from collections.abc import Sequence
from math import prod
from typing import Any

import hgq
import keras
import numpy as np
from hgq.layers import (
    QBatchNormalization,
    QBatchNormDense,
    QConv1D,
    QConv2D,
    QConv3D,
    QDense,
    QEinsumDense,
    QEinsumDenseBatchnorm,
    QSum,
)
from hgq.layers.core.base import MultipleQuantizers, Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase
from keras.layers import ReLU
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling
from keras.src.layers.pooling.base_pooling import BasePooling
from keras.src.ops.numpy import (
    Add,
    Concatenate,
    Divide,
    GetItem,
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


def mirror_quantizer(q: Quantizer, v: FixedVariableArray) -> FixedVariableArray:
    q_internal: FixedPointQuantizerBase = q.quantizer
    k, i, f = (np.array(x, dtype=np.int8)[0] for x in q_internal.kif)
    round_mode, overflow_mode = q_internal.round_mode, q_internal.overflow_mode
    return quantize(v, k, i, f, overflow_mode=overflow_mode, round_mode=round_mode)


_registry: dict[type, 'type[MirrorOperationBase]'] = {}


class MirrorOperationMeta(type):
    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, typing.Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        if name == 'MirrorOperationBase':
            return cls

        handles: type | tuple[type, ...] = namespace['handles']
        if not isinstance(handles, tuple):
            handles = (handles,)

        for handle in handles:
            _registry[handle] = cls  # type: ignore
        return cls


class MirrorOperationBase(metaclass=MirrorOperationMeta):
    handles: tuple[type, ...] = ()

    def __init__(self, layer: 'keras.Operation'):
        assert isinstance(layer, self.handles)
        self.op: Any = layer

    def call(self, *args, **kwargs) -> tuple[FixedVariableArray, ...] | FixedVariableArray: ...

    def __call__(self, *args, **kwargs) -> tuple[FixedVariableArray, ...]:
        assert all(not isinstance(a, FixedVariableArray) for a in kwargs.values())
        assert all(isinstance(a, FixedVariableArray) or isinstance(a, Sequence) for a in args)
        inputs = args[0] if len(args) == 1 else args

        if not isinstance(self.op, hgq.layers.QLayerBase):
            r = self.call(*args, **kwargs)
            return r if isinstance(r, tuple) else (r,)

        layer: hgq.layers.QLayerBase = self.op
        assert kwargs.pop('training', False) is False, 'Training mode is not supported in mirror operation'
        assert kwargs.pop('mask', None) is None, 'Masking is not supported in mirror operation'

        if layer.enable_iq:
            if isinstance(inputs, Sequence):
                assert isinstance(layer.iq, MultipleQuantizers)
                inputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.iq.quantizers, inputs))
            else:
                assert isinstance(layer.iq, Quantizer), f'Expected iq to be a Quantizer, got {type(layer.iq)}'
                inputs = mirror_quantizer(layer.iq, inputs)

        outputs = self.call(inputs, **kwargs)

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

        if layer.enable_oq:
            if isinstance(outputs, tuple):
                assert isinstance(layer.oq, MultipleQuantizers)
                outputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.oq.quantizers, outputs))
            else:
                assert isinstance(layer.oq, Quantizer)
                outputs = mirror_quantizer(layer.oq, outputs)

        if isinstance(outputs, FixedVariableArray):
            outputs = (outputs,)

        return outputs


class MirrorQuantizer(MirrorOperationBase):
    handles = (Quantizer,)

    def __init__(self, op: 'Quantizer'):
        super().__init__(op)
        assert isinstance(op.quantizer, FixedPointQuantizerBase)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return mirror_quantizer(self.op, inputs)


class MirrorQDense(MirrorOperationBase):
    handles = (QDense, QEinsumDense, QEinsumDenseBatchnorm, QBatchNormDense, QBatchNormalization, keras.layers.EinsumDense)

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
        elif isinstance(op, QBatchNormalization):
            qkernel, qbias = op.qscaler_and_qoffset
            dim = inputs._vars.ndim
            axis = op.axis
            assert axis != 0, 'Cannot normalizing on batch axis'
            axis = axis - 1 if axis >= 0 else dim + axis
            idx = ''.join(chr(ord('a') + i) for i in range(dim))
            eq = f'...{idx},{idx[axis]}->...{idx}'
        else:
            raise TypeError(f'Unsupported layer type: {type(op)}')

        qkernel = np.array(qkernel)
        qbias = np.array(qbias) if qbias is not None else None
        return (einsum(eq, inputs[None], qkernel) + qbias)[0]


class MirrorQConv(MirrorOperationBase):
    handles = (QConv1D, QConv2D, QConv3D)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: QConv1D | QConv2D | QConv3D = self.op
        qkernel = np.array(layer.qkernel)
        qbias = np.array(layer.qbias) if layer.qbias is not None else None
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


class MirrorReLU(MirrorOperationBase):
    handles = (ReLU,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return relu(inputs)


class MirrorReshape(MirrorOperationBase):
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


class MirrorMerge(MirrorOperationBase):
    handles = (keras.layers.Add, keras.layers.Concatenate, hgq.layers.QAdd)

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


class MirrorPool(MirrorOperationBase):
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


class MirrorRepeatVector(MirrorOperationBase):
    handles = (keras.layers.RepeatVector,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: keras.layers.RepeatVector = self.op
        if layer.n == 1:
            return inputs
        # return FixedVariableArray(np.repeat(inputs._vars, layer.n, axis=0), inputs.solver_options)
        return np.repeat(inputs[None], layer.n, axis=0)[0]  # type: ignore


class MirrorGetItem(MirrorOperationBase):
    handles = (GetItem,)

    def call(self, x: FixedVariableArray, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[None][key][0]


class MirrorSum(MirrorOperationBase):
    handles = (Sum,)

    def call(self, x: FixedVariableArray, axis=None, keepdims=False):
        return np.sum(x[None], axis=axis, keepdims=keepdims)[0]  # type: ignore


class MirrorQSum(MirrorOperationBase):
    handles = (QSum,)

    def call(self, x: FixedVariableArray):
        layer: QSum = self.op
        axes, scale, keepdims = layer.axes, layer.scale, layer.keepdims
        return np.sum(x[None], axis=axes, keepdims=keepdims)[0] * scale  # type: ignore


class MirrorArithmetic(MirrorOperationBase):
    handles = (Add, Subtract, Multiply, TrueDivide, Divide)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray):
        match self.op.__class__.__name__:
            case 'Add':
                return x1 + x2
            case 'Subtract':
                return x1 - x2
            case 'Multiply':
                return x1 * x2
            case 'TrueDivide' | 'Divide':
                return x1 / x2
            case _:
                raise TypeError(f'Unsupported arithmetic operation: {type(self.op)}')


class MirrorConcatenate(MirrorOperationBase):
    handles = (Concatenate,)

    def call(self, xs: Sequence[FixedVariableArray]):
        axis = self.op.axis
        # return backend.numpy.concatenate(xs, axis=self.axis)
        # return FixedVariableArray(np.concatenate([x._vars[None] for x in xs], axis=axis)[0], xs[0].solver_options)
        return np.concatenate([x[None] for x in xs], axis=axis)[0]  # type: ignore


class MirrorRepeat(MirrorOperationBase):
    handles = (Repeat,)

    def call(self, x: FixedVariableArray):
        repeats, axis = self.op.repeats, self.op.axis
        # return FixedVariableArray(np.repeat(x._vars[None], repeats, axis=axis)[0], x.solver_options)
        return np.repeat(x[None], repeats, axis=axis)[0]  # type: ignore


class MirrorTranspose(MirrorOperationBase):
    handles = (Transpose,)

    def call(self, x: FixedVariableArray):
        axes = self.op.axes
        return np.transpose(x, axes)  # type: ignore


class MirrorMoveaxis(MirrorOperationBase):
    handles = (Moveaxis,)

    def call(self, x: FixedVariableArray):
        source, destination = self.op.source, self.op.destination
        return np.moveaxis(x[None], source, destination)[0]  # type: ignore


noop_layers = []
for k, v in keras.layers.__dict__.items():
    name = k.lower()
    if 'dropout' in name or 'random' in name or 'noise' in name:
        noop_layers.append(v)


class MirrorNoOp(MirrorOperationBase):
    handles = tuple(noop_layers)

    def call(self, x: FixedVariableArray) -> FixedVariableArray:
        return x
