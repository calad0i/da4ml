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
)
from hgq.layers.core.base import MultipleQuantizers, Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase
from keras.layers import ReLU
from keras.src.layers.pooling.base_global_pooling import BaseGlobalPooling
from keras.src.layers.pooling.base_pooling import BasePooling
from keras.src.ops.numpy import Add, Concatenate, GetItem, Moveaxis, Repeat, Subtract, Transpose

from ...trace import FixedVariableArray
from ...trace.ops import conv, einsum, pool, quantize, reduce, relu


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
        _inputs = inputs[0] if len(inputs) == 1 else inputs

        if not isinstance(self.op, hgq.layers.QLayerBase):
            r = self.call(*args, **kwargs)
            return r if isinstance(r, tuple) else (r,)

        layer: hgq.layers.QLayerBase = self.op

        if layer.enable_iq:
            if isinstance(_inputs, Sequence):
                assert isinstance(layer.iq, MultipleQuantizers)
                _inputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.iq.quantizers, _inputs))
            else:
                assert isinstance(layer.iq, Quantizer), f'Expected iq to be a Quantizer, got {type(layer.iq)}'
                _inputs = mirror_quantizer(layer.iq, _inputs)

        outputs = self.call(_inputs, **kwargs)

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
    handles = (keras.layers.Reshape, keras.layers.Flatten)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: keras.layers.Reshape = self.op
        if isinstance(layer, keras.layers.Flatten):
            return inputs.flatten()
        else:
            return inputs.reshape(layer.target_shape)


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
        assert op == 'avg', 'Only average pooling is supported now'

        data_format = self.op.data_format
        if data_format == 'channels_first':
            inputs = FixedVariableArray(np.moveaxis(inputs._vars, 1, -1), inputs.solver_options)

        if isinstance(self.op, BaseGlobalPooling):
            pool_dim = self.op.input_spec.ndim - 2  # type: ignore
            opr = lambda a, b: a + b
            pool_size = prod(inputs.shape[:-1])
            out = reduce(opr, inputs, axis=tuple(range(pool_dim)), keepdims=self.op.keepdims)
            if op == 'avg':
                out = out * (1 / pool_size)
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
            if op == 'avg':
                out = out * (1 / prod(pool_size))

        if data_format == 'channels_first':
            out = FixedVariableArray(np.moveaxis(out._vars, -1, 1), out.solver_options)

        return out


class MirrorRepeatVector(MirrorOperationBase):
    handles = (keras.layers.RepeatVector,)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: keras.layers.RepeatVector = self.op
        if layer.n == 1:
            return inputs
        return FixedVariableArray(np.repeat(inputs._vars, layer.n, axis=0), inputs.solver_options)


class MirrorGetItem(MirrorOperationBase):
    handles = (GetItem,)

    def call(self, x: FixedVariableArray, key):
        if isinstance(key, list):
            key = tuple(key)
        return x[None][key][0]


class MirrorAdd(MirrorOperationBase):
    handles = (Add,)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray):
        return x1 + x2


class MirrorSubtract(MirrorOperationBase):
    handles = (Subtract,)

    def call(self, x1: FixedVariableArray, x2: FixedVariableArray):
        return x1 - x2


class MirrorConcatenate(MirrorOperationBase):
    handles = (Concatenate,)

    def call(self, xs: Sequence[FixedVariableArray]):
        axis = self.op.axis
        # return backend.numpy.concatenate(xs, axis=self.axis)
        return FixedVariableArray(np.concatenate([x._vars[None] for x in xs], axis=axis)[0], xs[0].solver_options)


class MirrorRepeat(MirrorOperationBase):
    handles = (Repeat,)

    def call(self, x: FixedVariableArray):
        repeats, axis = self.op.repeats, self.op.axis
        return FixedVariableArray(np.repeat(x._vars[None], repeats, axis=axis)[0], x.solver_options)


class MirrorTranspose(MirrorOperationBase):
    handles = (Transpose,)

    def call(self, x: FixedVariableArray):
        axes = self.op.axes
        return FixedVariableArray(np.transpose(x._vars[None], axes)[0], x.solver_options)


class MirrorMoveaxis(MirrorOperationBase):
    handles = (Moveaxis,)

    def call(self, x: FixedVariableArray):
        source, destination = self.op.source, self.op.destination
        return FixedVariableArray(np.moveaxis(x._vars[None], source, destination)[0], x.solver_options)
