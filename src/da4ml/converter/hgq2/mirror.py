import typing

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

from ...trace import FixedVariableArray
from ...trace.ops import conv, einsum, quantize, relu


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

    def __init__(self, layer: 'keras.Layer'):
        assert isinstance(layer, self.handles)
        self.layer = layer

    def call(self, inputs) -> tuple[FixedVariableArray, ...] | FixedVariableArray: ...

    def __call__(self, inputs: tuple[FixedVariableArray, ...]) -> tuple[FixedVariableArray, ...]:
        _inputs = inputs[0] if len(inputs) == 1 else inputs

        if not isinstance(self.layer, hgq.layers.QLayerBase):
            r = self.call(_inputs)
            return r if isinstance(r, tuple) else (r,)

        layer: hgq.layers.QLayerBase = self.layer

        if layer.enable_iq:
            if isinstance(_inputs, tuple):
                assert isinstance(layer.iq, MultipleQuantizers)
                _inputs = tuple(mirror_quantizer(q, v) for q, v in zip(layer.iq.quantizers, _inputs))
            else:
                assert isinstance(layer.iq, Quantizer)
                _inputs = mirror_quantizer(layer.iq, _inputs)

        outputs = self.call(_inputs)

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

    def __init__(self, layer: 'Quantizer'):
        super().__init__(layer)
        assert isinstance(layer.quantizer, FixedPointQuantizerBase)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        return mirror_quantizer(self.layer, inputs)


class MirrorQDense(MirrorOperationBase):
    handles = (QDense, QEinsumDense, QEinsumDenseBatchnorm, QBatchNormDense, QBatchNormalization)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer = self.layer
        if isinstance(layer, (QDense, QBatchNormDense)):
            qkernel = layer.qkernel
            qbias = layer.qbias
            eq = '...c,cC->...C'
        elif isinstance(layer, (QEinsumDense, QEinsumDenseBatchnorm)):
            qkernel = layer.qkernel
            qbias = layer.qbias
            eq = layer.equation
        elif isinstance(layer, QBatchNormalization):
            qkernel, qbias = layer.qscaler_and_qoffset
            dim = inputs._vars.ndim
            axis = layer.axis
            idx = ''.join(chr(ord('a') + i) for i in range(dim))
            eq = f'{idx,idx[axis]}->{idx}'
        else:
            raise TypeError(f'Unsupported layer type: {type(layer)}')

        qkernel = np.array(qkernel)
        qbias = np.array(qbias) if qbias is not None else None
        return (einsum(eq, inputs[None], qkernel) + qbias)[0]


class MirrorQConv(MirrorOperationBase):
    handles = (QConv1D, QConv2D, QConv3D)

    def call(self, inputs: FixedVariableArray) -> FixedVariableArray:
        layer: QConv1D = self.layer
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
        layer: keras.layers.Reshape = self.layer
        if isinstance(layer, keras.layers.Flatten):
            return inputs.flatten()
        else:
            return inputs.reshape(layer.target_shape)


class MirrorMerge(MirrorOperationBase):
    handles = (keras.layers.Add, keras.layers.Concatenate, hgq.layers.QAdd)

    def call(self, inputs: tuple[FixedVariableArray, FixedVariableArray]) -> FixedVariableArray:
        layer: keras.layers.Layer = self.layer
        if isinstance(layer, (keras.layers.Add, hgq.layers.QAdd)):
            return inputs[0] + inputs[1]
        elif isinstance(layer, keras.layers.Concatenate):
            axis = layer.axis
            data = np.concatenate([v._vars for v in inputs], axis=axis)
            return FixedVariableArray(data, inputs[0].solver_options)
        else:
            raise TypeError(f'Unsupported layer type: {type(layer)}')
