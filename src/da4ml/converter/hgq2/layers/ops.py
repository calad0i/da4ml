from collections.abc import Sequence

import hgq
import keras
import numpy as np
from hgq.layers import (
    QAdd,
    QDot,
    QEinsum,
    QMaximum,
    QMeanPow2,
    QMinimum,
    QMultiply,
    QSubtract,
    QSum,
)
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

from ....trace import FixedVariableArray
from ....trace.ops import einsum
from ._base import ReplayOperationBase


class ReplayQDot(ReplayOperationBase):
    handles = (QDot, keras.layers.Dot)

    def call(self, inputs: tuple[FixedVariableArray, FixedVariableArray]) -> FixedVariableArray:
        layer: QDot | keras.layers.Dot = self.op
        assert not layer.normalize, 'normalize is not supported in mirror operation'

        axes = layer.axes
        return np.dot(inputs[0][None], inputs[1][None], axes=axes)[0]  # type: ignore


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


class ReplayNoOp(ReplayOperationBase):
    __noop_layers = []
    for k, v in keras.layers.__dict__.items():
        name = k.lower()
        if 'dropout' in name or 'random' in name or 'noise' in name:
            __noop_layers.append(v)

    handles = tuple(__noop_layers)

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


__all__ = [
    'ReplayQDot',
    'ReplayReshape',
    'ReplayMerge',
    'ReplayRepeatVector',
    'ReplayGetItem',
    'ReplayReduction',
    'ReplayQReduction',
    'ReplayArithmetic',
    'ReplayConcatenate',
    'ReplayRepeat',
    'ReplayTranspose',
    'ReplayMoveaxis',
    'ReplayNoOp',
    'ReplayQEinsum',
    'ReplayEinsum',
    'ReplayMatmul',
    'ReplayAbs',
]
