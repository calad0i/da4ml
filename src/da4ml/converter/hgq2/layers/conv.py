import numpy as np
from hgq.layers import (
    QConv1D,
    QConv2D,
    QConv3D,
)
from keras.src.ops.image import ExtractPatches

from ....trace import FixedVariableArray
from ....trace.ops import conv, im2col, pad
from ._base import ReplayOperationBase, to_np_arr


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

        assert dilation_rate == 1 or all(d == 1 for d in dilation_rate), (
            f'Non-one dilation rate is not yet supported, got {dilation_rate} in layer {layer.name}'
        )
        if layer.data_format == 'channels_first':
            inputs = np.moveaxis(inputs, 0, -1)  # type: ignore

        outputs = conv(inputs, qkernel, qbias, strides=strides, padding=padding, format=layer.data_format, groups=groups)

        if layer.data_format == 'channels_first':
            outputs: FixedVariableArray = np.moveaxis(outputs, -1, 0)  # type: ignore

        return outputs


class ReplayExtractPatches(ReplayOperationBase):
    handles = (ExtractPatches,)

    def call(self, images: FixedVariableArray) -> FixedVariableArray:
        op: ExtractPatches = self.op
        pixel_shape = op.size
        strides = op.strides
        dilation_rate = op.dilation_rate
        padding = op.padding
        data_format = op.data_format

        if strides is None:
            strides = 1
        if not isinstance(strides, tuple):
            strides = (strides,) * len(pixel_shape)

        assert dilation_rate == 1, 'Non-one dilation rate is not yet supported'

        if data_format == 'channels_first':
            images = np.moveaxis(images, 0, -1)  # type: ignore

        images = pad(pixel_shape, padding, images)
        images = im2col(images, pixel_shape, strides)

        if data_format == 'channels_first':
            images = np.moveaxis(images, -1, 0)  # type: ignore

        return images


__all__ = ['ReplayQConv', 'ReplayExtractPatches']
