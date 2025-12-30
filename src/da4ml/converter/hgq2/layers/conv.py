import numpy as np
from hgq.layers import (
    QConv1D,
    QConv2D,
    QConv3D,
)
from keras import ops
from keras.src.ops.image import ExtractPatches, extract_patches_3d

from ....trace import FixedVariableArray
from ....trace.ops import conv
from ._base import ReplayOperationBase, to_np_arr


def symbolic_extract_patches_3d(
    images: FixedVariableArray,
    size: tuple[int, int, int],
    strides: tuple[int, int, int],
    dilation_rate: tuple[int, int, int],
    padding: str,
    data_format: str,
) -> FixedVariableArray:
    img_tensor = ops.reshape(ops.arange(images.size), images.shape)
    out_tensor = extract_patches_3d(
        img_tensor[None],
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,  # type: ignore
        padding=padding,
        data_format=data_format,
    )[0]
    out_index: np.ndarray = ops.convert_to_numpy(out_tensor)  # type: ignore
    images = images.ravel()[out_index]

    return images


def symbolic_extract_patches(
    images: FixedVariableArray,
    size: tuple[int, ...] | int,
    strides: tuple[int, ...] | int,
    dilation_rate: tuple[int, ...] | int,
    padding: str,
    data_format: str,
):
    rank = images.ndim - 1
    size = (size,) * rank if isinstance(size, int) else size
    strides = (strides,) * rank if isinstance(strides, int) else strides
    dilation_rate = (dilation_rate,) * rank if isinstance(dilation_rate, int) else dilation_rate

    assert rank == len(size) == len(strides) == len(dilation_rate), (
        f'Invalid rank {rank} for size {size}, strides {strides}, dilation_rate {dilation_rate}'
    )

    pad_rank = 3 - rank
    _size: tuple[int, int, int] = (1,) * pad_rank + size  # type: ignore
    _strides: tuple[int, int, int] = (1,) * pad_rank + strides  # type: ignore
    _dilation_rate: tuple[int, int, int] = (1,) * pad_rank + dilation_rate  # type: ignore

    _pad = (1,) * pad_rank
    if data_format == 'channels_first':
        images = np.moveaxis(images, 0, -1)  # type: ignore

    *spa, ch = images.shape
    images = images.reshape(*_pad, *spa, ch)

    r = symbolic_extract_patches_3d(
        images,
        size=_size,
        strides=_strides,
        dilation_rate=_dilation_rate,
        padding=padding,
        data_format='channels_last',
    )

    return r.reshape(r.shape[pad_rank:])


class ReplayExtractPatches(ReplayOperationBase):
    handles = (ExtractPatches,)

    def call(self, images: FixedVariableArray) -> FixedVariableArray:
        op: ExtractPatches = self.op
        pixel_shape = op.size
        strides = op.strides
        dilation_rate: int | tuple[int, int] = op.dilation_rate
        padding = op.padding
        data_format = op.data_format

        if strides is None:
            strides = 1

        return symbolic_extract_patches(images, pixel_shape, strides, dilation_rate, padding, data_format)


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
