from hgq.layers.table import QConvT1D, QConvT2D, QConvTBase

from ....trace import FixedVariableArray
from .conv import replay_extract_patches
from .dense_table import ReplayDenseTable


class ReplayConvTable(ReplayDenseTable):
    handles = (QConvT2D, QConvT1D)

    def call(self, inputs: FixedVariableArray):
        op: QConvTBase = self.op

        if op.rank == 1:
            inputs = inputs[:, :, None]

        inputs = replay_extract_patches(inputs, **op.im2col_params)

        if op.rank == 1:
            inputs = inputs[:, :, 0]

        return super().call(inputs)
