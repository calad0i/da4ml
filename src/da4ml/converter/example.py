import numpy as np

from ..trace import FixedVariableArray
from ..trace.ops import einsum, quantize, relu
from .plugin import DAISTracerPluginBase


def operation(inp):
    """An example operation to be traced. One can use numpy-based operations along
    with DAIS traceable operations provided in `da4ml.trace.ops`.
    """
    w = np.arange(-60, 60).reshape(4, 5, 6).astype(np.float32) / 2**7
    inp = quantize(inp, 1, 7, 0)  # Input must be quantized before any non-trivial operation
    out1 = relu(inp)  # Only activation supported for now; can attach quantization at the same time

    # many native numpy operations are supported
    out2 = inp[:, 1:3].transpose()
    out2 = quantize(np.sin(out2), 1, 0, 7, 'SAT', 'RND')
    out2 = np.repeat(out2, 2, axis=0) * 3 + 4
    out2 = np.amax(np.stack([out2, -out2 * 2], axis=0), axis=0)

    out3 = quantize(out2 @ out1, 1, 10, 2)  # can also be einsum here
    out = einsum('ijk,ij->ik', w, out3)  # CMVM optimization is performed for all
    return out


class ExampleModel:
    """A simple example model class for showcasing DAIS tracer plugin usage."""

    def __init__(self, input_shape: tuple[int, ...] | None = None):
        self.input_shape = input_shape

    def __call__(self, x):
        return operation(x)


class ExampleDAISTracer(DAISTracerPluginBase):
    """An example DAIS tracer plugin for the ExampleModel. Two methods must be implemented:
    - `get_input_shapes`
    - `apply_model`

    This plugin must be registered as an entry point under the group `dais_tracer.plugins`.
    The entry name should be the module name where the model class is defined. In this case,
    since the target model class `ExampleModel` is defined in `da4ml.converter.example_plugin`, the entry point
    should be registered under the name `da4ml`. See `pyproject.toml` for an example.
    """

    model: ExampleModel

    def get_input_shapes(self):
        return [self.model.input_shape] if self.model.input_shape is not None else None

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FixedVariableArray, ...],
    ) -> tuple[dict[str, FixedVariableArray], list[str]]:
        assert len(inputs) == 1, 'ExampleModel expects a single input.'
        x = inputs[0]
        out = operation(x)
        return {'output_name': out}, ['output_name']
