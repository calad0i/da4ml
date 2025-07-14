from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import keras
from keras import KerasTensor, Operation

from ...trace import FixedVariableArray, HWConfig
from ...trace.fixed_variable_array import FixedVariableArrayInput
from .replica import _registry


@dataclass
class OpObj:
    operation: Operation
    args: list
    kwargs: dict
    produces: tuple[KerasTensor, ...]
    requires: tuple[KerasTensor, ...]


def parse_model(model: keras.Model):
    operators: dict[int, list[OpObj]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        _oprs = []
        for node in nodes:
            assert isinstance(node.operation, keras.Operation)
            opr = OpObj(
                operation=node.operation,
                args=node.arguments.args,
                kwargs=node.arguments.kwargs,
                produces=node.outputs,
                requires=node.arguments.keras_tensors,
            )
            _oprs.append(opr)
        operators[depth] = _oprs
    return [operators[i] for i in range(max(operators.keys()), -1, -1)]


def replace_tensors(tensor_map: dict[KerasTensor, FixedVariableArray], obj: Any) -> Any:
    if isinstance(obj, KerasTensor):
        return tensor_map[obj]
    if isinstance(obj, list):
        return [replace_tensors(tensor_map, o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(replace_tensors(tensor_map, o) for o in obj)
    if isinstance(obj, dict):
        return {k: replace_tensors(tensor_map, v) for k, v in obj.items()}
    return obj


def _apply_nn(
    model: keras.Model, inputs: FixedVariableArray | Sequence[FixedVariableArray], verbose: bool = False
) -> tuple[FixedVariableArray, ...]:
    """
    Apply a keras model to a fixed variable array or a sequence of fixed variable arrays.

    Parameters
    ----------
    model : keras.Model
        The keras model to apply.
    inputs : FixedVariableArray or Sequence[FixedVariableArray]
        The input fixed variable array or sequence of fixed variable arrays.

    Returns
    -------
    tuple of FixedVariableArray
        A tuple containing the output(s) of the model as FixedVariableArray.
    """
    if isinstance(inputs, FixedVariableArray):
        inputs = (inputs,)

    assert len(model.inputs) == len(inputs), f'Model has {len(model.inputs)} inputs, got {len(inputs)}'
    tensor_map = {keras_tensor: da_tensor for keras_tensor, da_tensor in zip(model.inputs, inputs)}

    for ops in parse_model(model):
        for op in ops:
            assert all(t in tensor_map for t in op.requires)
            args = replace_tensors(tensor_map, op.args)
            kwargs: dict[str, Any] = replace_tensors(tensor_map, op.kwargs)
            if op.operation.__class__ is keras.layers.InputLayer:
                continue
            mirror_op = _registry[op.operation.__class__](op.operation)
            if verbose:
                print(f'Processing operation {op.operation.name} ({op.operation.__class__.__name__})')
            outputs = mirror_op(*args, **kwargs)
            for keras_tensor, da_tensor in zip(op.produces, outputs):
                tensor_map[keras_tensor] = da_tensor

    return tuple(tensor_map[keras_tensor] for keras_tensor in model.outputs)


def trace_model(
    model: keras.Model,
    hwconf: HWConfig = HWConfig(1, -1, -1),
    solver_options: dict[str, Any] | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | None = None,
) -> tuple[tuple[FixedVariableArray, ...], tuple[FixedVariableArray, ...]]:
    if inputs is None:
        inputs = tuple(
            FixedVariableArrayInput(inp.shape[1:], hwconf=hwconf, solver_options=solver_options) for inp in model.inputs
        )
    outputs = _apply_nn(model, inputs, verbose=verbose)
    return inputs, outputs
