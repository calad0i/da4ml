from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import keras
import numpy as np
from keras import KerasTensor, Operation

from ...trace import FixedVariableArray, FixedVariableArrayInput, HWConfig, comb_trace
from ...trace.fixed_variable import FixedVariable
from .replica import _registry


@dataclass
class OpObj:
    operation: Operation
    args: list
    kwargs: dict
    produces: tuple[KerasTensor, ...]
    requires: tuple[KerasTensor, ...]


def parse_model(model: keras.Model):
    if isinstance(model, keras.Sequential):
        model = model._functional
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


def _flatten_arr(args: Any) -> FixedVariableArray:
    if isinstance(args, FixedVariableArray):
        return np.ravel(args)  # type: ignore
    if isinstance(args, FixedVariable):
        return FixedVariableArray(np.array([args]))
    if not isinstance(args, Sequence):
        return None  # type: ignore
    args = [_flatten_arr(a) for a in args]
    args = [a for a in args if a is not None]
    return np.concatenate(args)  # type: ignore


def _apply_nn(
    model: keras.Model,
    inputs: FixedVariableArray | Sequence[FixedVariableArray],
    verbose: bool = False,
    dump: bool = False,
) -> tuple[FixedVariableArray, ...] | dict[str, FixedVariableArray]:
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

    _inputs = _flatten_arr(inputs)

    for ops in parse_model(model):
        for op in ops:
            assert all(t in tensor_map for t in op.requires)
            args = replace_tensors(tensor_map, op.args)
            kwargs: dict[str, Any] = replace_tensors(tensor_map, op.kwargs)
            if op.operation.__class__ is keras.layers.InputLayer:
                continue
            mirror_op = _registry[op.operation.__class__](op.operation)
            if verbose:
                print(f'Processing operation {op.operation.name} ({op.operation.__class__.__name__})', end='')
            outputs = mirror_op(*args, **kwargs)
            for keras_tensor, da_tensor in zip(op.produces, outputs):
                tensor_map[keras_tensor] = da_tensor
            if verbose:
                cost = comb_trace(_inputs, _flatten_arr(outputs)).cost
                print(f' cumcost: {cost}')

    if not dump:
        return tuple(tensor_map[keras_tensor] for keras_tensor in model.outputs)
    else:
        return {k.name: v for k, v in tensor_map.items()}


@overload
def trace_model(  # type: ignore
    model: keras.Model,
    hwconf: HWConfig = HWConfig(1, -1, -1),
    solver_options: dict[str, Any] | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    dump: Literal[False] = False,
) -> tuple[FixedVariableArray, FixedVariableArray]: ...


@overload
def trace_model(  # type: ignore
    model: keras.Model,
    hwconf: HWConfig = HWConfig(1, -1, -1),
    solver_options: dict[str, Any] | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    dump: Literal[True] = False,  # type: ignore
) -> dict[str, FixedVariableArray]: ...


def trace_model(  # type: ignore
    model: keras.Model,
    hwconf: HWConfig = HWConfig(1, -1, -1),
    solver_options: dict[str, Any] | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | None = None,
    dump=False,
):
    if inputs is None:
        inputs = tuple(
            FixedVariableArrayInput(inp.shape[1:], hwconf=hwconf, solver_options=solver_options) for inp in model.inputs
        )
    outputs = _apply_nn(model, inputs, verbose=verbose, dump=dump)
    if not dump:
        return _flatten_arr(inputs), _flatten_arr(outputs)
    else:
        return {k: _flatten_arr(v) for k, v in outputs.items()}  # type: ignore
