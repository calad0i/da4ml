from collections.abc import Sequence
from itertools import chain
from logging import warning
from typing import Any

import hgq
import keras
import numpy as np
from hgq.quantizer import Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase
from keras import KerasTensor
from numpy.typing import NDArray

from ...trace import FixedVariableArray, HWConfig
from .mirror import _registry

dependency_t = tuple[tuple[str, ...], tuple[str, ...], list[tuple[str, tuple[str, ...], tuple[str, ...]]], dict[str, KerasTensor]]


def get_io_tensors(layer: keras.Layer, node_whitelist: set[int] | None = None):
    """
    Given a keras layer, return a list of tuples of input and output tensors.
    If the layer is called only once (i.e., layer is not used multiple times in the same model),
    the list will contain only one tuple.

    The layer must have been built before calling this function.

    Parameters
    ----------
    layer : keras.Layer
        The layer to get input and output tensors from.
    node_whitelist : set of int, optional
        If not None, only return tensors from nodes with ids in this set, used to filter out nodes
        that are not part of the model. Defaults to None.

    Returns
    -------
    list of tuple of tuple of KerasTensor
        A list of tuples of input and output tensors. Each inner tuple contains two tuples:
        the first with input KerasTensors and the second with output KerasTensors.
    """

    in_nodes = layer._inbound_nodes
    if node_whitelist is not None:
        in_nodes = [node for node in in_nodes if id(node) in node_whitelist]

    ret: list[tuple[tuple['KerasTensor', ...], tuple['KerasTensor', ...]]] = []
    for node in in_nodes:
        in_tensors = tuple(node.arguments.keras_tensors)
        out_tensors = tuple(node.outputs)
        ret.append((in_tensors, out_tensors))
    return ret


def resolve_dependency_relation(model: keras.Model):
    """
    Given a keras model, return the following information:

    Parameters
    ----------
    model : keras.Model
        The keras model to analyze.

    Returns
    -------
    tuple
        inp_tensor_names : tuple of str
            A tuple of input tensor names.
        out_tensor_names : tuple of str
            A tuple of output tensor names.
        layer_io : list of tuple (str, tuple of str, tuple of str)
            A list of tuples, where each tuple contains the layer name, a tuple of its input tensor names,
            and a tuple of its output tensor names.
        tensors : dict of str to KerasTensor
            A dictionary mapping tensor names to KerasTensor objects.
    """

    tensors: dict[str, KerasTensor] = {}
    'tensor_name -> KerasTensor'
    depends_on: dict[str, tuple[str, ...]] = {}
    'tensor_name -> {tensor_name}'
    layer_io: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    'layer_name -> ((input_tensor_names), (output_tensor_names))'

    inputs: tuple[str, ...] = tuple(t.name for t in model.inputs)
    outputs: tuple[str, ...] = tuple(t.name for t in model.outputs)
    node_whitelist = {id(node) for v in model._nodes_by_depth.values() for node in v}

    for layer in model.layers:
        for in_tensors, out_tensors in get_io_tensors(layer, node_whitelist):
            in_tensor_names = tuple(t.name for t in in_tensors)
            out_tensor_names = tuple(t.name for t in out_tensors)
            for t in chain(in_tensors, out_tensors):
                tensors[t.name] = t
            for o_name in out_tensor_names:
                depends_on[o_name] = in_tensor_names
            layer_io.append((layer.name, in_tensor_names, out_tensor_names))

    return inputs, outputs, layer_io, tensors


def _apply_nn(
    model: keras.Model, dependency: dependency_t, inputs: FixedVariableArray | Sequence[FixedVariableArray], verbose: bool = False
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

    _inputs, _outputs, _layer_io, _ = dependency
    assert len(_inputs) == len(inputs), f'Expected {len(_inputs)} inputs, got {len(inputs)}'

    satisfied = dict(zip(_inputs, inputs))

    while any(n not in satisfied for n in _outputs):
        for layer_name, in_tensor_names, out_tensor_names in _layer_io:
            if not in_tensor_names:
                # Input layer
                continue
            if not all(n in satisfied for n in in_tensor_names):
                continue
            for out_tensor_name in out_tensor_names:
                assert (
                    out_tensor_name not in satisfied
                ), f'Output tensor {out_tensor_name} by layer {layer_name} already satisfied'

            layer: keras.Layer = model.get_layer(layer_name)
            if verbose:
                print(f'Processing layer {layer_name} ({layer.__class__.__name__})')
            mirror_layer = _registry[layer.__class__](layer)
            inp_tensors = tuple(satisfied[n] for n in in_tensor_names)
            outputs = mirror_layer(inp_tensors)
            for out_name, out_tensor in zip(out_tensor_names, outputs):
                satisfied[out_name] = out_tensor

    return tuple(satisfied[n] for n in _outputs)


def get_inp_kif(model, dependency: dependency_t, _input: str, reshape_to: tuple[int, ...] | None = None) -> NDArray[np.int8]:
    kifs = []
    for layer_name, in_tensor_names, out_tensor_names in dependency[2]:
        if _input in in_tensor_names:
            layer = model.get_layer(layer_name)
            if isinstance(layer, hgq.layers.QLayerBase) and layer.enable_iq:
                if len(in_tensor_names) == 1:
                    q: Quantizer = layer.iq
                else:
                    idx = in_tensor_names.index(_input)
                    q: Quantizer = layer.iq.quantizers[idx]
            elif isinstance(layer, Quantizer):
                q = layer
            else:
                if isinstance(layer, (keras.layers.Flatten, keras.layers.Reshape)):
                    reshape_to = reshape_to or dependency[3][_input].shape[1:]  # type: ignore
                    return get_inp_kif(model, dependency, out_tensor_names[0], reshape_to=reshape_to)
                raise ValueError(
                    f'Input {_input} is followed by layer {layer_name} which is neighter a quantizing layer or a idendity layer'
                )

            assert isinstance(q.quantizer, FixedPointQuantizerBase), 'Only fixed point quantizers are supported'
            kif = np.empty((3, 1) + dependency[-1][_input].shape[1:], dtype=np.int8)
            kif[:] = np.array(q.quantizer.kif, dtype=np.int8)
            if q.quantizer.overflow_mode != 'WRAP':
                warning(
                    f'Input {_input} is quantized with overflow mode {q.quantizer.overflow_mode}. However, WRAP overflow map happen at the inputs due to unknown external input wire width'
                )
            if q.quantizer.round_mode == 'RND':
                kif[2] += np.sum(kif, axis=0) > 0
            elif q.quantizer.round_mode != 'TRN':
                warning(
                    f'Input {_input} is quantized with round mode {q.quantizer.round_mode}. However, TRN round mode may take place at the inputs due to unknown external input wire width'
                )
            kifs.append(kif)
    if not kifs:
        raise ValueError(f'Input {_input} not found in dependency relation')

    kif = np.max(kifs, axis=0)
    return kif[:, 0]


def get_inputs(
    model: keras.Model, dependency: dependency_t, hwconf: HWConfig, solver_options: dict[str, Any] | None
) -> tuple[FixedVariableArray, ...]:
    # input_shapes: tuple[tuple[int, ...],...] = tuple(_keras_tensors[n].shape[1:] for n in _inputs) # type: ignore
    kifs = tuple(get_inp_kif(model, dependency, _input) for _input in dependency[0])
    for kif in kifs:
        mask = np.sum(kif, axis=0) <= 0
        kif[:, mask] = 0
    arrs = tuple(FixedVariableArray.from_kif(*kif, hwconf=hwconf, solver_options=solver_options) for kif in kifs)
    return arrs


def trace_model(
    model: keras.Model, hwconf: HWConfig = HWConfig(1, -1, -1), solver_options: dict[str, Any] | None = None, verbose=False
) -> tuple[tuple[FixedVariableArray, ...], tuple[FixedVariableArray, ...]]:
    assert isinstance(model, keras.Model), 'model must be a keras.Model instance'
    if isinstance(model, keras.Sequential):
        model = model._functional
    dependency = resolve_dependency_relation(model)
    inputs = get_inputs(model, dependency, hwconf, solver_options)
    outputs = _apply_nn(model, dependency, inputs, verbose=verbose)
    return inputs, outputs
