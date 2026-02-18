from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..cmvm import solver_options_t
from ..trace import FixedVariable, FixedVariableArray, FixedVariableArrayInput, HWConfig


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


class DAISTracerPluginBase:
    """
    Base class for DAIS tracer plugins.

    Methods to be implemented by subclasses:
    - `apply_model`
    - `get_input_shapes`
    """

    def __init__(
        self,
        model: Callable,
        hwconf: HWConfig,
        solver_options: solver_options_t | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.hwconf = hwconf
        self.solver_options = solver_options
        assert not kwargs, f'Unexpected keyword arguments: {kwargs}'

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FixedVariableArray, ...],
    ) -> tuple[dict[str, FixedVariableArray], list[str]]:
        """Apply the model and return all intermediate traces.

        Parameters
        ==========
        model: The model to be traced.
        verbose: Whether to print verbose output.
        inputs: Optional inputs to the model.

        Returns
        =======
        A tuple containing:
        - dict[str, FixedVariableArray]: A dictionary of intermediate names -> FixedVariableArray
        - list[str]: A list of output names.
        """
        ...

    def get_input_shapes(
        self,
    ) -> Sequence[tuple[int, ...]] | None:
        """Get the input shapes for the model. Only used if get_input_kifs returns None.

        Returns
        =======
        A list of input shapes, or None if not applicable. If cannot be determined, return None.
        """
        ...

    def _get_inputs(
        self, inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None, inputs_kif: tuple[int, int, int] | None
    ) -> tuple[FixedVariableArray, ...]:
        if inputs is not None:
            return inputs if isinstance(inputs, tuple) else (inputs,)

        shapes = self.get_input_shapes()

        assert shapes is not None, 'Inputs must be provided: cannot determine input shapes automatically.'

        if inputs_kif is None:
            return tuple(FixedVariableArrayInput(shape, self.hwconf, self.solver_options) for shape in shapes)

        kif = tuple(tuple(np.full(shape, v, dtype=np.int8) for v in inputs_kif) for shape in shapes)
        return tuple(FixedVariableArray.from_kif(k, i, f, self.hwconf, 0, self.solver_options) for k, i, f in kif)

    def trace(
        self,
        verbose: bool = False,
        inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
        inputs_kif: tuple[int, int, int] | None = None,
        dump: bool = False,
    ) -> dict[str, FixedVariableArray] | tuple[FixedVariableArray, FixedVariableArray]:
        """Trace the model.

        Parameters
        ==========
        verbose: Whether to print verbose output.
        inputs: Optional inputs to the model.
        inputs_kif: Optional input kif values, only used if inputs is None.
        dump: Whether to dump all intermediate traces.

        Returns
        =======
        If dump is True, returns a dictionary of all intermediate names -> FixedVariableArray.
        If dump is False, returns a list of output FixedVariableArray.
        """

        inputs = self._get_inputs(inputs, inputs_kif)

        all_traces, output_names = self.apply_model(
            verbose=verbose,
            inputs=inputs,
        )

        if dump:
            return all_traces

        outputs = _flatten_arr([all_traces[name] for name in output_names])
        inputs = _flatten_arr(inputs)
        return inputs, outputs
