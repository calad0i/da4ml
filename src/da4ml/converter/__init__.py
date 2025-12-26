from collections.abc import Callable
from typing import Literal, overload

from ..cmvm.api import solver_options_t
from ..trace import FixedVariableArray, HWConfig

__all__ = ['trace_model']


@overload
def trace_model(  # type: ignore
    model: Callable,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump: Literal[False] = False,
) -> tuple[FixedVariableArray, FixedVariableArray]: ...


@overload
def trace_model(  # type: ignore
    model: Callable,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump: Literal[True] = False,  # type: ignore
) -> dict[str, FixedVariableArray]: ...


def trace_model(  # type: ignore
    model: Callable,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump=False,
):
    hwconf = HWConfig(*hwconf) if isinstance(hwconf, tuple) else hwconf

    module = type(model).__module__
    if module.startswith('keras.'):
        import keras

        from .hgq2 import trace_model as keras_trace_model

        assert isinstance(model, keras.Model)

        return keras_trace_model(
            model,
            hwconf,
            solver_options=solver_options,
            verbose=verbose,
            inputs=inputs,
            inputs_kif=inputs_kif,
            dump=dump,
        )
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')
