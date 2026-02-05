from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from typing import Literal, overload

from ..cmvm.api import solver_options_t
from ..trace import FixedVariableArray, HWConfig
from .plugin import DAISTracerPluginBase

__all__ = ['trace_model']

ENTRY_POINT_GROUP = 'dais_tracer.plugins'


def get_available_plugins() -> dict[str, EntryPoint]:
    group_eps = entry_points().select(group=ENTRY_POINT_GROUP)
    plugin_repo = {ep.name: ep for ep in group_eps}
    return plugin_repo


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

    module = type(model).__module__.split('.', 1)[0]

    plugins = get_available_plugins()
    if module not in plugins:
        raise ValueError(f'No plugin found for model type from module: {module}. Available plugins: {list(plugins.keys())}')

    entry = plugins[module]

    if verbose:
        print(f'Loading DAIS tracer plugin from {entry.module}:{entry.attr}.')

    _class: type[DAISTracerPluginBase] = entry.load()
    tracer = _class(model, hwconf, solver_options)
    return tracer.trace(
        verbose=verbose,
        inputs=inputs,
        inputs_kif=inputs_kif,
        dump=dump,
    )
