from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal, overload

from ..cmvm import solver_options_t
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
    model: Any,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump: Literal[False] = False,
    framework: None | str = None,
    **kwargs: Any,
) -> tuple[FixedVariableArray, FixedVariableArray]: ...


@overload
def trace_model(  # type: ignore
    model: Any,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump: Literal[True] = False,  # type: ignore
    framework: None | str = None,
    **kwargs: Any,
) -> dict[str, FixedVariableArray]: ...


def trace_model(  # type: ignore
    model: Any,
    hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
    solver_options: solver_options_t | None = None,
    verbose: bool = False,
    inputs: tuple[FixedVariableArray, ...] | None = None,
    inputs_kif: tuple[int, int, int] | None = None,
    dump=False,
    framework: None | str = None,
    **kwargs: Any,
):
    hwconf = HWConfig(*hwconf) if isinstance(hwconf, tuple) else hwconf

    framework = framework or type(model).__module__.split('.', 1)[0]

    plugins = get_available_plugins()
    if framework not in plugins:
        raise ValueError(f'No plugin found for model type from module: {framework}. Available plugins: {list(plugins.keys())}')

    entry = plugins[framework]

    if verbose:
        print(f'Loading DAIS tracer plugin from {entry.module}:{entry.attr}.')

    _class: type[DAISTracerPluginBase] = entry.load()
    tracer = _class(model, hwconf, solver_options, **kwargs)
    return tracer.trace(
        verbose=verbose,
        inputs=inputs,
        inputs_kif=inputs_kif,
        dump=dump,
    )
