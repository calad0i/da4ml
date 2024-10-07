import re
from collections.abc import Callable
import numpy as np

from .codegen import Namer
from .utils import DAState
from .graph_compile import graph_compile_states
from .cmvm import compile_kernel
from .codegen import PyCodegenBackend

m = re.compile(r'Latency: (\d+)')


def fn_from_kernel(
    kernel: np.ndarray,
    signs: list[bool],
    bits: list[int],
    int_bits: list[int],
    symmetrics: list[bool],
    depths: list[int],
    n_beams: int = 1,
    dc: int | None = None,
    n_inp_max: int = -1,
    n_out_max: int = -1,
    codegen_backend: PyCodegenBackend = PyCodegenBackend()
) -> tuple[Callable[[list[float]], list[float]], str]:
    states = compile_kernel(
        kernel=kernel,
        signs=signs,
        bits=bits,
        int_bits=int_bits,
        symmetrics=symmetrics,
        depths=depths,
        n_beams=n_beams,
        dc=dc,
        n_inp_max=n_inp_max,
        n_out_max=n_out_max
    )
    with Namer().tmp_scope():
        inp, out = graph_compile_states(states)
        fn, fn_str = codegen_backend(inp, out)
    return fn, fn_str


def cost(fn_str: str):
    n_add = fn_str.count('\n') - 3 - fn_str.count('out[')
    latency = m.findall(fn_str)[-1]
    return n_add, int(latency)
