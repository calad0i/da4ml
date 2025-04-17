import re
import typing
from collections.abc import Callable

import numpy as np

from .cmvm import compile_kernel
from .codegen import Namer, PyCodegenBackend
from .graph_compile import graph_compile_states

m = re.compile(r'Latency: (\d+)')
T = typing.TypeVar('T')


def fn_from_kernel(
    kernel: np.ndarray,
    signs: list[bool],
    bits: list[int],
    int_bits: list[int],
    symmetrics: list[bool],
    depths: list[int] | None = None,
    n_beams: int = 1,
    dc: int | None = None,
    n_inp_max: int = -1,
    n_out_max: int = -1,
    codegen_backend: PyCodegenBackend = PyCodegenBackend(),
    signed_balanced_reduction: bool = True,
) -> tuple[Callable[[list[T]], list[T]], str]:
    """Compile a CMVM operation, with the constant kernel, into a function with only accumulation/subtraction/shift operations.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel to compile. Must be of shape (n_inp, n_out).
    signs : list[bool]
        If the input is signed. Must be of length n_inp.
    bits : list[int]
        The bitwidth of the inputs. Must be of length n_inp.
    int_bits : list[int]
        The number of integer bits in the inputs (incl. sign bit!). Must be of length n_inp.
    symmetrics : list[bool]
        If the input is symmetricly quantized. Must be of length n_inp.
    depths : list[int]|None, optional
        The depth associated with each input. Must be of length n_inp. Defaults to [0]*n_inp.
    n_beams : int, optional
        Number of beams to use in beam search. Defaults to 1. (Currently disabled!)
    dc : int | None, optional
        Delay constraint. Not (properly) implemented yet. Defaults to None.
    n_inp_max : int, optional
        Number of inputs to process in one block. Defaults to -1 (no limit). Decrease to improve performance, but result will be less optimal.
    n_out_max : int, optional
        Number of outputs to process in one block. Defaults to -1 (no limit). Decrease to improve performance, but result will be less optimal.
    codegen_backend : PyCodegenBackend, optional
        The codegen backend to be used. Defaults to PyCodegenBackend().
    signed_balanced_reduction : bool, optional
        If the reduction tree should isolate the plus and minus terms. Set to False to improve latency. Defaults to True.

    Returns
    -------
    tuple[Callable[[list[T]], list[T]], str]
        fn : Callable[[list[T]], list[T]]
            The compiled python function. It takes a list of inputs and returns a list of outputs with only accumulation/subtraction/powers of 2 operations.
        fn_str : str
            The code of the compiled function, depending on the codegen_backend used.
    """

    assert n_beams == 1, 'n_beams>1 is disabled for now. Change line 159 & 160 in this file to enable it.'
    if depths is None:
        depths = [0] * len(signs)
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
        n_out_max=n_out_max,
    )
    with Namer().tmp_scope():
        inp, out = graph_compile_states(states, signed_balanced_reduction)
        fn, fn_str = codegen_backend(inp, out)
    return fn, fn_str


def cost(fn_str: str):
    n_add = fn_str.count('\n') - 3 - fn_str.count('out[')
    latency = m.findall(fn_str)[-1]
    return n_add, int(latency)
