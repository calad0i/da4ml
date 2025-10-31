import ctypes
import os
import typing
from ctypes import _Pointer, c_double, c_int32
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_binary = Path(__file__).parent


if typing.TYPE_CHECKING:

    class DAISLib:
        def openmp_enabled(self) -> bool: ...

        def run_interp(
            self,
            bin_data: _Pointer[c_int32],
            bin_data_len: int,
            inputs: _Pointer[c_double],
            outputs: _Pointer[c_double],
            n_samples: int,
        ) -> None: ...

        def run_interp_openmp(
            self,
            bin_data: _Pointer[c_int32],
            bin_data_len: int,
            inputs: _Pointer[c_double],
            outputs: _Pointer[c_double],
            n_samples: int,
            n_threads: int,
        ) -> None: ...

    dais_lib: DAISLib | None


match os.uname().sysname:
    case 'Darwin':
        ext = 'dylib'
    case 'Linux':
        ext = 'so'
    case 'Windows':
        ext = 'dll'
    case _:
        ext = 'so'

if (_binary / f'libdais.{ext}').exists():
    dais_lib = ctypes.CDLL(str(_binary / f'libdais.{ext}'))  # type: ignore
else:
    dais_lib = None


msg = f"""DAIS interpreter shared library is not available.
You can compile it with the following commands:
$(CXX) -O3 -std=c++17 -fPIC -fopenmp -shared {_binary}/cpp/DAISInterpreter.cc -o {_binary}/libdais.{ext}
"""


def dais_interp_run(bin_logic: NDArray[np.int32], data: NDArray, n_threads: int = 1):
    if dais_lib is None:
        raise ImportError(msg)

    inp_size, out_size = map(int, bin_logic[2:4])

    assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'
    n_sample = data.size // inp_size

    bin_logic_buf = bin_logic.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    inp_data = data.astype(np.float64).ravel()
    out_data: NDArray[np.float64] = np.empty(n_sample * out_size, dtype=np.float64)

    inp_buf = inp_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_buf = out_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if dais_lib.openmp_enabled() and n_threads != 1:
        dais_lib.run_interp_openmp(bin_logic_buf, len(bin_logic), inp_buf, out_buf, n_sample, n_threads)
    else:
        dais_lib.run_interp(bin_logic_buf, len(bin_logic), inp_buf, out_buf, n_sample)

    return out_data.reshape(-1, out_size)
