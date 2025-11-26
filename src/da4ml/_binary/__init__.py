import numpy as np
from numpy.typing import NDArray

from .dais_bin import run_interp


def dais_interp_run(bin_logic: NDArray[np.int32], data: NDArray, n_threads: int = 1):
    inp_size = int(bin_logic[2])

    assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'

    inputs = np.ascontiguousarray(np.ravel(data), dtype=np.float64)
    bin_logic = np.ascontiguousarray(np.ravel(bin_logic), dtype=np.int32)

    return run_interp(bin_logic, inputs, n_threads)
