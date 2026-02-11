import numpy
from numpy.typing import NDArray

def run_interp(bin_logic: NDArray[numpy.int32], data: NDArray[numpy.float64], n_threads: int = 1) -> NDArray[numpy.float64]: ...
