import numpy
from numpy.typing import NDArray

def get_lsb_loc(x: float) -> int: ...
def csd_decompose(inp: NDArray[numpy.float32], center: bool = True) -> tuple: ...
def kernel_decompose(kernel: NDArray[numpy.float32], dc: int = -2) -> tuple: ...
def solve(
    kernel: NDArray[numpy.float32],
    method0: str = 'wmc',
    method1: str = 'auto',
    hard_dc: int = -1,
    decompose_dc: int = -2,
    qintervals: object | None = None,
    latencies: object | None = None,
    adder_size: int = -1,
    carry_size: int = -1,
    search_all_decompose_dc: bool = True,
) -> object: ...
def cost_add(
    q0: tuple[float, float, float], q1: tuple[float, float, float], shift: int, sub: bool, adder_size: int, carry_size: int
) -> tuple[float, float]: ...
