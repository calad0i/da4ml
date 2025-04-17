import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def _volatile_int_arr_to_csd(x: NDArray) -> NDArray[np.int8]:
    x = x
    N = np.max(np.ceil(np.log2(np.abs(x + 1e-19) * 1.5)))
    N = int(max(N, 1))
    buf = np.zeros((*np.shape(x), N), dtype=np.int8)

    for n in range(N - 1, -1, -1):
        _2pn = 2**n
        thres = _2pn / 1.5
        bit = (x > thres).astype(np.int8)
        bit -= (x < -thres).astype(np.int8)
        x -= _2pn * bit
        buf[..., n] = bit
    return buf


@njit(error_model='numpy')
def to_csd(x: NDArray) -> tuple[list[NDArray[np.int8]], NDArray[np.int8]]:
    low, high = -32, 32
    if np.all(x == 0):
        high = low = 0
    while high - low > 1:
        mid = (high + low) // 2
        xs = x * (2.0**mid)
        if np.all(xs == np.floor(xs)):
            high = mid
        else:
            low = mid
    _x = x * (2.0**high)
    csd = _volatile_int_arr_to_csd(_x)
    shifts = np.arange(csd.shape[-1], dtype=np.int8) - high
    return list(csd), shifts


@njit
def _volatile_int_arr_to_binary(x: NDArray) -> NDArray[np.int8]:
    x = x
    N = np.max(np.ceil(np.log2(np.abs(x) + 1)))
    N = int(max(N, 1))
    buf = np.zeros((*np.shape(x), N), dtype=np.int8)

    for n in range(N - 1, -1, -1):
        _2pn = 2**n
        thres = _2pn
        bit = (x >= thres).astype(np.int8)
        bit -= (x <= -thres).astype(np.int8)
        x -= _2pn * bit
        buf[..., n] = bit
    return buf


@njit(error_model='numpy')
def to_binary(x: NDArray) -> tuple[list[NDArray[np.int8]], NDArray[np.int8]]:
    low, high = -32, 32
    if np.all(x == 0):
        high = low = 0
    while high - low > 1:
        mid = (high + low) // 2
        xs = x * (2.0**mid)
        if np.all(xs == np.floor(xs)):
            high = mid
        else:
            low = mid
    _x = x * (2.0**high)
    csd = _volatile_int_arr_to_binary(_x)
    shifts = np.arange(csd.shape[-1], dtype=np.int8) - high
    return list(csd), shifts
