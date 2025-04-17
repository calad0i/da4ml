import numpy as np
from numba import jit
from numpy.typing import NDArray


@jit
def _volatile_int_arr_to_csd(x: NDArray) -> NDArray[np.int8]:
    x = x
    N = np.max(np.ceil(np.log2(np.abs(x) * 1.5 + 1e-19)))
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


@jit(error_model='numpy')
def _shift_centering(arr: NDArray):
    low, high = -64, 64
    if np.all(arr == 0):
        high = low = 0
    while high - low > 1:
        mid = (high + low) // 2
        xs = arr * (2.0**mid)
        if np.all(xs == np.floor(xs)):
            high = mid
        else:
            low = mid
    return -high


@jit(error_model='numpy')
def shift_centering(arr: NDArray, axis: int):
    n = arr.shape[axis]
    shifts = np.empty(n, dtype=np.int8)
    for i in range(n):
        shifts[i] = _shift_centering(arr.take(i, axis=axis))
    return shifts


@jit
def _center(arr: NDArray):
    shift1 = shift_centering(arr, 1)  # d_out
    arr = arr * (2.0**-shift1)
    shift0 = shift_centering(arr, 0)  # d_in
    arr = arr * (2.0 ** -shift0[:, None])
    return arr, shift0, shift1


@jit
def csd_decompose(arr: NDArray, center=True):
    """
    Convert an 2D array to CSD representation.

    Parameters
    ----------
    arr : ndarray
        Input array to be converted.
    center : bool, optional
        If True, the array is centered before conversion. Default is True.
        If False, the function may accept non-2D arrays.

    Returns
    -------
    csd : ndarray
        CSD representation of the input array after centering, if center is True.
    shift0 : ndarray
        Shift values for the first axis.
    shift1 : ndarray
        Shift values for the second axis.
    """

    if center:
        arr, shift0, shift1 = _center(arr)
    else:
        shift0 = np.zeros(arr.shape[0], dtype=np.int8)
        shift1 = np.zeros(arr.shape[1], dtype=np.int8)
        arr = arr.copy()
    csd = _volatile_int_arr_to_csd(arr)
    return csd, shift0, shift1
