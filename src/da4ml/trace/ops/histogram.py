"""Histogram operation for FixedVariable arrays using thermometer code counting."""

import typing
from typing import Literal

import numpy as np
from numpy.typing import NDArray

if typing.TYPE_CHECKING:
    from ..fixed_variable_array import FixedVariableArray

_range = range


def histogram(
    a: 'FixedVariableArray | NDArray',
    bins: 'int | NDArray' = 10,
    range: 'tuple[float, float] | None' = None,
    weights: 'FixedVariableArray | NDArray | None' = None,
    density: Literal[False] = False,
) -> 'tuple[FixedVariableArray | NDArray, NDArray]':
    """Compute histogram using thermometer code counting.

    Bin edges must be compile-time constants. Only internal edges are compared;
    out-of-range elements naturally land in the first/last bins.

    Parameters
    ----------
    a : FixedVariableArray or ndarray
        Input data. Flattened before processing.
    bins : int or 1-D array-like
        If int: number of equal-width bins (``range`` required).
        If array: monotonically increasing bin edges.
    range : (float, float) or None
        Required when ``bins`` is an int.
    weights : FixedVariableArray or ndarray or None
        Per-element weights (same shape as ``a``). When given, each element
        contributes its weight instead of 1 to the bin count.
    density : bool
        Not supported, raises ValueError if True.

    Returns
    -------
    counts : FixedVariableArray or ndarray
    bin_edges : ndarray
    """
    from ..fixed_variable_array import FixedVariableArray

    assert not density, 'density=True is not supported'

    if not isinstance(a, FixedVariableArray):
        return np.histogram(a, bins=bins, range=range, weights=weights)

    if isinstance(bins, (int, np.integer)):
        if range is None:
            raise ValueError('range=(lo, hi) required when bins is an int')
        lo, hi = float(range[0]), float(range[1])
        if lo >= hi:
            raise ValueError(f'range must satisfy lo < hi, got ({lo}, {hi})')
        edges = np.linspace(lo, hi, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError('bins must be 1-D with at least 2 edges')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('bin edges must be strictly increasing')

    flat = a.ravel()
    M = len(flat)

    if weights is not None:
        _weights = np.asarray(weights).ravel()
        if _weights.size != M:
            raise ValueError(f'weights must have same length as a, got {len(weights)} vs {M}')
    else:
        _weights = 1

    if M == 0:
        return FixedVariableArray(np.zeros(len(edges) - 1), a.solver_options, hwconf=a.hwconf), edges

    # Thermometer encoding

    _cum1 = np.sum((flat[None, :] >= edges[:-1, None]) * _weights, axis=-1)  # type: ignore
    _cum2 = np.sum((flat[None, :] > edges[-1:, None]) * _weights, axis=-1)  # type: ignore
    cum = np.concatenate([_cum1, _cum2], axis=0)

    counts = cum[:-1] - cum[1:]

    return counts, edges  # type: ignore
