from inspect import signature
from typing import Any

import numpy as np
from numba.typed import List as NumbaList
from numpy.typing import NDArray

from ..cmvm import solve
from .fixed_variable import FixedVariable, HWConfig, QInterval


class FixedVariableArray:
    def __init__(
        self,
        vars: NDArray,
        solver_options: dict[str, Any] | None = None,
    ):
        self._vars = np.array(vars)
        _solver_options = signature(solve).parameters
        _solver_options = {k: v.default for k, v in _solver_options.items() if v.default is not v.empty}
        if solver_options is not None:
            _solver_options.update(solver_options)
        _solver_options.pop('qintervals', None)
        _solver_options.pop('latencies', None)
        self.solver_options = _solver_options

    @classmethod
    def from_lhs(
        cls,
        low: NDArray[np.floating],
        high: NDArray[np.floating],
        step: NDArray[np.floating],
        hwconf: HWConfig,
        latency: np.ndarray | float = 0.0,
        solver_options: dict[str, Any] | None = None,
    ):
        shape = low.shape
        assert shape == high.shape == step.shape

        low, high, step = low.ravel(), high.ravel(), step.ravel()
        latency = np.full_like(low, latency) if isinstance(latency, (int, float)) else latency.ravel()

        vars = []
        for i, (l, h, s, lat) in enumerate(zip(low, high, step, latency)):
            var = FixedVariable(
                low=float(l),
                high=float(h),
                step=float(s),
                hwconf=hwconf,
                latency=float(
                    lat,
                ),
            )
            vars.append(var)
        vars = np.array(vars).reshape(shape)
        return cls(vars, solver_options)

    __array_priority__ = 100

    @classmethod
    def from_kif(
        cls,
        k: NDArray[np.bool_ | np.integer],
        i: NDArray[np.integer],
        f: NDArray[np.integer],
        hwconf: HWConfig,
        latency: NDArray[np.floating] | float = 0.0,
        solver_options: dict[str, Any] | None = None,
    ):
        step = 2.0**-f
        _high = 2.0**i
        high, low = _high - step, -_high * k
        return cls.from_lhs(low, high, step, hwconf, latency, solver_options)

    def __matmul__(self, other):
        assert isinstance(other, np.ndarray)
        kwargs = (self.solver_options or {}).copy()
        shape0, shape1 = self.shape, other.shape
        assert shape0[-1] == shape1[0], f'Matrix shapes do not match: {shape0} @ {shape1}'
        c = shape1[0]
        out_shape = shape0[:-1] + shape1[1:]
        mat0, mat1 = self.reshape((-1, c)), other.reshape((c, -1))
        r = []
        for i in range(mat0.shape[0]):
            vec = mat0[i]
            _qintervals = [QInterval(float(v.low), float(v.high), float(v.step)) for v in vec._vars]
            _latencies = [float(v.latency) for v in vec._vars]
            qintervals = NumbaList(_qintervals)  # type: ignore
            latencies = NumbaList(_latencies)  # type: ignore
            hwconf = self._vars.ravel()[0].hwconf
            kwargs.update(adder_size=hwconf.adder_size, carry_size=hwconf.carry_size)
            _mat = np.ascontiguousarray(mat1.astype(np.float32))
            sol = solve(_mat, qintervals=qintervals, latencies=latencies, **kwargs)
            _r = sol(vec._vars)
            r.append(_r)
        r = np.array(r).reshape(out_shape)
        return FixedVariableArray(r, self.solver_options)

    def __rmatmul__(self, other):
        mat1 = np.moveaxis(other, -1, 0)
        mat0 = np.moveaxis(self._vars, 0, -1)
        ndim0, ndim1 = mat0.ndim, mat1.ndim
        r = FixedVariableArray(mat0, self.solver_options) @ mat1

        _axes = tuple(range(0, ndim0 + ndim1 - 2))
        axes = _axes[ndim0 - 1 :] + _axes[: ndim0 - 1]
        return r.transpose(axes)

    def __getitem__(self, item):
        vars = self._vars[item]
        if isinstance(vars, np.ndarray):
            return FixedVariableArray(vars, self.solver_options)
        else:
            return vars

    def __len__(self):
        return len(self._vars)

    @property
    def shape(self):
        return self._vars.shape

    def __add__(self, other):
        return FixedVariableArray(self._vars + other, self.solver_options)

    def __sub__(self, other):
        return FixedVariableArray(self._vars - other, self.solver_options)

    def __mul__(self, other):
        return FixedVariableArray(self._vars * other, self.solver_options)

    def __truediv__(self, other):
        return FixedVariableArray(self._vars * (1 / other), self.solver_options)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return FixedVariableArray(-self._vars, self.solver_options)

    def __repr__(self):
        shape = self._vars.shape
        hwconf_str = str(self._vars.ravel()[0].hwconf)[8:]
        max_lat = max(v.latency for v in self._vars.ravel())
        return f'FixedVariableArray(shape={shape}, hwconf={hwconf_str}, latency={max_lat})'

    def relu(self, i: NDArray[np.integer] | None = None, f: NDArray[np.integer] | None = None, round_mode: str = 'TRN'):
        shape = self._vars.shape
        i = np.broadcast_to(i, shape) if i is not None else np.full(shape, None)
        f = np.broadcast_to(f, shape) if f is not None else np.full(shape, None)
        ret = []
        for v, i, f in zip(self._vars.ravel(), i.ravel(), f.ravel()):
            ret.append(v.relu(i=i, f=f, round_mode=round_mode))
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options)

    def quantize(
        self,
        k: NDArray[np.integer] | None = None,
        i: NDArray[np.integer] | None = None,
        f: NDArray[np.integer] | None = None,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        shape = self._vars.shape
        k = np.broadcast_to(k, shape) if k is not None else np.full(shape, None)
        i = np.broadcast_to(i, shape) if i is not None else np.full(shape, None)
        f = np.broadcast_to(f, shape) if f is not None else np.full(shape, None)
        ret = []
        for v, k, i, f in zip(self._vars.ravel(), k.ravel(), i.ravel(), f.ravel()):
            ret.append(v.quantize(k=k, i=i, f=f, overflow_mode=overflow_mode, round_mode=round_mode))
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options)

    def flatten(self):
        return FixedVariableArray(self._vars.flatten(), self.solver_options)

    def reshape(self, shape):
        return FixedVariableArray(self._vars.reshape(shape), self.solver_options)

    def transpose(self, axes=None):
        return FixedVariableArray(self._vars.transpose(axes), self.solver_options)

    def ravel(self):
        return FixedVariableArray(self._vars.ravel(), self.solver_options)

    @property
    def dtype(self):
        return self._vars.dtype
