from inspect import signature
from typing import Any, TypeVar

import numpy as np
from numba.typed import List as NumbaList
from numpy.typing import NDArray

from ..cmvm import solve
from .fixed_variable import FixedVariable, FixedVariableInput, HWConfig, QInterval
from .ops import einsum, reduce

T = TypeVar('T')


def to_raw_arr(obj: T) -> T:
    if isinstance(obj, tuple):
        return tuple(to_raw_arr(x) for x in obj)  # type: ignore
    elif isinstance(obj, list):
        return [to_raw_arr(x) for x in obj]  # type: ignore
    elif isinstance(obj, dict):
        return {k: to_raw_arr(v) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, FixedVariableArray):
        return obj._vars  # type: ignore
    return obj


def _max_of(a, b):
    if isinstance(a, FixedVariable):
        return a.max_of(b)
    elif isinstance(b, FixedVariable):
        return b.max_of(a)
    else:
        return max(a, b)


def _min_of(a, b):
    if isinstance(a, FixedVariable):
        return a.min_of(b)
    elif isinstance(b, FixedVariable):
        return b.min_of(a)
    else:
        return min(a, b)


class FixedVariableArray:
    __array_priority__ = 100

    def __array_function__(self, func, types, args, kwargs):
        if func is np.matmul:
            if len(args) == 1 and isinstance(args[0], np.ndarray):
                return self.__matmul__(args[0])
            elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                return self.__rmatmul__(args[1])

        if func in (np.mean, np.sum, np.amax, np.amin, np.max, np.min):
            match func:
                case np.mean:
                    _x = reduce(lambda x, y: x + y, self, *args[1:], **kwargs)
                    return _x * (_x.size / self._vars.size)
                case np.sum:
                    return reduce(lambda x, y: x + y, self, *args[1:], **kwargs)
                case np.max | np.amax:
                    return reduce(_max_of, self, *args[1:], **kwargs)
                case np.min | np.amin:
                    return reduce(_min_of, self, *args[1:], **kwargs)
                case _:
                    raise NotImplementedError(f'Unsupported function: {func}')

        if func is np.clip:
            assert len(args) == 3, 'Clip function requires exactly three arguments'
            x, low, high = args
            _x, low, high = np.broadcast_arrays(x, low, high)
            x = FixedVariableArray(_x, self.solver_options)
            x = np.amax(np.stack((x, low), axis=-1), axis=-1)  # type: ignore
            return np.amin(np.stack((x, high), axis=-1), axis=-1)

        if func is np.einsum:
            # assert len(args) == 2
            sig = signature(np.einsum)
            bind = sig.bind(*args, **kwargs)
            eq = args[0]
            operands = bind.arguments['operands']
            if isinstance(operands[0], str):
                operands = operands[1:]
            assert len(operands) == 2, 'Einsum on FixedVariableArray requires exactly two operands'
            assert bind.arguments.get('out', None) is None, 'Output argument is not supported'
            return einsum(eq, *operands)

        if func in (np.dot, np.matmul):
            assert len(args) in (2, 3), 'Dot function requires exactly two or three arguments'

            assert len(args) == 2
            a, b = args
            if not isinstance(a, FixedVariableArray):
                a = np.array(a)
            if not isinstance(b, FixedVariableArray):
                b = np.array(b)
            if a.shape[-1] == b.shape[0]:
                return a @ b

            assert a.size == 1 or b.size == 1, f'Error in dot product: {a.shape} @ {b.shape}'
            return a * b

        args, kwargs = to_raw_arr(args), to_raw_arr(kwargs)
        return FixedVariableArray(
            func(*args, **kwargs),
            self.solver_options,
        )

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
        mask = k + i + f <= 0
        k = np.where(mask, 0, k)
        i = np.where(mask, 0, i)
        f = np.where(mask, 0, f)
        step = 2.0**-f
        _high = 2.0**i
        high, low = _high - step, -_high * k
        return cls.from_lhs(low, high, step, hwconf, latency, solver_options)

    def __matmul__(self, other):
        if isinstance(other, FixedVariableArray):
            other = other._vars
        if not isinstance(other, np.ndarray):
            other = np.array(other)
        if any(isinstance(x, FixedVariable) for x in other.ravel()):
            mat0, mat1 = self._vars, other
            shape = mat0.shape[:-1] + mat1.shape[1:]
            mat0, mat1 = mat0.reshape((-1, mat0.shape[-1])), mat1.reshape((mat1.shape[0], -1))
            _shape = (mat0.shape[0], mat1.shape[1])
            _vars = np.empty(_shape, dtype=object)
            for i in range(mat0.shape[0]):
                for j in range(mat1.shape[1]):
                    vec0 = mat0[i]
                    vec1 = mat1[:, j]
                    _vars[i, j] = reduce(lambda x, y: x + y, vec0 * vec1)
            return FixedVariableArray(_vars.reshape(shape), self.solver_options)

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
        mat0 = np.moveaxis(self, 0, -1)  # type: ignore
        ndim0, ndim1 = mat0.ndim, mat1.ndim
        r = mat0 @ mat1

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
        if isinstance(other, FixedVariableArray):
            return FixedVariableArray(self._vars + other._vars, self.solver_options)
        return FixedVariableArray(self._vars + other, self.solver_options)

    def __sub__(self, other):
        if isinstance(other, FixedVariableArray):
            return FixedVariableArray(self._vars - other._vars, self.solver_options)
        return FixedVariableArray(self._vars - other, self.solver_options)

    def __mul__(self, other):
        if isinstance(other, FixedVariableArray):
            return FixedVariableArray(self._vars * other._vars, self.solver_options)
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

    def __pow__(self, power: int | float):
        _power = int(power)
        assert _power == power, 'Power must be an integer'
        return FixedVariableArray(self._vars**_power, self.solver_options)

    def relu(self, i: NDArray[np.integer] | None = None, f: NDArray[np.integer] | None = None, round_mode: str = 'TRN'):
        shape = self._vars.shape
        i = np.broadcast_to(i, shape) if i is not None else np.full(shape, None)
        f = np.broadcast_to(f, shape) if f is not None else np.full(shape, None)
        ret = []
        for v, i, f in zip(self._vars.ravel(), i.ravel(), f.ravel()):  # type: ignore
            ret.append(v.relu(i=i, f=f, round_mode=round_mode))
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options)

    def quantize(
        self,
        k: NDArray[np.integer] | np.integer | int | None = None,
        i: NDArray[np.integer] | np.integer | int | None = None,
        f: NDArray[np.integer] | np.integer | int | None = None,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        shape = self._vars.shape
        k = np.broadcast_to(k, shape) if k is not None else np.full(shape, None)
        i = np.broadcast_to(i, shape) if i is not None else np.full(shape, None)
        f = np.broadcast_to(f, shape) if f is not None else np.full(shape, None)
        ret = []
        for v, k, i, f in zip(self._vars.ravel(), k.ravel(), i.ravel(), f.ravel()):  # type: ignore
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

    @property
    def size(self):
        return self._vars.size

    @property
    def ndim(self):
        return self._vars.ndim

    @property
    def kif(self):
        shape = self._vars.shape
        kif = np.array([v.kif for v in self._vars.ravel()]).reshape(*shape, 3)
        return np.moveaxis(kif, -1, 0)


class FixedVariableArrayInput(FixedVariableArray):
    def __init__(
        self,
        shape: tuple[int, ...] | int,
        hwconf: HWConfig = HWConfig(1, -1, -1),
        solver_options: dict[str, Any] | None = None,
        latency=0.0,
    ):
        _vars = np.empty(shape, dtype=object)
        _vars_f = _vars.ravel()
        for i in range(_vars.size):
            _vars_f[i] = FixedVariableInput(latency, hwconf)
        super().__init__(_vars, solver_options)
