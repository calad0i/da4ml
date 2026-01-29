from collections.abc import Callable
from inspect import signature
from typing import TypeVar

import numpy as np
from numba.typed import List as NumbaList
from numpy.typing import NDArray

from ..cmvm.api import solve, solver_options_t
from .fixed_variable import FixedVariable, FixedVariableInput, HWConfig, LookupTable, QInterval
from .ops import _quantize, einsum, reduce

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


def mmm(mat0: np.ndarray, mat1: np.ndarray):
    shape = mat0.shape[:-1] + mat1.shape[1:]
    mat0, mat1 = mat0.reshape((-1, mat0.shape[-1])), mat1.reshape((mat1.shape[0], -1))
    _shape = (mat0.shape[0], mat1.shape[1])
    _vars = np.empty(_shape, dtype=object)
    for i in range(mat0.shape[0]):
        for j in range(mat1.shape[1]):
            vec0 = mat0[i]
            vec1 = mat1[:, j]
            _vars[i, j] = reduce(lambda x, y: x + y, vec0 * vec1)
    return _vars.reshape(shape)


def cmvm(cm: np.ndarray, v: 'FixedVariableArray', solver_options: solver_options_t) -> np.ndarray:
    offload_fn = solver_options.get('offload_fn', None)
    mask = offload_fn(cm, v) if offload_fn is not None else None
    if mask is not None and np.any(mask):
        mask = np.astype(mask, np.bool_)
        assert mask.shape == cm.shape, f'Offload mask shape {mask.shape} does not match CM shape {cm.shape}'
        offload_cm = cm * mask.astype(cm.dtype)
        cm = cm * (~mask).astype(cm.dtype)
        if np.all(cm == 0):
            return mmm(v._vars, offload_cm)
    else:
        offload_cm = None
    _qintervals = [QInterval(float(_v.low), float(_v.high), float(_v.step)) for _v in v._vars]
    _latencies = [float(_v.latency) for _v in v._vars]
    qintervals = NumbaList(_qintervals)  # type: ignore
    latencies = NumbaList(_latencies)  # type: ignore
    hwconf = v._vars.ravel()[0].hwconf
    solver_options = solver_options.copy()
    solver_options.setdefault('adder_size', hwconf.adder_size)
    solver_options.setdefault('carry_size', hwconf.carry_size)
    _mat = np.ascontiguousarray(cm.astype(np.float32))
    solver_options.pop('offload_fn', None)
    sol = solve(_mat, qintervals=qintervals, latencies=latencies, **solver_options)  # type: ignore
    _r: np.ndarray = sol(v._vars)
    if offload_cm is not None:
        _r = _r + mmm(v._vars, offload_cm)
    return _r


def offload_mask(cm: NDArray, v: 'FixedVariableArray') -> NDArray[np.bool_]:
    assert v.ndim == 1
    assert cm.ndim == 2
    assert cm.shape[0] == v.shape[0]
    bits = np.sum(v.kif, axis=0)[:, None]
    return (bits == 0) & (cm != 0)


_unary_functions = (
    np.sin,
    np.cos,
    np.tan,
    np.exp,
    np.log,
    np.invert,
    np.sqrt,
    np.tanh,
    np.sinh,
    np.cosh,
    np.arccos,
    np.arcsin,
    np.arctan,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.exp2,
    np.expm1,
    np.log2,
    np.log10,
    np.log1p,
    np.cbrt,
    np.reciprocal,
)


class FixedVariableArray:
    """Symbolic array of FixedVariable for tracing operations. Supports numpy ufuncs and array functions."""

    __array_priority__ = 100

    def __array_function__(self, func, types, args, kwargs):
        if func in (np.mean, np.sum, np.amax, np.amin, np.prod, np.max, np.min):
            match func:
                case np.mean:
                    _x = reduce(lambda x, y: x + y, *args, **kwargs)
                    return _x * (_x.size / self._vars.size)
                case np.sum:
                    return reduce(lambda x, y: x + y, *args, **kwargs)
                case np.max | np.amax:
                    return reduce(_max_of, *args, **kwargs)
                case np.min | np.amin:
                    return reduce(_min_of, *args, **kwargs)
                case np.prod:
                    return reduce(lambda x, y: x * y, *args, **kwargs)
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

        if func is np.dot:
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == '__call__', f'Only __call__ method is supported for ufuncs, got {method}'

        match ufunc:
            case np.add | np.subtract | np.multiply | np.true_divide | np.negative:
                inputs = [to_raw_arr(x) for x in inputs]
                return FixedVariableArray(ufunc(*inputs, **kwargs), self.solver_options)

            case np.negative:
                assert len(inputs) == 1
                return FixedVariableArray(ufunc(to_raw_arr(inputs[0]), **kwargs), self.solver_options)

            case np.maximum | np.minimum:
                op = _max_of if ufunc is np.maximum else _min_of
                a, b = np.broadcast_arrays(inputs[0], inputs[1])
                shape = a.shape
                a, b = a.ravel(), b.ravel()
                r = np.empty(a.size, dtype=object)
                for i in range(a.size):
                    r[i] = op(a[i], b[i])
                return FixedVariableArray(r.reshape(shape), self.solver_options)

            case np.matmul:
                assert len(inputs) == 2
                assert isinstance(inputs[0], FixedVariableArray) or isinstance(inputs[1], FixedVariableArray)
                if isinstance(inputs[0], FixedVariableArray):
                    return inputs[0].matmul(inputs[1])
                else:
                    return inputs[1].rmatmul(inputs[0])

            case np.power:
                assert len(inputs) == 2
                base, exp = inputs
                return base**exp

            case np.abs | np.absolute:
                assert len(inputs) == 1
                assert inputs[0] is self
                arr = self._vars.ravel()
                r = np.array([v.__abs__() for v in arr])
                return FixedVariableArray(r.reshape(self.shape), self.solver_options)

            case np.square:
                assert len(inputs) == 1
                assert inputs[0] is self
                return self**2

        if ufunc in _unary_functions:
            assert len(inputs) == 1
            assert inputs[0] is self
            return self.apply(ufunc)

        raise NotImplementedError(f'Unsupported ufunc: {ufunc}')

    def __init__(
        self,
        vars: NDArray,
        solver_options: solver_options_t | None = None,
    ):
        _vars = np.array(vars)
        _vars_f = _vars.ravel()
        hwconf = next(iter(v for v in _vars_f if isinstance(v, FixedVariable))).hwconf
        for i, v in enumerate(_vars_f):
            if not isinstance(v, FixedVariable):
                _vars_f[i] = FixedVariable(float(v), float(v), 1.0, hwconf=hwconf)
        self._vars = _vars
        _solver_options = signature(solve).parameters
        _solver_options = {k: v.default for k, v in _solver_options.items() if v.default is not v.empty}
        if solver_options is not None:
            _solver_options.update(solver_options)
        _solver_options.pop('qintervals', None)
        _solver_options.pop('latencies', None)
        self.solver_options: solver_options_t = _solver_options  # type: ignore

    @classmethod
    def from_lhs(
        cls,
        low: NDArray[np.floating],
        high: NDArray[np.floating],
        step: NDArray[np.floating],
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
        latency: np.ndarray | float = 0.0,
        solver_options: solver_options_t | None = None,
    ):
        low, high, step = np.array(low), np.array(high), np.array(step)
        shape = low.shape
        assert shape == high.shape == step.shape

        low, high, step = low.ravel(), high.ravel(), step.ravel()
        latency = np.full_like(low, latency) if isinstance(latency, (int, float)) else latency.ravel()

        vars = []
        for l, h, s, lat in zip(low, high, step, latency):
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
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
        latency: NDArray[np.floating] | float = 0.0,
        solver_options: solver_options_t | None = None,
    ):
        mask = k + i + f <= 0
        k = np.where(mask, 0, k)
        i = np.where(mask, 0, i)
        f = np.where(mask, 0, f)
        step = 2.0**-f
        _high = 2.0**i
        high, low = _high - step, -_high * k
        return cls.from_lhs(low, high, step, hwconf, latency, solver_options)

    def matmul(self, other) -> 'FixedVariableArray':
        if self.collapsed:
            self_mat = np.array([v.low for v in self._vars.ravel()], dtype=np.float64).reshape(self._vars.shape)
            if isinstance(other, FixedVariableArray):
                if not other.collapsed:
                    return self_mat @ other  # type: ignore
                other_mat = np.array([v.low for v in other._vars.ravel()], dtype=np.float64).reshape(other._vars.shape)
            else:
                other_mat = np.array(other, dtype=np.float64)

            r = self_mat @ other_mat
            return FixedVariableArray.from_lhs(
                low=r,
                high=r,
                step=np.ones_like(r),
                hwconf=self._vars.ravel()[0].hwconf,
                solver_options=self.solver_options,
            )

        if isinstance(other, FixedVariableArray):
            other = other._vars
        if not isinstance(other, np.ndarray):
            other = np.array(other)
        if any(isinstance(x, FixedVariable) for x in other.ravel()):
            mat0, mat1 = self._vars, other
            _vars = mmm(mat0, mat1)
            return FixedVariableArray(_vars, self.solver_options)

        solver_options = (self.solver_options or {}).copy()
        shape0, shape1 = self.shape, other.shape
        assert shape0[-1] == shape1[0], f'Matrix shapes do not match: {shape0} @ {shape1}'
        contract_len = shape1[0]
        out_shape = shape0[:-1] + shape1[1:]
        mat0, mat1 = self.reshape((-1, contract_len)), other.reshape((contract_len, -1))
        r = []
        for i in range(mat0.shape[0]):
            vec = mat0[i]
            _r = cmvm(mat1, vec, solver_options)
            r.append(_r)
        r = np.array(r).reshape(out_shape)
        return FixedVariableArray(r, self.solver_options)

    def __matmul__(self, other):
        return self.matmul(other)

    def rmatmul(self, other):
        mat1 = np.moveaxis(other, -1, 0)
        mat0 = np.moveaxis(self, 0, -1)  # type: ignore
        ndim0, ndim1 = mat0.ndim, mat1.ndim
        r = mat0 @ mat1

        _axes = tuple(range(0, ndim0 + ndim1 - 2))
        axes = _axes[ndim0 - 1 :] + _axes[: ndim0 - 1]
        return r.transpose(axes)

    def __rmatmul__(self, other):
        return self.rmatmul(other)

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
        if _power == power and _power >= 0:
            return FixedVariableArray(self._vars**_power, self.solver_options)
        else:
            return self.apply(lambda x: x**power)

    def relu(
        self,
        i: NDArray[np.integer] | None = None,
        f: NDArray[np.integer] | None = None,
        round_mode: str = 'TRN',
    ):
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
        if any(x is None for x in (k, i, f)):
            kif = self.kif
        k = np.broadcast_to(k, shape) if k is not None else kif[0]  # type: ignore
        i = np.broadcast_to(i, shape) if i is not None else kif[1]  # type: ignore
        f = np.broadcast_to(f, shape) if f is not None else kif[2]  # type: ignore
        ret = []
        for v, k, i, f in zip(self._vars.ravel(), k.ravel(), i.ravel(), f.ravel()):  # type: ignore
            ret.append(v.quantize(k=k, i=i, f=f, overflow_mode=overflow_mode, round_mode=round_mode))
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options)

    def flatten(self):
        return FixedVariableArray(self._vars.flatten(), self.solver_options)

    def reshape(self, *shape):
        return FixedVariableArray(self._vars.reshape(*shape), self.solver_options)

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
        """[k, i, f] array"""
        shape = self._vars.shape
        kif = np.array([v.kif for v in self._vars.ravel()]).reshape(*shape, 3)
        return np.moveaxis(kif, -1, 0)

    @property
    def lhs(self):
        """[low, high, step] array"""
        shape = self._vars.shape
        lhs = np.array([(v.low, v.high, v.step) for v in self._vars.ravel()], dtype=np.float32).reshape(*shape, 3)
        return np.moveaxis(lhs, -1, 0)

    @property
    def latency(self):
        """Maximum latency among all elements."""
        return np.array([v.latency for v in self._vars.ravel()]).reshape(self._vars.shape)

    @property
    def collapsed(self):
        return all(v.low == v.high for v in self._vars.ravel())

    def apply(self, fn: Callable[[NDArray], NDArray]) -> 'RetardedFixedVariableArray':
        """Apply a unary operator to all elements, returning a RetardedFixedVariableArray."""
        return RetardedFixedVariableArray(
            self._vars,
            self.solver_options,
            operator=fn,
        )

    @property
    def T(self):
        return self.transpose()


class FixedVariableArrayInput(FixedVariableArray):
    """Similar to FixedVariableArray, but initializes all elements as FixedVariableInput - the precisions are unspecified when initialized, and the highest precision requested (i.e., quantized to) will be recorded for generation of the logic."""

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, -1, -1),
        solver_options: solver_options_t | None = None,
        latency=0.0,
    ):
        _vars = np.empty(shape, dtype=object)
        _vars_f = _vars.ravel()
        for i in range(_vars.size):
            _vars_f[i] = FixedVariableInput(latency, hwconf)
        super().__init__(_vars, solver_options)


def make_table(fn: Callable[[NDArray], NDArray], qint: QInterval) -> LookupTable:
    low, high, step = qint
    n = round(abs(high - low) / step) + 1
    return LookupTable(fn(np.linspace(low, high, n)))


class RetardedFixedVariableArray(FixedVariableArray):
    """Ephemeral FixedVariableArray generated from operations of unspecified output precision.
    This object translates to normal FixedVariableArray upon quantization.
    Does not inherit the maximum precision like FixedVariableArrayInput.

    This object can be used in two ways:
    1. Quantization with specified precision, which converts to FixedVariableArray.
    2. Apply an further unary operation, which returns another RetardedFixedVariableArray. (e.g., composite functions)
    """

    def __init__(self, vars: NDArray, solver_options: solver_options_t | None, operator: Callable[[NDArray], NDArray]):
        self._operator = operator
        super().__init__(vars, solver_options)

    def __array_function__(self, ufunc, method, *inputs, **kwargs):
        raise RuntimeError('RetardedFixedVariableArray only supports quantization or further unary operations.')

    def apply(self, fn: Callable[[NDArray], NDArray]) -> 'RetardedFixedVariableArray':
        return RetardedFixedVariableArray(
            self._vars,
            self.solver_options,
            operator=lambda x: fn(self._operator(x)),
        )

    def quantize(
        self,
        k: NDArray[np.integer] | np.integer | int | None = None,
        i: NDArray[np.integer] | np.integer | int | None = None,
        f: NDArray[np.integer] | np.integer | int | None = None,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        if any(x is None for x in (k, i, f)):
            assert all(x is not None for x in (k, i, f)), 'Either all or none of k, i, f must be specified'
            _k = _i = _f = [None] * self.size
        else:
            _k = np.broadcast_to(k, self.shape).ravel()  # type: ignore
            _i = np.broadcast_to(i, self.shape).ravel()  # type: ignore
            _f = np.broadcast_to(f, self.shape).ravel()  # type: ignore

            op = lambda x: _quantize(self._operator(x), k, i, f, overflow_mode, round_mode)  # type: ignore

        local_tables: dict[tuple[QInterval, tuple[int, int, int]] | QInterval, LookupTable] = {}
        variables = []
        for v, _kk, _ii, _ff in zip(self._vars.ravel(), _k, _i, _f):
            v: FixedVariable
            qint = v.qint if v._factor >= 0 else QInterval(v.qint.max, v.qint.min, v.qint.step)
            if (_kk is None) or (_ii is None) or (_ff is None):
                op = self._operator
                _key = qint
            else:
                op = lambda x: _quantize(self._operator(x), _kk, _ii, _ff, overflow_mode, round_mode)  # type: ignore
                _key = (qint, (int(_kk), int(_ii), int(_ff)))

            if _key in local_tables:
                table = local_tables[_key]
            else:
                table = make_table(op, qint)
                local_tables[_key] = table
            variables.append(v.lookup(table))

        variables = np.array(variables).reshape(self._vars.shape)
        return FixedVariableArray(variables, self.solver_options)

    def __repr__(self):
        return 'Retarded' + super().__repr__()

    @property
    def kif(self):
        raise RuntimeError('RetardedFixedVariableArray does not have defined kif until quantized.')
