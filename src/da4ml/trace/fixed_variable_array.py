from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from .._binary import get_lsb_loc
from ..cmvm import solve, solver_options_t
from .fixed_variable import FixedVariable, FixedVariableInput, HWConfig, LookupTable, QInterval
from .ops import _quantize, einsum, reduce, sort

T = TypeVar('T')

_ARRAY_FN: dict = {}
_UFUNC: dict = {}


def _array_fn(*funcs):
    def deco(fn):
        for f in funcs:
            _ARRAY_FN[f] = fn
        return fn

    return deco


def _ufunc(*ufuncs):
    def deco(fn):
        for u in ufuncs:
            _UFUNC[u] = fn
        return fn

    return deco


def to_raw_arr(obj: T) -> T:
    if isinstance(obj, tuple):
        return tuple(to_raw_arr(x) for x in obj)  # type: ignore
    elif isinstance(obj, list):
        return [to_raw_arr(x) for x in obj]  # type: ignore
    elif isinstance(obj, dict):
        return {k: to_raw_arr(v) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, FixedVariableArray):
        return np.asarray(obj)  # type: ignore
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
    v_raw = np.asarray(v)
    if mask is not None and np.any(mask):
        mask = np.astype(mask, np.bool_)
        assert mask.shape == cm.shape, f'Offload mask shape {mask.shape} does not match CM shape {cm.shape}'
        offload_cm = cm * mask.astype(cm.dtype)
        cm = cm * (~mask).astype(cm.dtype)
        if np.all(cm == 0):
            return mmm(v_raw, offload_cm)
    else:
        offload_cm = None
    qintervals = [QInterval(float(_v.low), float(_v.high), float(_v.step)) for _v in v_raw]
    latencies = [float(_v.latency) for _v in v_raw]
    _mat = np.ascontiguousarray(cm.astype(np.float32))
    solver_options = solver_options.copy()
    solver_options.pop('offload_fn', None)
    sol = solve(_mat, qintervals=qintervals, latencies=latencies, **solver_options)  # type: ignore
    _r: np.ndarray = sol(v_raw)
    if offload_cm is not None:
        _r = _r + mmm(v_raw, offload_cm)
    return _r


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


class FixedVariableArray(np.ndarray):
    """Symbolic array of FixedVariable for tracing operations. Supports numpy ufuncs and array functions."""

    __array_priority__ = 100

    def __new__(
        cls,
        vars: NDArray,
        solver_options: solver_options_t | None = None,
        hwconf: HWConfig | tuple[int, int, int] | None = None,
    ):
        _arr = np.array(vars, dtype=object)
        _f = _arr.ravel()
        if hwconf is None:
            hwconf = next(iter(v for v in _f if isinstance(v, FixedVariable))).hwconf
        hwconf = HWConfig(*hwconf)
        for i, v in enumerate(_f):
            if not isinstance(v, FixedVariable):
                v = float(v)
                _f[i] = FixedVariable(v, v, 2 ** get_lsb_loc(v), hwconf=hwconf)
        obj = np.ndarray.__new__(cls, shape=_arr.shape, dtype=object)
        obj[...] = _arr
        obj.hwconf = hwconf
        _so = solver_options.copy() if solver_options is not None else {}
        _so.pop('qintervals', None)
        _so.pop('latencies', None)
        obj.solver_options: solver_options_t = _so  # type: ignore
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.solver_options = obj.solver_options
        self.hwconf = obj.hwconf

    def __array_function__(self, func, types, args, kwargs):
        if func in _ARRAY_FN:
            return _ARRAY_FN[func](*args, **kwargs)
        args, kwargs = to_raw_arr(args), to_raw_arr(kwargs)
        return FixedVariableArray(func(*args, **kwargs), self.solver_options, hwconf=self.hwconf)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == '__call__', f'Only __call__ method is supported for ufuncs, got {method}'
        if ufunc in _UFUNC:
            return _UFUNC[ufunc](self, ufunc, *inputs, **kwargs)
        if ufunc in _unary_functions:
            assert len(inputs) == 1 and inputs[0] is self
            return self.apply(ufunc)
        raise NotImplementedError(f'Unsupported ufunc: {ufunc}')

    @classmethod
    def from_lhs(
        cls,
        low: NDArray[np.floating],
        high: NDArray[np.floating],
        step: NDArray[np.floating],
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, 1, -1),
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
                latency=float(lat),
            )
            vars.append(var)
        vars = np.array(vars).reshape(shape)
        return cls(vars, solver_options)

    @classmethod
    def from_kif(
        cls,
        k: NDArray[np.bool_ | np.integer],
        i: NDArray[np.integer],
        f: NDArray[np.integer],
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, 1, -1),
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
            self_mat = np.array([v.low for v in np.asarray(self).ravel()], dtype=np.float64).reshape(self.shape)
            if isinstance(other, FixedVariableArray):
                if not other.collapsed:
                    return self_mat @ other  # type: ignore
                other_mat = np.array([v.low for v in np.asarray(other).ravel()], dtype=np.float64).reshape(other.shape)
            else:
                other_mat = np.array(other, dtype=np.float64)
            r = self_mat @ other_mat
            return FixedVariableArray.from_lhs(
                low=r,
                high=r,
                step=np.ones_like(r),
                hwconf=self.hwconf,
                solver_options=self.solver_options,
            )

        _other = np.asarray(other) if isinstance(other, FixedVariableArray) else np.array(other)
        if any(isinstance(x, FixedVariable) for x in _other.ravel()):
            _vars = mmm(np.asarray(self), _other)
            return FixedVariableArray(_vars, self.solver_options, hwconf=self.hwconf)

        solver_options = (self.solver_options or {}).copy()
        shape0, shape1 = self.shape, _other.shape
        assert shape0[-1] == shape1[0], f'Matrix shapes do not match: {shape0} @ {shape1}'
        contract_len = shape1[0]
        out_shape = shape0[:-1] + shape1[1:]
        mat0, mat1 = self.reshape((-1, contract_len)), _other.reshape((contract_len, -1))
        r = []
        for i in range(mat0.shape[0]):
            vec = mat0[i]
            _r = cmvm(mat1, vec, solver_options)
            r.append(_r)
        r = np.array(r).reshape(out_shape)
        return FixedVariableArray(r, self.solver_options, hwconf=self.hwconf)

    def rmatmul(self, other):
        mat1 = np.moveaxis(other, -1, 0)
        mat0 = np.moveaxis(self, 0, -1)  # type: ignore
        ndim0, ndim1 = mat0.ndim, mat1.ndim
        r = mat0 @ mat1

        _axes = tuple(range(0, ndim0 + ndim1 - 2))
        axes = _axes[ndim0 - 1 :] + _axes[: ndim0 - 1]
        return r.transpose(axes)

    def __getitem__(self, item):
        if isinstance(item, _ArgsortDelayedIndex):
            ret = sort(*item.args, **item.kwargs, aux_value=self)[1]
            for s in item._slicing:
                ret = ret[s]
            return ret
        return super().__getitem__(item)

    def __gt__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av > bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __lt__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av < bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __ge__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av >= bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __le__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av <= bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __and__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av & bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __or__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av | bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __invert__(self):
        r = np.array([~av for av in np.asarray(self).ravel()])
        return FixedVariableArray(r.reshape(self.shape), self.solver_options, hwconf=self.hwconf)

    def __xor__(self, other):
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av ^ bv for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __eq__(self, other):  # type: ignore
        return ~(self.__ne__(other))

    def __ne__(self, other):  # type: ignore
        _b = np.asarray(other) if isinstance(other, FixedVariableArray) else other
        a, b = np.broadcast_arrays(np.asarray(self), _b)
        shape = a.shape
        r = np.array([av._ne(bv) for av, bv in zip(a.ravel(), b.ravel())])
        return FixedVariableArray(r.reshape(shape), self.solver_options, hwconf=self.hwconf)

    def __repr__(self):
        shape = self.shape
        _raw = np.asarray(self)
        hwconf_str = str(_raw.ravel()[0].hwconf)[8:]
        max_lat = max(v.latency for v in _raw.ravel())
        return f'FixedVariableArray(shape={shape}, hwconf={hwconf_str}, latency={max_lat})'

    def to_bool(self, reduction='any'):
        assert reduction in ('any', 'all'), f'Reduction must be either "any" or "all", got {reduction}'
        _arr = np.array([v.unary_bit_op(reduction) for v in np.asarray(self).ravel()]).reshape(self.shape)
        return FixedVariableArray(_arr, self.solver_options, hwconf=self.hwconf)

    def relu(
        self,
        i: NDArray[np.integer] | None = None,
        f: NDArray[np.integer] | None = None,
        round_mode: str = 'TRN',
    ):
        shape = self.shape
        _i = np.broadcast_to(i, shape) if i is not None else np.full(shape, None)
        _f = np.broadcast_to(f, shape) if f is not None else np.full(shape, None)
        ret = [v.relu(i=iv, f=fv, round_mode=round_mode) for v, iv, fv in zip(np.asarray(self).ravel(), _i.ravel(), _f.ravel())]  # type: ignore
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options, hwconf=self.hwconf)

    def quantize(
        self,
        k: NDArray[np.integer] | np.integer | int | None = None,
        i: NDArray[np.integer] | np.integer | int | None = None,
        f: NDArray[np.integer] | np.integer | int | None = None,
        overflow_mode: str = 'WRAP',
        round_mode: str = 'TRN',
    ):
        shape = self.shape
        if any(x is None for x in (k, i, f)):
            kif = self.kif
        k = np.broadcast_to(k, shape) if k is not None else kif[0]  # type: ignore
        i = np.broadcast_to(i, shape) if i is not None else kif[1]  # type: ignore
        f = np.broadcast_to(f, shape) if f is not None else kif[2]  # type: ignore
        ret = [
            v.quantize(k=kk, i=ii, f=ff, overflow_mode=overflow_mode, round_mode=round_mode)
            for v, kk, ii, ff in zip(np.asarray(self).ravel(), k.ravel(), i.ravel(), f.ravel())
        ]  # type: ignore
        return FixedVariableArray(np.array(ret).reshape(shape), self.solver_options, hwconf=self.hwconf)

    @property
    def kif(self):
        """[k, i, f] array"""
        shape = self.shape
        kif = np.array([v.kif for v in np.asarray(self).ravel()]).reshape(*shape, 3)
        return np.moveaxis(kif, -1, 0)

    @property
    def lhs(self):
        """[low, high, step] array"""
        shape = self.shape
        lhs = np.array([(v.low, v.high, v.step) for v in np.asarray(self).ravel()], dtype=np.float32).reshape(*shape, 3)
        return np.moveaxis(lhs, -1, 0)

    @property
    def latency(self):
        """Maximum latency among all elements."""
        return np.array([v.latency for v in np.asarray(self).ravel()]).reshape(self.shape)

    @property
    def collapsed(self):
        return all(v.low == v.high for v in np.asarray(self).ravel())

    def apply(self, fn: Callable[[NDArray], NDArray]) -> 'RetardedFixedVariableArray':
        """Apply a unary operator to all elements, returning a RetardedFixedVariableArray."""
        return RetardedFixedVariableArray(
            np.asarray(self),
            self.solver_options,
            operator=fn,
        )

    def as_new(self):
        """Create a new FixedVariableArray with the same shape and hardware configuration, but new FixedVariable instances."""
        shape = self.shape
        _arr = np.array([v._with(_from=(), opr='new', renew_id=True) for v in np.asarray(self).ravel()]).reshape(shape)
        return FixedVariableArray(_arr, self.solver_options, hwconf=self.hwconf)


class FixedVariableArrayInput(FixedVariableArray):
    """Similar to FixedVariableArray, but initializes all elements as FixedVariableInput - the precisions are unspecified when initialized, and the highest precision requested (i.e., quantized to) will be recorded for generation of the logic."""

    def __new__(
        cls,
        shape: tuple[int, ...] | int,
        hwconf: HWConfig | tuple[int, int, int] = HWConfig(1, 1, -1),
        solver_options: solver_options_t | None = None,
        latency=0.0,
    ):
        _arr = np.empty(shape, dtype=object)
        for i in range(_arr.size):
            _arr.ravel()[i] = FixedVariableInput(latency, hwconf)
        return super().__new__(cls, _arr, solver_options, hwconf=hwconf)


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

    def __new__(cls, vars: NDArray, solver_options: solver_options_t | None, operator: Callable[[NDArray], NDArray]):
        obj = super().__new__(cls, vars, solver_options)
        obj._operator = operator
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._operator = getattr(obj, '_operator', None)

    def __array_function__(self, func, types, args, kwargs):
        raise RuntimeError('RetardedFixedVariableArray only supports quantization or further unary operations.')

    def apply(self, fn: Callable[[NDArray], NDArray]) -> 'RetardedFixedVariableArray':
        return RetardedFixedVariableArray(
            np.asarray(self),
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

        local_tables: dict[tuple[QInterval, tuple[int, int, int]] | QInterval, LookupTable] = {}
        variables = []
        for v, _kk, _ii, _ff in zip(np.asarray(self).ravel(), _k, _i, _f):
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

        variables = np.array(variables).reshape(self.shape)
        return FixedVariableArray(variables, self.solver_options, hwconf=self.hwconf)

    def __repr__(self):
        return 'Retarded' + super().__repr__()

    @property
    def kif(self):
        raise RuntimeError('RetardedFixedVariableArray does not have defined kif until quantized.')


class _ArgsortDelayedIndex:
    def __init__(self, args, kwargs, slicing: tuple[slice | int, ...] = ()):
        self.args = args
        self.kwargs = kwargs
        self._slicing: tuple[slice | int, ...] = slicing

    def __getitem__(self, idx):
        return _ArgsortDelayedIndex(self.args, self.kwargs, self._slicing + (idx,))


@_array_fn(np.sum)
def _np_sum(*args, **kwargs):
    return reduce(lambda x, y: x + y, *args, **kwargs)


@_array_fn(np.mean)
def _np_mean(x, axis=None, **kw):
    r = reduce(lambda a, b: a + b, x, axis=axis)
    size = r.size if isinstance(r, FixedVariableArray) else 1
    return r * (size / x.size)


@_array_fn(np.amax, np.max)
def _np_amax(*args, **kwargs):
    return reduce(_max_of, *args, **kwargs)


@_array_fn(np.amin, np.min)
def _np_amin(*args, **kwargs):
    return reduce(_min_of, *args, **kwargs)


@_array_fn(np.prod)
def _np_prod(*args, **kwargs):
    return reduce(lambda x, y: x * y, *args, **kwargs)


@_array_fn(np.all)
def _np_all(x, axis=None, keepdims=False, **kw):
    _arr = np.array([v.unary_bit_op('any') for v in np.asarray(x).ravel()]).reshape(x.shape)
    x2 = FixedVariableArray(_arr, x.solver_options, hwconf=x.hwconf)
    return reduce(lambda a, b: a & b, x2, axis=axis, keepdims=keepdims)


@_array_fn(np.any)
def _np_any(x, axis=None, keepdims=False, **kw):
    _arr = np.array([v.unary_bit_op('any') for v in np.asarray(x).ravel()]).reshape(x.shape)
    x2 = FixedVariableArray(_arr, x.solver_options, hwconf=x.hwconf)
    return reduce(lambda a, b: a | b, x2, axis=axis, keepdims=keepdims)


@_array_fn(np.clip)
def _np_clip(a, a_min, a_max, out=None, **kw):
    _a, _amin, _amax = np.broadcast_arrays(np.asarray(a), a_min, a_max)
    shape = _a.shape
    r = np.array([_max_of(v, lo) for v, lo in zip(_a.ravel(), _amin.ravel())])
    r = np.array([_min_of(v, hi) for v, hi in zip(r, _amax.ravel())])
    return FixedVariableArray(r.reshape(shape), a.solver_options, hwconf=a.hwconf)


@_array_fn(np.einsum)
def _np_einsum(*args, **kwargs):
    from inspect import signature

    sig = signature(np.einsum)
    bind = sig.bind(*args, **kwargs)
    eq = args[0]
    operands = bind.arguments['operands']
    if isinstance(operands[0], str):
        operands = operands[1:]
    assert len(operands) == 2, 'Einsum on FixedVariableArray requires exactly two operands'
    assert bind.arguments.get('out', None) is None, 'Output argument is not supported'
    return einsum(eq, *operands)


@_array_fn(np.dot)
def _np_dot(a, b, out=None):
    assert out is None
    if not isinstance(a, FixedVariableArray):
        a = np.array(a)
    if not isinstance(b, FixedVariableArray):
        b = np.array(b)
    if a.shape and b.shape and a.shape[-1] == b.shape[0]:
        return a @ b
    assert a.size == 1 or b.size == 1, f'Error in dot product: {a.shape} @ {b.shape}'
    return a * b


@_array_fn(np.where)
def _np_where(condition, x=None, y=None):
    fva = next(v for v in (condition, x, y) if isinstance(v, FixedVariableArray))
    if isinstance(condition, FixedVariableArray):
        cond_fva = condition.to_bool('any')
        _cond, _x, _y = np.broadcast_arrays(
            np.asarray(cond_fva),
            np.asarray(x) if isinstance(x, FixedVariableArray) else x,
            np.asarray(y) if isinstance(y, FixedVariableArray) else y,
        )
        shape = _cond.shape
        r = [c.msb_mux(xv, yv) for c, xv, yv in zip(_cond.ravel(), _x.ravel(), _y.ravel())]
        return FixedVariableArray(np.array(r).reshape(shape), fva.solver_options, hwconf=fva.hwconf)
    return FixedVariableArray(
        np.where(condition, to_raw_arr(x), to_raw_arr(y)),
        fva.solver_options,
        hwconf=fva.hwconf,
    )


@_array_fn(np.sort)
def _np_sort(a, axis=-1, kind=None, order=None):
    return sort(a, axis=axis)


@_array_fn(np.argsort)
def _np_argsort(a, axis=-1, **kw):
    assert np.asarray(a).ndim == 1, 'Argsort on FixedVariableArray only supports 1D arrays'
    return _ArgsortDelayedIndex((a,), {'axis': axis})


@_ufunc(np.add, np.subtract, np.multiply, np.true_divide, np.negative)
def _ufunc_elementwise(arr, ufunc, *inputs, **kwargs):
    return FixedVariableArray(ufunc(*[to_raw_arr(x) for x in inputs], **kwargs), arr.solver_options, hwconf=arr.hwconf)


@_ufunc(np.maximum, np.minimum)
def _ufunc_minmax(arr, ufunc, *inputs, **kwargs):
    op = _max_of if ufunc is np.maximum else _min_of
    a, b = np.broadcast_arrays(to_raw_arr(inputs[0]), to_raw_arr(inputs[1]))
    shape = a.shape
    r = np.empty(a.size, dtype=object)
    for i in range(a.size):
        r[i] = op(a.ravel()[i], b.ravel()[i])
    return FixedVariableArray(r.reshape(shape), arr.solver_options, hwconf=arr.hwconf)


@_ufunc(np.matmul)
def _ufunc_matmul(arr, ufunc, *inputs, **kwargs):
    a, b = inputs
    if isinstance(a, FixedVariableArray):
        return a.matmul(b)
    return b.rmatmul(a)


@_ufunc(np.power)
def _ufunc_power(arr, ufunc, *inputs, **kwargs):
    base, exp = inputs
    if isinstance(exp, (int, float, np.integer, np.floating)):
        _exp = int(exp)
        if _exp == exp and _exp >= 0:
            return FixedVariableArray(to_raw_arr(base) ** _exp, arr.solver_options, hwconf=arr.hwconf)
    if isinstance(base, FixedVariableArray):
        return base.apply(lambda x: x**exp)
    raise NotImplementedError(f'Unsupported power: base={type(base)}, exp={type(exp)}')


@_ufunc(np.abs, np.absolute)
def _ufunc_abs(arr, ufunc, *inputs, **kwargs):
    r = np.array([v.__abs__() for v in np.asarray(arr).ravel()])
    return FixedVariableArray(r.reshape(arr.shape), arr.solver_options, hwconf=arr.hwconf)


@_ufunc(np.square)
def _ufunc_square(arr, ufunc, *inputs, **kwargs):
    return arr**2
