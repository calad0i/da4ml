from collections.abc import Sequence
from decimal import Decimal
from functools import reduce, singledispatch
from math import ceil, floor, log2
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import numpy as np
from numba import jit
from numpy import float32, int8
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..trace.tracer import FixedVariable


class QInterval(NamedTuple):
    """A class representing a quantized interval: [min, max] with a step size."""

    min: float
    max: float
    step: float

    @classmethod
    def from_kif(cls, k: int | bool, i: int, f: int):
        _high = 2.0**i
        step = 2.0**-f
        low, high = -k * step, _high - step
        return cls(low, high, step)

    @classmethod
    def from_precision(cls, prec: 'Precision'):
        return cls.from_kif(*prec)

    @property
    def precision(self):
        return Precision.from_qint(self)

    def __repr__(self):
        return f'[{self.min}, {self.max}, {self.step}]'


class Precision(NamedTuple):
    """A class representing the precision of a quantized interval."""

    keep_negative: bool
    integers: int
    fractional: int

    def __str__(self):
        k, i, f = self.keep_negative, self.integers, self.fractional
        k, B, I = k, i + f + k, i + k
        return f'fixed({k}, {B}, {I})'

    def __repr__(self):
        return str(self)

    @classmethod
    def from_qint(cls, qint: QInterval, symmetric: bool = False):
        return _minimal_kif(qint, symmetric=symmetric)

    @property
    def qint(self):
        return QInterval.from_kif(*self)


class Op(NamedTuple):
    """An operation representing data[id0] +/- data[id1] * 2**shift, and precision & latency & cost associated with it."""

    id0: int
    id1: int
    # -1: copy from external id0; -2: relu; -3: qinterval change with bitdrop (WRAP & TRN); -4 copy from same buffer
    sub: bool
    shift: int
    qint: QInterval
    latency: float
    cost: float


class Pair(NamedTuple):
    """An operation representing data[id0] +/- data[id1] * 2**shift."""

    id0: int
    id1: int
    sub: bool
    shift: int


class DAState(NamedTuple):
    """Internal state of the DA algorithm."""

    shifts: tuple[NDArray[int8], NDArray[int8]]
    expr: list[NDArray[int8]]
    ops: list[Op]
    freq_stat: dict[Pair, int]
    kernel: NDArray[float32]


def _minimal_kif(qi: QInterval, symmetric: bool = False) -> Precision:
    """Calculate the minimal KIF for a given QInterval.

    Parameters
    ----------
    qi : QInterval
        The QInterval to calculate the KIF for.
    symmetric : bool
        Only relevant if qi may be negative. If True, -2**i will be regarded as forbidden.
        May be useful in special cases only.
        Default is False.

    Returns
    -------
    Precision
        A named tuple with the KIF values.
    """

    if qi.min == qi.max == 0:
        return Precision(keep_negative=False, integers=0, fractional=0)
    keep_negative = qi.min < 0
    fractional = int(-log2(qi.step))
    int_min, int_max = round(qi.min / qi.step), round(qi.max / qi.step)
    if symmetric:
        bits = int(ceil(log2(max(abs(int_min), int_max) + 1)))
    else:
        bits = int(ceil(log2(max(abs(int_min), int_max + 1))))
    integers = bits - fractional
    return Precision(keep_negative=keep_negative, integers=integers, fractional=fractional)


if TYPE_CHECKING:

    def minimal_kif(qi: QInterval, symmetric: bool = False) -> Precision: ...
else:
    minimal_kif = jit(_minimal_kif)


T = TypeVar('T', 'FixedVariable', float, int, np.float32, np.float64, Decimal)


@singledispatch
def _relu(v: 'T', i: int | None = None, f: int | None = None, inv: bool = False) -> 'T':
    from ..trace.fixed_veriable import FixedVariable

    assert isinstance(v, FixedVariable), f'Unknown type {type(v)} for symbolic relu'
    return v.relu(i, f)


@_relu.register(float)
@_relu.register(int)
@_relu.register(np.float32)
@_relu.register(np.float64)
def _(v, i: int | None = None, f: int | None = None, inv: bool = False):
    if inv:
        v = -v
    v = max(0, v)
    if f is not None:
        sf = 2.0**f
        v = floor(v * sf) / sf
    if i is not None:
        v = v % 2.0**i
    return v


@_relu.register
def _(v: Decimal, i: int | None = None, f: int | None = None, inv: bool = False):
    if inv:
        v = -v
    v = max(Decimal(0), v)
    if f is not None:
        sf = Decimal(2) ** f
        v = floor(v * sf) / sf
    if i is not None:
        v = v % Decimal(2) ** i
    return v


def relu(
    vars: list['FixedVariable | float | int'],
    i: Sequence[int | None] | None | int = None,
    f: Sequence[int | None] | None | int = None,
    inv=False,
):
    _i = i if isinstance(i, Sequence) else [i] * len(vars)
    _f = f if isinstance(f, Sequence) else [f] * len(vars)
    return [_relu(v, i_, f_, inv) for v, i_, f_ in zip(vars, _i, _f)]


class Solution(NamedTuple):
    """Solution of single CMVM problem. The

    Attributes
    ----------
    inp_qint: list[QInterval]
        Input quantized intervals.
    inp_lat: list[float]
        Input latencies (time in which the input data is available).
    in_shift: list[int]
        Input shifts
    out_qint: list[QInterval]
        Output quantized intervals.
    out_lat: list[float]
        Output latencies (time in which the output data is available).
    out_idx: list[int]
        Index of the operations to which generates the outputs.
    out_shift: list[int]
        Output shifts.
    out_neg: list[bool]
        Output signs, True if the output should be negated.
    ops: list[Op]
        Operations used in the solution.

    Properties
    ----------
    kernel: NDArray[float32]
        The kernel which the solution implements: vec @ kernel = solution(vec).
    cost: float
        The cost of the solution.
    latency: tuple[float, float]
        The minimum and maximum latency of the solution.
    shape: tuple[int, int]
        The shape of the corresponding kernel matrix.



    The core part of the solution is the operations in the ops list.
    For any i, ops[i] will execute as
    if ops[i].id1 == -1:
        data[i] = inp_data[op.id0] * 2**in_shift[op.id0]
    else:
        data[i] = data[op.id0] +/- data[op.id1] * 2**op.shift
    After all operations are executed, the output data is read from data[op.out_idx] and multiplied by 2**out_shift.

    """

    shape: tuple[int, int]
    inp_shift: list[int]
    out_idxs: list[int]
    out_shifts: list[int]
    out_neg: list[bool]
    ops: list[Op]
    carry_size: int
    adder_size: int

    def __call__(self, inp: list | np.ndarray | tuple, quantize=False, debug=False):
        """Executes the solution on the input data.

        Parameters
        ----------
        inp : list | np.ndarray | tuple
            Input data to be processed. The input data should be a list or numpy array of objects.
        quantize : bool
            If True, the input data will be quantized to the output quantization intervals.
            Only floating point data types are supported when quantize is True.
            Default is False.

        Returns
        -------
        np.ndarray
            The output data after applying the operations defined in the solution.

        """
        buf = np.empty(len(self.ops), dtype=object)
        inp = np.asarray(inp)

        inp_qint = [op.qint for op in self.ops if op.id1 == -1]
        if quantize:  # TRN and WRAP
            k, i, f = map(np.array, zip(*map(minimal_kif, inp_qint)))
            eps = 2.0**-f
            _low, _high = -k * 2.0 ** (i + f), 2.0 ** (i + f) - 1
            inp = eps * ((np.floor(inp / eps) - _low) % (_high - _low) + _low)

        inp = inp * (2.0 ** np.asarray(self.inp_shift))
        for i, op in enumerate(self.ops):
            if op.id1 == -1:  # copy from external buf
                buf[i] = inp[op.id0]
                continue
            elif op.id1 == -2:  # relu, -(relu(-))
                v = buf[op.id0]
                _, _i, _f = _minimal_kif(op.qint)
                buf[i] = _relu(v, _i, _f, inv=op.sub)
                # buf[i] = max(0, v)
                continue
            elif op.id1 == -3:  # quantization only
                v = buf[op.id0]
                _k, _i, _f = _minimal_kif(op.qint)
                bias = _k * 2.0**_i
                b = _k + _i + _f
                eps = 2.0**-_f
                buf[i] = eps * ((np.floor(v / eps) + bias) % 2**b - bias)
                continue
            elif op.id1 == -4:  # define constant
                buf[i] = op.qint.min
                continue
            assert op.id1 >= 0, f'Unknown id1 {op.id1} in {op}'
            v0, v1 = buf[op.id0], buf[op.id1]
            if op.sub:
                v1 = -v1
            buf[i] = v0 + 2.0**op.shift * v1

        sf = 2.0 ** np.asarray(self.out_shifts)
        sign = np.where(self.out_neg, -1, 1)
        out_idx = np.asarray(self.out_idxs)
        mask = np.where(out_idx < 0, 0, 1)
        if debug:
            for i, v in enumerate(buf):
                print(f'buf[{i}] = {v}')
        return buf[out_idx] * sf * sign * mask

    @property
    def kernel(self):
        """the kernel represented by the solution."""
        kernel = np.empty(self.shape, dtype=np.float32)
        for i, one_hot in enumerate(np.identity(self.shape[0])):
            kernel[i] = self(one_hot)
        return kernel

    @property
    def cost(self):
        """Returns the cost of the solution."""
        return float(sum(op.cost for op in self.ops))

    @property
    def latency(self):
        """Returns the latency of the solution."""
        latency = [self.ops[i].latency for i in self.out_idxs]
        if len(latency) == 0:
            return 0.0, 0.0
        return min(latency), max(latency)

    def __repr__(self):
        n_in, n_out = self.shape
        cost = self.cost
        lat_min, lat_max = self.latency
        return f'Solution([{n_in} -> {n_out}], cost={cost}, latency={lat_min}-{lat_max})'

    @property
    def out_latency(self):
        """Returns the output latency of the solution."""
        return [self.ops[i].latency if i >= 0 else 0.0 for i in self.out_idxs]

    @property
    def out_qint(self):
        """Returns the output quantization intervals of the solution."""
        buf = []
        for i, idx in enumerate(self.out_idxs):
            _min, _max, _step = self.ops[idx].qint
            sf = 2.0 ** self.out_shifts[i]
            _min, _max, _step = _min * sf, _max * sf, _step * sf
            if self.out_neg[i]:
                _min, _max = -_max, -_min
            buf.append(QInterval(_min, _max, _step))
        return buf

    @property
    def inp_latency(self):
        """Returns the input latencies of the solution."""
        return [self.ops[i].latency for i in range(self.shape[0])]

    @property
    def inp_qint(self):
        """Returns the input quantization intervals of the solution."""
        return [self.ops[i].qint for i in range(self.shape[0])]


class CascadedSolution(NamedTuple):
    """A solution that implements cascaded matrix-vector multiplications through multiple CMVM stages.

    CascadedSolution represents a sequence of Solution objects where the output of each stage
    is fed as input to the next stage.

    Attributes
    ----------
    solutions: tuple[Solution, ...]
        A tuple containing the individual Solution objects for each stage of the cascade.

    Properties
    ----------
    kernel: NDArray[float32]
        The overall kernel matrix which the cascaded solution implements: vec @ kernel = solution(vec).
        This is calculated as the matrix product of all individual solution kernels.
    cost: float
        The total cost of the cascaded solution, computed as the sum of the costs of all stages.
    latency: tuple[float, float]
        The minimum and maximum latency of the cascaded solution.
    inp_qint: list[QInterval]
        Input quantization intervals
    inp_lat: list[float]
        Input latencies
    in_shift: list[int]
        Input shifts
    out_qint: list[QInterval]
        Output quantization intervals
    out_lat: list[float]
        Output latencies
    out_shift: list[int]
        Output shifts
    out_neg: list[bool]
        Output signs
    shape: tuple[int, int]
        The shape of the corresponding kernel matrix.
    """

    solutions: tuple[Solution, ...]

    def __call__(self, inp: list | np.ndarray | tuple, quantize=False, debug=False):
        out = np.asarray(inp)
        for sol in self.solutions:
            out = sol(out, quantize=quantize, debug=debug)
        return out

    @property
    def kernel(self):
        return reduce(lambda x, y: x @ y, [sol.kernel for sol in self.solutions])

    @property
    def cost(self):
        return sum(sol.cost for sol in self.solutions)

    @property
    def latency(self):
        return self.solutions[-1].latency

    @property
    def inp_qint(self):
        return self.solutions[0].inp_qint

    @property
    def inp_latency(self):
        return self.solutions[0].inp_latency

    @property
    def out_qint(self):
        return self.solutions[-1].out_qint

    @property
    def out_latencies(self):
        return self.solutions[-1].out_latency

    @property
    def shape(self):
        return self.solutions[0].shape[0], self.solutions[-1].shape[1]

    @property
    def inp_shift(self):
        return self.solutions[0].inp_shift

    @property
    def out_shift(self):
        return self.solutions[-1].out_shifts

    @property
    def out_neg(self):
        return self.solutions[-1].out_neg

    def __repr__(self) -> str:
        n_ins = [sol.shape[0] for sol in self.solutions] + [self.shape[1]]
        shape_str = ' -> '.join(map(str, n_ins))
        _cost = self.cost
        lat_min, lat_max = self.latency
        return f'CascatedSolution([{shape_str}], cost={_cost}, latency={lat_min}-{lat_max})'
