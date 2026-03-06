from math import gcd
from uuid import uuid4

from ..._binary import get_lsb_loc, iceil_log2
from ...trace.fixed_variable import LookupTable
from ...types import CombLogic, Op, QInterval, minimal_kif


def step_gcd(step0: float, step1: float) -> float:
    sf = 2 ** -min(get_lsb_loc(step0), get_lsb_loc(step1))
    s0_int, s1_int = round(step0 * sf), round(step1 * sf)
    return gcd(s0_int, s1_int) / sf


class PrimitiveInterval:
    def __init__(self, qint: QInterval):
        self.qint = qint
        self.uuid = uuid4()

    def __hash__(self):
        return hash(self.uuid)

    def __repr__(self):
        return str(self.qint)


class AffineInterval:
    def __init__(self, coeffs: dict[PrimitiveInterval, float], bias: float):
        self.coeffs = coeffs
        self.bias = bias

    @classmethod
    def new(cls, qint: QInterval):
        return cls({PrimitiveInterval(qint): 1.0}, 0.0)

    def eval(self, canon=True) -> QInterval:
        min_val = self.bias
        max_val = self.bias
        step = 2 ** get_lsb_loc(self.bias)
        for prim, coeff in self.coeffs.items():
            if coeff == 0:
                continue
            if coeff > 0:
                min_val += coeff * prim.qint.min
                max_val += coeff * prim.qint.max
            else:
                min_val += coeff * prim.qint.max
                max_val += coeff * prim.qint.min
            step = min(step, 2 ** get_lsb_loc(coeff * prim.qint.step))
        if canon:
            step = 2 ** get_lsb_loc(step)
        return QInterval(min_val, max_val, step)

    def __add__(self, other):
        if not isinstance(other, AffineInterval):
            return AffineInterval(self.coeffs, self.bias + other)
        new_coeffs = self.coeffs.copy()
        for k, v in other.coeffs.items():
            new_coeffs[k] = new_coeffs.get(k, 0.0) + v
        new_bias = self.bias + other.bias
        return AffineInterval(new_coeffs, new_bias)

    def __neg__(self):
        new_coeffs = {k: -v for k, v in self.coeffs.items()}
        new_bias = -self.bias
        return AffineInterval(new_coeffs, new_bias)

    def __mul__(self, other):
        if not isinstance(other, AffineInterval):
            new_coeffs = {k: v * other for k, v in self.coeffs.items()}
            new_bias = self.bias * other
            return AffineInterval(new_coeffs, new_bias)
        qint1, qint2 = self.eval(), other.eval()
        if qint2.min == qint2.max:
            return self * qint2.min
        vals = (qint1.min * qint2.min, qint1.min * qint2.max, qint1.max * qint2.min, qint1.max * qint2.max)
        _min, _max = min(vals), max(vals)
        _step = qint1.step * qint2.step
        return AffineInterval.new(QInterval(_min, _max, _step))

    def __or__(self, other):
        if not isinstance(other, AffineInterval):
            qint = self.eval()
            other_step = 2 ** get_lsb_loc(other)
            new_min = min(qint.min, other)
            new_max = max(qint.max, other)
            new_step = min(qint.step, other_step)
            return AffineInterval.new(QInterval(new_min, new_max, new_step))
        qint1, qint2 = self.eval(), other.eval()
        new_min = min(qint1.min, qint2.min)
        new_max = max(qint1.max, qint2.max)
        new_step = min(qint1.step, qint2.step)
        return AffineInterval.new(QInterval(new_min, new_max, new_step))

    def quantize_to(self, qint: QInterval):
        k = qint.min < 0
        step = qint.step
        f = iceil_log2(step)
        i = max(iceil_log2(qint.min), iceil_log2(qint.step + qint.max))
        _min, _max = -k * 2**i, 2**i - 2**-f
        qint0 = self.eval()
        if qint0.min >= _min and qint0.max <= _max:
            if qint0.step >= step:
                return self  # trivial
            _min = (qint0.min // step) * step
            _max = (qint0.max // step) * step
        return AffineInterval.new(qint)

    def relu(self, qint: QInterval):
        _qint = self.eval()
        if _qint.max <= 0:
            return AffineInterval({}, 0.0)
        elif _qint.min >= 0 and _qint.step <= qint.step and _qint.max <= qint.max:
            return self
        else:
            _qint = QInterval(0, min(_qint.max, qint.max), max(_qint.step, qint.step))
            return AffineInterval.new(qint)

    def __lshift__(self, other):
        if not isinstance(other, int):
            raise NotImplementedError('Shift amount must be an integer')
        new_coeffs = {k: v * (2**other) for k, v in self.coeffs.items()}
        new_bias = self.bias * (2**other)
        return AffineInterval(new_coeffs, new_bias)

    def __rshift__(self, other):
        return self.__lshift__(-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def bit_shuffle(self):
        qint = self.eval()
        step = qint.step
        f = iceil_log2(step)
        i = max(iceil_log2(qint.min), iceil_log2(qint.step + qint.max))
        k = qint.min < 0
        new_min, new_max = -k * 2**i, 2**i - 2**-f
        new_qint = QInterval(new_min, new_max, step)
        return AffineInterval.new(new_qint)


def affine_range_recomp(comb: CombLogic) -> CombLogic:
    "Range recomputation using affined intervals"
    als: list[AffineInterval] = []
    new_ops = []
    new_luts: dict[str, LookupTable] = {}
    for ii, op in enumerate(comb.ops):
        data = None
        match op.opcode:
            case -1:  # READ
                r = AffineInterval.new(op.qint)
            case 0:  # add
                r = als[op.id0] + (als[op.id1] << op.data)
            case 1:  # sub
                r = als[op.id0] - (als[op.id1] << op.data)
            case 2:  # relu
                r = als[op.id0].relu(op.qint)
            case -2:  # relu(-)
                r = (-als[op.id0]).relu(op.qint)
            case 3:  # quant
                r = als[op.id0].quantize_to(op.qint)
            case -3:  # quant(-)
                r = (-als[op.id0]).quantize_to(op.qint)
            case 4:  # cadd
                bias = op.data * op.qint.step
                r = als[op.id0] + bias
            case 5:  # const
                r = AffineInterval({}, op.data * op.qint.step)
            case 6 | -6:  # mux
                v0, v1 = als[op.id0], als[op.id1]
                if op.opcode == -6:
                    v1 = -v1
                shift = (op.data >> 32) & 0xFFFFFFFF
                shift = shift if shift < 0x80000000 else shift - 0x100000000
                r = v0 | (v1 << shift)
                r = r.quantize_to(op.qint)
            case 7:  # mul
                r = als[op.id0] * als[op.id1]
            case 8:  # lut
                assert comb.lookup_tables is not None
                table = comb.lookup_tables[op.data]
                qint = table.spec.out_qint
                r = AffineInterval.new(qint)
                qint_in_old = comb.ops[op.id0].qint
                qint_in_new = als[op.id0].eval()
                if qint_in_old != qint_in_new:
                    b0 = (qint_in_new.min - qint_in_old.min) // qint_in_old.step
                    b1 = (qint_in_new.max - qint_in_old.min) // qint_in_old.step
                    stride = qint_in_new.step // qint_in_old.step
                    new_table = table[b0:b1:stride]
                    new_luts[new_table.spec.hash] = new_table
                    data = list(new_luts.keys()).index(new_table.spec.hash)
                else:
                    new_luts[table.spec.hash] = table
                    data = list(new_luts.keys()).index(table.spec.hash)
            case 9 | -9:  # unary bitops
                v = als[op.id0]
                if op.opcode == -9:
                    v = -v
                subop = op.data
                if subop == 0:  # NOT
                    r = v.bit_shuffle()
                else:
                    r = AffineInterval.new(QInterval(0, 1, 1))
            case 10:  # binary bitops
                inv0, inv1 = (op.data >> 32) & 1, (op.data >> 33) & 1
                shift = ((int(op.data) & 0xFFFFFFFF) + (1 << 31)) % (1 << 32) - (1 << 31)
                v0, v1 = als[op.id0], als[op.id1]
                if inv0:
                    v0 = -v0
                if inv1:
                    v1 = -v1
                qint0, qint1 = v0.eval(), (v1 << shift).eval()
                k0, i0, f0 = minimal_kif(qint0)
                k1, i1, f1 = minimal_kif(qint1)
                _k, _i, _f = max(k0, k1), max(i0, i1), max(f0, f1)
                qint = QInterval(-_k * 2.0**_i, 2.0**_i - 2.0**-_f, 2.0**-_f)
                r = AffineInterval.new(qint)
                # print(i, op.qint, '->', qint)
            case _:
                raise NotImplementedError(f'Unsupported opcode: {op.opcode}')
        als.append(r)
        data = data if data is not None else op.data
        qint = als[ii].eval(canon=True)
        if qint.min != qint.max:
            new_op = Op(op.id0, op.id1, op.opcode, data, qint, op.latency, op.cost)
        else:
            # Collapsed into const
            data = qint.min / qint.step
            new_op = Op(-1, -1, 5, int(data), qint, op.latency, 0)
        print(ii, op.qint, '->', new_op.qint)
        new_ops.append(new_op)

    new_tables = tuple(new_luts.values())
    new_tables = new_tables if new_tables else None

    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        comb.out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        new_tables,
    )
