from numba import njit

from .nb_fixed_precision import NBFixedPrecision


def py_resource_scorer(p1, p2) -> float:
    b0, b1 = p1.b, p2.b
    f1, f2 = p1.shift, p2.shift
    pl0, pl1 = -f1, -f2
    ph0, ph1 = pl0 + b0, pl1 + b1
    pl, pL = (pl0, pl1) if pl0 < pl1 else (pl1, pl0)
    ph, pH = (ph0, ph1) if ph0 < ph1 else (ph1, ph0)

    return ph - pL


def py_latency_scorer(p1, p2) -> float:
    return -float(abs(p1._depth - p2._depth))


def py_scorer(p1, p2, dshift: int = 0, dsign: int = 0) -> float:
    p2 = p2 << dshift
    rs = py_resource_scorer(p1, p2)
    ls = py_latency_scorer(p1, p2)
    score = rs + ls
    return score


@njit
def resource_scorer(p1: NBFixedPrecision, p2: NBFixedPrecision) -> float:
    b0, b1 = p1.b, p2.b
    f1, f2 = p1.shift, p2.shift
    pl0, pl1 = -f1, -f2
    ph0, ph1 = pl0 + b0, pl1 + b1
    pl, pL = (pl0, pl1) if pl0 < pl1 else (pl1, pl0)
    ph, pH = (ph0, ph1) if ph0 < ph1 else (ph1, ph0)

    return ph - pL
    n_full = max(0, ph - pL)

    return n_full**2 / (pH - pl)


@njit
def latency_scorer(p1: NBFixedPrecision, p2: NBFixedPrecision) -> float:
    return -float(abs(p1._depth - p2._depth))


@njit
def scorer(p1: NBFixedPrecision, p2: NBFixedPrecision, dshift: int, dsign: int) -> float:
    p2 = p2 << dshift
    rs = resource_scorer(p1, p2)
    ls = latency_scorer(p1, p2)
    score = rs + ls
    return score
