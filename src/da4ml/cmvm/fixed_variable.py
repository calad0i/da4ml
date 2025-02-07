from math import ceil, log2


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Namer(metaclass=Singleton):
    def __init__(self):
        self._counters = {}
        self._scope = 'global'

    def set_scope(self, scope: str):
        self._scope = scope

    def tmp_scope(self):
        _counters = self._counters
        outer = self

        class _Ctx:
            def __enter__(self):
                outer._counters = {}

            def __exit__(self, *args) -> None:
                outer._counters = _counters

        return _Ctx()

    def __call__(self, name: str, scope: str | None = None) -> str:
        scope = self._scope if scope is None else scope
        counters = self._counters.setdefault(scope, {})
        if name not in counters:
            counters[name] = -1
        counters[name] += 1
        return f'{name}{counters[name]}'


class FixedVariable:
    def __init__(
        self,
        int_min: int,
        int_max: int,
        shift: int,
        symmetric: bool = False,
        _depth: int = 0,
        name: str = '',
        _factor: int = 1,
        _from: tuple['FixedVariable', 'FixedVariable'] | None = None,
        namer=Namer(),
    ):
        self.int_min = int_min
        self.int_max = int_max
        self.shift = shift
        self.symmetric = symmetric
        self._depth = _depth
        self.name = name
        self._factor = _factor
        self._from = _from
        self.namer = namer

        if self.int_min > self.int_max:
            raise ValueError('int_min must be less than or equal to int_max')

    @property
    def k(self) -> int:
        return int(self.min < 0)

    @property
    def b(self) -> int:
        return ceil(log2(max(self.int_max + 1, -self.int_min)))

    @property
    def i(self) -> int:
        return self.b - self.shift

    @property
    def min(self) -> float:
        return self.int_min * 2.0 ** (-self.shift)

    @property
    def max(self) -> float:
        return self.int_max * 2.0 ** (-self.shift)

    def __str__(self) -> str:
        s = '' if self.k else 'u'
        p = f'{s}fixed({self.b+self.k}, {self.i+self.k})'
        if self.int_min == self.int_max:
            return f'{p}({self.min})'
        return p

    def __add__(self, other: 'FixedVariable|float'):
        if other == 0:
            return self
        if not isinstance(other, FixedVariable):
            return self + self.from_const(other, self.namer)

        assert self.namer is other.namer, 'Namer must be the same'
        shift = max(self.shift, other.shift)
        _shift0, _shift1 = shift - self.shift, shift - other.shift
        int_min = (self.int_min << _shift0) + (other.int_min << _shift1)
        int_max = (self.int_max << _shift0) + (other.int_max << _shift1)

        return FixedVariable(
            int_min,
            int_max,
            shift,
            symmetric=False,
            _depth=max(self._depth, other._depth) + 1,
            _from=(self, other),
            _factor=1,
            name=self.namer('v'),
            namer=self.namer,
        )

    def __radd__(self, other: 'FixedVariable|float'):
        return self + other

    def __neg__(self):
        return FixedVariable(
            -self.int_max,
            -self.int_min,
            self.shift,
            symmetric=False,
            _depth=self._depth,
            _from=self._from,
            _factor=-self._factor,
            name=self.name,
            namer=self.namer,
        )

    def __sub__(self, other: 'FixedVariable'):
        return self + (-other)

    def __lshift__(self, other: int):
        return FixedVariable(
            self.int_min,
            self.int_max,
            self.shift - other,
            False,
            self._depth,
            _from=self._from,
            _factor=self._factor * 2**other,
            name=self.name,
            namer=self.namer,
        )

    def __rshift__(self, other: int):
        return self << -other

    def __mul__(self, other: float):
        if other == 1:
            return self
        if other == -1:
            return -self
        if other == 0:
            return self.from_const(0, self.namer)
        assert log2(abs(other)) % 1 == 0
        sign = -1 if other < 0 else 1
        shift = int(log2(abs(other)))
        return self << shift if sign == 1 else -self << shift

    def __rmul__(self, other: float):
        return self * other

    def __repr__(self) -> str:
        if self._factor == 1:
            return self.__str__()
        return f'({self._factor}) {self.__str__()}'

    @classmethod
    def from_nb_precision(cls, p, name: str | None = None, namer=Namer()):
        name = Namer()('inp') if name is None else name
        return cls(p.int_min, p.int_max, p.shift, p.symmetric, p._depth, name=name, namer=namer)

    @classmethod
    def from_const(cls, value: float, namer=Namer()):
        if value == 0:
            return cls(0, 0, 0, False, 0, '0', namer=namer)
        _low, _high = -32, 32
        while _high - _low > 1:
            _mid = (_high + _low) // 2
            _value = value * (2.0**_mid)
            if _value == int(_value):
                _high = _mid
            else:
                _low = _mid
        _value = value * (2.0**_high)
        shift = int(_high)
        int_min = int_max = int(_value)
        return cls(
            int_max,
            int_min,
            shift,
            symmetric=False,
            _depth=0,
            _from=None,
            _factor=1,
            name=str(value),
            namer=namer,
        )
