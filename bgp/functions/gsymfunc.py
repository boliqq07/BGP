import numbers

import numpy as np
import sympy
from sympy import ImmutableDenseNDimArray
from sympy.tensor.array.arrayop import Flatten


class NewArray(ImmutableDenseNDimArray):
    def __new__(cls, iterable, shape=None, **kwargs):
        return cls._new(iterable, shape, **kwargs)

    def __add__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        if not isinstance(other, NewArray):
            result_list = [i + other for i in Flatten(self)]
            return type(self)(result_list, self.shape)

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i + j for i, j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __sub__(self, other):

        if not isinstance(other, NewArray):
            result_list = [i - other for i in Flatten(self)]
            return type(self)(result_list, self.shape)

        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        result_list = [i - j for i, j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __rsub__(self, other):

        if not isinstance(other, NewArray):
            result_list = [other - i for i in Flatten(self)]
            return type(self)(result_list, self.shape)
        else:
            raise NotImplementedError("array shape mismatch")

    def __mul__(self, other):

        if isinstance(other, (np.ndarray, NewArray)):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            else:
                result_list = [i * j for i, j in zip(Flatten(self), Flatten(other))]
                return type(self)(result_list, self.shape)
        else:
            result_list = [i * other for i in Flatten(self)]
            return type(self)(result_list, self.shape)

    def __div__(self, other):

        if isinstance(other, (np.ndarray, NewArray)):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            else:
                result_list = [i / j for i, j in zip(Flatten(self), Flatten(other))]
                return type(self)(result_list, self.shape)
        else:
            result_list = [i / other for i in Flatten(self)]
            return type(self)(result_list, self.shape)

    def __rdiv__(self, other):

        if isinstance(other, (np.ndarray, NewArray)):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            else:
                result_list = [i / j for i, j in zip(Flatten(self), Flatten(other))]
                return type(self)(result_list, self.shape)
        else:
            result_list = [other / i for i in Flatten(self)]
            return type(self)(result_list, self.shape)

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            result_list = [i ** other for i in Flatten(self)]
            return type(self)(result_list, self.shape)
        else:
            raise ValueError("array shape mismatch")

    def __abs__(self):
        result_list = [sympy.Abs(i) for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rpow__(self, other):
        raise ValueError("array shape mismatch")

    def __pos__(self):
        return self

    __radd__ = __add__
    __truediv__ = __div__  # ?
    __rtruediv__ = __rdiv__  # ?

    def as_coefficient(self, _):
        return None


def gsym_map():
    """user's sympy.expr to np.ndarray function"""

    def Flat(x):
        if isinstance(x, (np.ndarray, NewArray)):
            return sum(x)
        else:
            return x

    def Comp(x):
        if isinstance(x, (np.ndarray, NewArray)):
            return np.prod(x)
        else:
            return x

    def Diff(x):
        if isinstance(x, (np.ndarray, NewArray)):
            if x.shape[0] == 2:
                return x[0] - x[1]
            else:
                return x
        else:
            return x

    def Quot(x):
        if isinstance(x, (np.ndarray, NewArray)):
            if x.shape[0] == 2:
                return x[0] / x[1]
            else:
                return x
        else:
            return x

    def Conv(x):
        if isinstance(x, (np.ndarray, NewArray)):
            if x.shape[0] == 2:
                return NewArray((x[1], x[0]), x.shape)
            else:
                return x
        else:
            return x

    def my_abs(x):
        if isinstance(x, (np.ndarray, NewArray)):
            result_list = [sympy.Abs(i) for i in Flatten(x)]
            return NewArray(result_list, x.shape)
        else:
            return sympy.Abs(x)

    def my_sqrt(x):
        return x.__pow__(0.5)

    def my_exp(x):

        if isinstance(x, (np.ndarray, NewArray)):
            result_list = [sympy.exp(i) for i in Flatten(x)]
            return NewArray(result_list, x.shape)
        else:
            return sympy.exp(x)

    def my_ln(x):

        if isinstance(x, (np.ndarray, NewArray)):
            result_list = [sympy.ln(i) for i in Flatten(x)]
            return NewArray(result_list, x.shape)
        else:
            return sympy.ln(x)

    my_log = my_ln

    def my_sin(x):

        if isinstance(x, (np.ndarray, NewArray)):
            result_list = [sympy.sin(i) for i in Flatten(x)]
            return NewArray(result_list, x.shape)
        else:
            return sympy.sin(x)

    def my_cos(x):

        if isinstance(x, (np.ndarray, NewArray)):
            result_list = [sympy.cos(i) for i in Flatten(x)]
            return NewArray(result_list, x.shape)
        else:
            return sympy.cos(x)

    def my_der(self, other):
        if isinstance(self, (np.ndarray, NewArray)):
            if isinstance(other, (np.ndarray, NewArray)):
                if self.shape != other.shape:
                    raise ValueError("array shape mismatch")
                else:
                    result_list = [sympy.diff(i, j, evaluate=False) for i, j in zip(Flatten(self), Flatten(other))]
                    return type(self)(result_list, self.shape)
            else:
                result_list = [sympy.diff(i, other, evaluate=False) for i in Flatten(self)]
                return type(self)(result_list, self.shape)
        else:
            if isinstance(other, (np.ndarray, NewArray)):
                result_list = [sympy.diff(self, i, evaluate=False) for i in Flatten(other)]
                return type(self)(result_list, other.shape)
            else:
                return sympy.diff(self, other, evaluate=False)

    return {"MAdd": Flat, "MMul": Comp, "MSub": Diff, "MDiv": Quot, "Conv": Conv,
            "Self": lambda x_: x_,
            "Abs": my_abs, "exp": my_exp, "ln": my_ln, 'cos': my_cos, 'sin': my_sin, "log": my_log, "Der": my_der,
            'sqrt': my_sqrt,
            }
