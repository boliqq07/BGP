#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

"""
Notes:
    These are some of parts coped from sympy.
"""

from __future__ import division

import numbers

import numpy as np
import numpy.core.numeric as numeric
import sympy
from numpy.linalg import matrix_rank
from sklearn.utils import check_array
from sympy import Add, Mul, Pow, Tuple, sympify
from sympy.core.compatibility import reduce, Iterable
from sympy.physics.units import Dimension
from sympy.physics.units.quantities import Quantity


def dim_map():
    """expr to dim function """

    def my_comp(dim):

        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            else:
                n = dim.shape[0]
                dc = dim[0].copy()
                return dc.__pow__(n)
        else:
            return dim

    def my_quot(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            elif dim.shape[0] == 2:
                if dim.anyisnan():
                    return dnan
                return dless
            else:
                return dim
        else:
            return dim

    def my_diff(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            elif dim.shape[0] == 2:
                return dim[0].copy()

            else:
                return dim
        else:
            return dim

    def my_conv(dim):
        if isinstance(dim, Dim):
            return dim
        else:
            return dim

    def my_flat(dim):
        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            else:
                return dim[0].copy()
        else:
            return dim

    def my_abs(dim):
        if isinstance(dim, Dim):
            return dim
        else:
            return dim

    def my_sqrt(dim):

        return dim.__pow__(0.5)

    my_self = my_abs

    def my_exp(dim):

        if isinstance(dim, Dim):
            if dim == dless:
                return dless.get_n(dim)
            else:
                return dnan.get_n(dim)
        else:
            return dim

    def my_der(dim1, dim2):

        if isinstance(dim1, Dim):
            return dim1.__div__(dim2)
        elif isinstance(dim2, Dim):
            return dim2.__rdiv__(dim1)

    my_log = my_ln = my_cos = my_sin = my_exp

    my_funcs = {"Abs": my_abs, "exp": my_exp, "ln": my_ln, 'cos': my_cos, 'sin': my_sin, "log": my_log, "Der": my_der,
                'sqrt': my_sqrt, "MAdd": my_flat, "MMul": my_comp, "MSub": my_diff,
                "MDiv": my_quot, "Self": my_self, "Conv": my_conv}
    return my_funcs


class Dim(numeric.ndarray):
    """Redefine the Dimension of sympy, the default dimension SI system with 7 number.\n
    1.can be constructed by list of number.\n
    2.can be translated from a sympy.physics.unit. \n

    Examples::

        from sympy.physics.units import N
        scale,dim = Dim.convert_to_Dim(N)

    Examples::

        dim=[1,0,1,0,1,0,0]
        dim = Dim(dim)

    Notes:
        self.unit = [str(i) for i in SI._base_units]\n
        self.unit_map = {'meter': "m", 'kilogram': "kg", 'second': "s",
        'ampere': "A", 'mole': "mol", 'candela': "cd", 'kelvin': "K"}\n
        self.dim = ['length', 'mass', 'time', 'current', 'amount_of_substance',
        'luminous_intensity', 'temperature']
    """

    def __new__(cls, data):

        assert isinstance(data, (numeric.ndarray, (list, tuple)))
        dtype = np.float16

        arr = numeric.array(data, dtype=dtype, copy=True)

        shape = arr.shape

        ret = numeric.ndarray.__new__(cls, shape, dtype=np.float16,
                                      buffer=arr,
                                      order='c')
        return ret

    def __eq__(self, other):
        if isinstance(other, Dim):
            se = self.copy()
            ot = other.copy()
            if se.ndim == 2:
                se = se[0]
            if ot.ndim == 2:
                ot = ot[0]
            return all(np.equal(se, ot))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):

        if isinstance(other, Dim) and self != other:
            return dnan.get_n(self.get_n(other))  # ?
        elif isinstance(other, Dim) and self == other:
            return self.get_n(other)

        else:
            return self

    def __sub__(self, other):
        return self.__add__(other)

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return Dim(np.array(self) * other)
        else:
            return dnan  #

    def __mul__(self, other):
        if isinstance(other, Dim):
            return Dim(np.array(self) + np.array(other))
        else:
            return self

    def __div__(self, other):

        if isinstance(other, Dim):
            return Dim(np.array(self) - np.array(other))
        else:
            return self

    def __rdiv__(self, other):

        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self.__pow__(-1)
        elif isinstance(other, Dim):
            return Dim(np.array(other) - np.array(self))
        else:
            return dnan  #

    def __abs__(self):
        return self

    def __rpow__(self, other):
        return dnan  #

    def __neg__(self):
        return self

    def __pos__(self):
        return self

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__

    # @property
    def allisnan(self):
        return np.all(np.isnan(self))

    # @property
    def anyisnan(self):
        return np.any(np.isnan(self))

    # @property
    def isfloat(self):
        return np.any(np.modf(self)[0])

    # @property
    def isinteger(self):
        return not np.any(np.modf(self)[0])

    def is_same_base(self, others):
        se = self.copy()
        if others.ndim == 2:
            others = others[0]
        if se.ndim == 2:
            se = se[0]

        if isinstance(others, Dim):
            npself = np.array(se)
            npothers = np.array(others)
            x1 = np.linalg.norm(npself)
            x2 = np.linalg.norm(npothers)

            if others ** x1 == se ** x2:
                return True
            else:
                return False
        else:
            return False

    def get_n(self, others):
        se = self.copy()

        if others.ndim == 2:
            n = others.shape[0]
        else:
            n = 1

        if self.ndim == 2:
            m = self.shape[0]
        else:
            m = 1

        if m > 1 and n > 1:
            return dnan  #
        elif n == 1:
            return se
        elif m == 1 and n > 1:
            return Dim(np.array([se] * n))
        else:
            return dnan  #

    @staticmethod
    def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
        """depend on sympy 1.5-1.6!!!"""
        from sympy import Matrix

        dimension_system = unit_system.get_dimension_system()

        expr_dim = Dimension(unit_system.get_dimensional_expr(expr))
        dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
        target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]
        canon_dim_units = [i for x in target_dims for i in
                           dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]
        canon_expr_units = {i for i in dim_dependencies}

        if not canon_expr_units.issubset(set(canon_dim_units)):
            raise TypeError("There is an invalid character in '%s'" % expr,
                            "the expr must be sympy.physics.unit or number")

        seen = set([])
        canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]

        camat = Matrix(
            [[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims]
             for j in canon_dim_units])
        exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

        res_exponents = camat.solve_least_squares(exprmat, method=None)

        return res_exponents, canon_dim_units

    @classmethod
    def convert_to(cls, expr, target_units=None, unit_system="SI"):
        """depend on sympy 1.5-1.6!!!"""
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_unit_system(unit_system)
        if not target_units:
            target_units = unit_system._base_units

        if not isinstance(target_units, (Iterable, Tuple)):
            target_units = [target_units]

        if isinstance(expr, Add):
            raise TypeError("can not be add")

        expr = sympify(expr)

        if not isinstance(expr, Quantity) and expr.has(Quantity):
            expr = expr.replace(lambda x: isinstance(x, Quantity), lambda x: x.convert_to(target_units, unit_system))

        def get_total_scale_factor(expr0):
            if isinstance(expr0, Mul):
                return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr0.args])
            elif isinstance(expr0, Pow):
                return get_total_scale_factor(expr0.base) ** expr0.exp
            elif isinstance(expr0, Quantity):
                return unit_system.get_quantity_scale_factor(expr0)
            return expr0

        depmat, canon_dim_units = cls._get_conversion_matrix_for_expr(expr, target_units, unit_system)
        if depmat is None:
            raise TypeError("There is an invalid character in '%s'" % expr,
                            "the expr must be sympy.physics.unit or number")

        expr_scale_factor = get_total_scale_factor(expr)
        dim_dict = {}
        for u, p in zip(target_units, depmat):
            expr_scale_factor /= get_total_scale_factor(u) ** p
            dim_dict["%s" % u] = p

        d = cls(np.array(list(dim_dict.values())))
        d.dim = canon_dim_units
        d.unit = target_units
        return expr_scale_factor, d

    @classmethod
    def convert_to_Dim(cls, u, target_units=None, unit_system="SI"):
        """
        depend on sympy 1.5-1.6!!!

        Parameters
        ----------
        u: sympy.physics.unit, Expr of sympy.physics.unit
            unit.
        target_units: None, list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"
        """
        if isinstance(u, Dim):
            return 1, u
        else:
            expr_scale_factor, d = cls.convert_to(u, target_units=target_units, unit_system=unit_system)
            return expr_scale_factor, d

    @classmethod
    def convert_xi(cls, xi, ui, target_units=None, unit_system="SI"):
        """
        depend on sympy 1.5-1.6!!!
        Quick method. translate xi and ui to standard system.

        Parameters
        ----------
        xi: np.ndarray
            xi
        ui: sympy.physics.unit or Expr of sympy.physics.unit
            unit
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        xi: np.ndarray

        expr: Expr
        """
        expr_scale_factor, d = cls.convert_to_Dim(ui, target_units=target_units, unit_system=unit_system)
        xi = expr_scale_factor * xi
        if isinstance(xi, np.ndarray):
            xi = xi.astype(np.float32)
        return xi, d

    @classmethod
    def convert_x(cls, x, u, target_units=None, unit_system="SI"):
        """
         depend on sympy 1.5-1.6!!!
         Quick method. translate x and u to standard system.
         
         Parameters
         ----------
         x: np.ndarray or list of ndarray,list of float,list of int
             x
         u: list of sympy.physics.unit or Expr of sympy.physics.unit
             units
         target_units: None or list of sympy.physics.unit
             if None, the target_units is 7 SI units
         unit_system: str
             default is unit_system="SI"

         Returns
         -------
         x: np.ndarray
         expr: Expr
         
         """
        if isinstance(x, list):
            pass
        elif isinstance(x, np.ndarray):
            x = x.T
        else:
            raise TypeError("values must be list or np.array")
        x_and_d = [cls.convert_xi(xi, x_ui, target_units, unit_system) for xi, x_ui in zip(x, u)]
        x = np.array([xi for xi, d in x_and_d]).T
        x_dim = [d for xi, d in x_and_d]
        return x, x_dim

    @classmethod
    def inverse_convert(cls, dim, scale=1, target_units=None, unit_system="SI"):
        """
        depend on sympy 1.5-1.6!!!
        Quick method. Translate ui to other unit.

        Parameters
        ----------
        dim: Dim
            dim
        scale: float
            scale generated before.
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        scale:float

        expr: Expr
        """
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_unit_system(unit_system)
        if not target_units:
            target_units = unit_system._base_units

        if not isinstance(target_units, (Iterable, Tuple)):
            target_units = [target_units]

        def get_total_scale_factor(expr):
            if isinstance(expr, Mul):
                return reduce(lambda x, y: x * y, [get_total_scale_factor(i) for i in expr.args])
            elif isinstance(expr, Pow):
                return get_total_scale_factor(expr.base) ** expr.exp
            elif isinstance(expr, Quantity):
                return unit_system.get_quantity_scale_factor(expr)
            return expr

        sc = scale * Mul.fromiter(1 / get_total_scale_factor(u) ** p for u, p in zip(unit_system._base_units, dim))
        sc = Mul.fromiter(1 / get_total_scale_factor(u) for u in target_units) / sc

        tar = Mul.fromiter((1 / get_total_scale_factor(u) * u) for u in target_units)
        bas = Mul.fromiter((get_total_scale_factor(u)) ** p for u, p in zip(unit_system._base_units, dim))
        return sc, scale * tar * bas

    @classmethod
    def inverse_convert_xi(cls, xi, dim, scale=1, target_units=None, unit_system="SI"):
        """
        depend on sympy 1.5-1.6!!!
        Quick method. Translate xi, dim to other unit.

        Parameters
        ----------
        xi:np.ndarray
            xi
        dim: Dim
            dim
        scale: float
            if xi is have been scaled, the scale is 1.
        target_units: None or list of sympy.physics.unit
            if None, the target_units is 7 SI units
        unit_system: str
            default is unit_system="SI"

        Returns
        -------
        scale: float

        expr: Expr
        """
        expr_scale_factor, d = cls.inverse_convert(dim, scale=scale, target_units=target_units, unit_system=unit_system)
        xi = expr_scale_factor * xi
        if isinstance(xi, np.ndarray):
            xi = xi.astype(np.float32)
        return expr_scale_factor * xi, d


def check_dimension(x, y=None):
    """
    check the consistency of dimension.
    
    Parameters
    ----------
    x: container
        dim of x
    y: Dim
        dim of y

    Returns
    -------
    bool
    """
    if y is not None:
        x.append(y)
    x = np.array(x).T
    x = check_array(x, ensure_2d=True)
    assert isinstance(x, np.ndarray)
    x = x.astype(np.float64)
    det = matrix_rank(x)
    che = []
    for i in range(x.shape[1]):
        x_new = np.delete(x, i, 1)
        det2 = matrix_rank(x_new)
        che.append(det - det2)
    sum(che)

    if sum(che) == 0:
        return True
    else:
        return False


dnan = Dim(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
dless = Dim(np.array([0, 0, 0, 0, 0, 0, 0]))

if __name__ == "__main__":
    a = [Dim([1, 2, 3, 4, 5, 6, 7])] * 100
    b = [Dim([2, 2, 3, 4, 5, 6, 7])] * 100

    dl = dless
    dn = dnan

    c = Dim([1, 2, 3, 4, 5, 6, 7])

    t = zip(a, b)


    def func(k):
        ss = k[1] * k[0]
        return ss
