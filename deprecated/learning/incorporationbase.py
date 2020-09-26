# -*- coding: utf-8 -*-

# @Time   : 2019/3/20 14:42
# @Author : Administrator
# @Project : feature-optimization
# @FileName: incorporationbase.py
# @Software: PyCharm

import numbers
import warnings

import numpy as np
import sympy
from sympy import sympify, Mul, Pow, Add, symbols
from sympy.physics.units import Dimension, Quantity

warnings.filterwarnings("ignore")

"""
this is a description
"""


######################
# dimension part
######################


def dim_func():
    """dim function """
    from sympy import exp, log, sin, cos

    def my_abs(dim):

        if isinstance(dim, Dim):
            return dim
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return abs(dim)
        else:
            raise TypeError

    def my_exp(dim):

        if isinstance(dim, Dim):
            if dim.name == 1:
                return dim
            else:
                raise TypeError
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return exp(dim)
        else:
            raise TypeError

    def my_sqrt(dim):

        return dim.__pow__(0.5)

    def my_log(dim):

        if isinstance(dim, Dim):
            if dim.name == 1:
                return dim
            else:
                raise TypeError
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return log(dim)
        else:
            raise TypeError

    def my_sin(dim):

        if isinstance(dim, Dim):
            if dim.name == 1:
                return dim
            else:
                raise TypeError
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return sin(dim)
        else:
            raise TypeError

    def my_cos(dim):

        if isinstance(dim, Dim):
            if dim.name == 1:
                return dim
            else:
                raise TypeError
        elif isinstance(dim, (numbers.Real, sympy.Rational, sympy.Float)):
            return cos(dim)
        else:
            raise TypeError

    my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin, 'sqrt': my_sqrt}
    return my_funcs


class Dim(Dimension):
    """re define the Dimension of sympy """

    def __init__(self, p):
        self.__dict__ = p.__dict__.copy()

    def __add__(self, other):

        if isinstance(other, Dim) and self != other:
            raise TypeError
        elif isinstance(other, Dim) and self == other:
            return self
        elif not isinstance(other, Dim):
            if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
                return self
            else:
                raise TypeError
        else:
            raise TypeError

    def __sub__(self, other):
        return self + other

    def __pow__(self, other):
        return self._eval_power(other)

    def _eval_power(self, other):
        other = sympify(other)
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return Dim(Dimension(self.name ** other))
        elif isinstance(other, Dim):
            if other.name == 1:
                return self
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, Dim):
            return Dim(Dimension(self.name * other.name))

        elif isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self

        else:
            raise TypeError

    def __div__(self, other):

        if isinstance(other, Dim):
            if self.name == other.name:
                return Dim(Dimension(1))
            else:
                return Dim(Dimension(self.name / other.name))
        elif isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self
        else:
            raise TypeError

    def __rdiv__(self, other):
        # return other*spath._eval_power(-1)
        if isinstance(other, (numbers.Real, sympy.Rational, sympy.Float)):
            return self.__pow__(-1)
        else:
            raise TypeError

    def __abs__(self):
        return self

    def __rpow__(self, other):
        raise TypeError

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


######################
# departure part
######################


def spilt_couple(x_fea, x, compount_index=None, pro_index=None, n=2):
    """
    :param x_fea: x_fea
    :param x: np.ndarray
    :param compount_index: default is [0...n-1]
    :param pro_index: default is [n-1...shape]
    :param n:  kinds of ingredients

    :rtype: list like [c1,c2,p1,p2]
    """
    if compount_index is None:
        compount_index = list(range(n))
    if pro_index is None:
        pro_index = [n, None]
    if pro_index[1] is None:
        pro_index[1] = x.shape[1]

    index = spilt_index(compount_index, pro_index, n=n)
    x_fea_i0 = []
    for indexi in index:
        x_fea_ii = []
        for item in x_fea:
            x_fea_ii.append([item[_] for _ in indexi])
        x_fea_i0.append(x_fea_ii)

    x_i = [x[:, _] for _ in index]
    return x_fea_i0, x_i


def spilt_index(compount_index, pro_index, n=2):
    """slice and  together"""
    compount_index = np.array(compount_index)
    pro_index1 = np.arange(pro_index[0], pro_index[1], n)
    spilt_index0 = []
    for i in pro_index1:
        compount_index1 = compount_index
        j = i
        while j < i + n:
            compount_index1 = np.append(compount_index1, j)
            j += 1
        spilt_index0.append(compount_index1)
    return spilt_index0


def collect_factor_and_dimension(expr):
    """just a copy from deap.
    please read https://deap.readthedocs.io/en/master/index.html
    """
    if isinstance(expr, Quantity):
        return expr.scale_factor, expr.dimension
    elif isinstance(expr, Mul):
        factor = 1
        dimension = 1
        for arg in expr.args:
            arg_factor, arg_dim = collect_factor_and_dimension(arg)
            factor *= arg_factor
            dimension *= arg_dim
        return factor, dimension
    elif isinstance(expr, Pow):
        factor, dim = collect_factor_and_dimension(expr.base)
        return factor ** expr.exp, dim ** expr.exp
    elif isinstance(expr, Add):
        raise NotImplementedError
    else:
        return 1, 1


def fea_compile(unit_list0, symbols_name=None, p=True, pre=True):
    """if symbols_name is None, default x_name is xi"""
    if symbols_name is None:
        symbols_name = "x"
    if len(symbols_name) == 1:
        symbols_name = ['{}{}'.format(symbols_name, i) for i in str(symbols_name)]
        return fea_compile_with_name(unit_list0, name=symbols_name, p=p, pre=pre)
    elif len(symbols_name) == len(unit_list0):
        try:
            return fea_compile_with_name(unit_list0, name=symbols_name, p=p, pre=pre)
        except TypeError:
            raise TypeError("each symbols_name should with simple form and without space")
    else:
        raise IndexError("unit and x_name should have same size or just a single prefix_with_upper like 'x'.")


def fea_compile_with_name(unit_list0, name=None, p=True, pre=True, ):
    """
    :param unit_list0: as x_name
    :param name: list of feature x_name
    :param p: print or false ,default is True
    :param pre: take unit and number transformation into account or not, if true ,return
    the sym_unify_list0(like 10^-6*m)  ,else the sym_list(like m)
    :return:
            sym_unify_list0:list of sym with prefix_with_upper
            sym_list0:list of sym
            dim_list0:list of dim
            unit_list0:list of unit
    """

    def replace_unit(unit_list1):
        unit_list2 = []
        for _ in unit_list1:
            if _ == 1:
                _ = Quantity('Uint_one', 1, 1)
            unit_list2.append(_)
        return unit_list2

    unit_list0 = replace_unit(unit_list0)
    sym_unify_list0 = []
    sym_list0 = []
    dim_list0 = []
    for i, j in enumerate(unit_list0):
        if isinstance(j, (Quantity, Mul, Add, Pow)):
            sn = collect_factor_and_dimension(j)[0] * symbols(name[i])
            un = collect_factor_and_dimension(j)[1]
            sn2 = un.get_dimensional_dependencies()
            if 'mass' in sn2:
                sn = sn / (1000 ** sn2['mass'])
            sym_unify_list0.append(sn)
            sym_list0.append(symbols(name[i]))
            un = Dim(un)
            dim_list0.append(un)
        elif isinstance(j, Dim):
            sn = symbols(name[i])
            un = j
            sym_unify_list0.append(sn)
            sym_list0.append(sn)
            dim_list0.append(un)
    sym_list_str = [str(i) for i in sym_list0]
    if p:
        print(sym_unify_list0, sym_list0, dim_list0, unit_list0, sym_list_str)
    if pre is True:
        return sym_unify_list0, sym_list0, dim_list0, unit_list0
    else:
        return sym_list0, sym_list0, dim_list0, unit_list0


def fea_compile_y(y_unit0, p=True):
    if y_unit0 == 1:
        y_unit0 = Quantity('Uint_one', 1, 1)
    sn = collect_factor_and_dimension(y_unit0)[0] * symbols('y%s' % 1)
    un = collect_factor_and_dimension(y_unit0)[1]
    sn2 = un.get_dimensional_dependencies()
    if 'mass' in sn2:
        sn = sn / (1000 ** sn2['mass'])
    un = Dim(un)
    sym = symbols('y%s' % 1)
    sym_str = str(sym)
    if p:
        print(sn, sym, un, y_unit0, sym_str)
    else:
        pass
    return sn, sym, un, y_unit0
