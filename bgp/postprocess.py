# -*- coding: utf-8 -*-

# @Time    : 2020/10/19 13:32
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import pandas as pd
import copy
import warnings
from collections.abc import Callable

import numpy as np
import sympy
from scipy.optimize import least_squares
from sympy import Expr
warnings.filterwarnings("ignore")


def top_n(loop, n=10, gen=-1, key="value", ascending=False):
    """return the top result of loop.
    PendingDeprecation.

    please use loop.top_n() directly.
    """
    data = loop.data_all
    data = pd.DataFrame(data)
    if gen == -1:
        gen = max(data["gen"])

    data = data[data["gen"] == gen]

    data = data.drop_duplicates(['expr'], keep="first")

    if key is not None:
        data[key] = data[key].str.replace("(", "")
        data[key] = data[key].str.replace(")", "")
        data[key] = data[key].str.replace(",", "")
        try:
            data[key] = data[key].astype(float)
        except ValueError:
            raise TypeError("check this key column can be translated into float")

        data = data.sort_values(by='value', ascending=ascending).iloc[:n, :]

    return data


def cla(pre_y, cl=True):
    pre_y = 1.0 / (1.0 + np.exp(-pre_y))
    if cl:
        pre_y[np.where(pre_y >= 0.5)] = 1
        pre_y[np.where(pre_y < 0.5)] = 0
    return pre_y


def format_input(expr01, x, y, init_c=None, terminals=None, c_terminals=None, np_maps=None, x_mark="x",
                 c_mark="c"):
    """
    Check and format_input for add_coef_fitting.

    Parameters
    ----------
    expr01:sympy.Expr
        expr for fitting.
    x: list of np.ndarray or np.ndarray
        real data with: [x1,x2,x3,...,x_n_feature] or x with shape (n_sample,n_feature).
    y: np.ndarray with shape (n_sample,)
        real data of target.
    init_c: list of float or float.
        default 1.
    terminals: list of sympy.Symbol
        placeholder for xi, with the same features in expr01.
    c_terminals:list of sympy.Symbol
        placeholder for ci, with the same coefficients/constants in expr01.
    np_maps: dict,default is None
        for self-definition.
        1. make your function with sympy.Function and arrange in in expr01.
        >>> x1, x2, x3, c1,c2,c3,c4 = sympy.symbols("x1,x2,x3,c1,c2,c3,c4")
        >>> Seg = sympy.Function("Seg")
        >>> expr01 = Seg(x1*x2,c1)
        2. write the numpy calculation method for this function.
        >>> def np_seg(x,c):
        >>>     res = -x
        >>>     res[res>-c]=0
        >>>     return res
        3. pass the np_maps parameters.
        >>> np_maps = {"Seg":np_seg}

        In total, when parse the expr01, find the numpy function in sequence by:
        (np_maps -> numpy's function -> system -> Error)
    x_mark:str
        mark for x
    c_mark:str
        mark for c

    Returns
    -------
    format_parameters:tuple
        (expr01, x, y, init_c, terminals, c_terminals, np_maps)
    """
    assert isinstance(expr01, Expr)
    assert isinstance(x, (np.ndarray, list))
    if isinstance(x, list):
        assert all([isinstance(i, np.ndarray) and i.ndim == 1 for i in x])
    else:
        x = [i for i in x.T]

    assert isinstance(y, np.ndarray) and y.ndim == 1

    if terminals is None:
        terminals = [i for i in list(expr01.free_symbols) if x_mark in i.name]
        terminals.sort(key=lambda x:x.name)
    else:
        assert set(terminals) & set(expr01.free_symbols) == set(terminals)
        assert len(terminals) == len(x)

    if c_terminals is None:

        c_terminals = [i for i in list(expr01.free_symbols) if c_mark in i.name]
        c_terminals.sort(key=lambda x:x.name)
    else:
        assert set(c_terminals) & set(expr01.free_symbols) == set(c_terminals)

    if init_c is None:
        init_c = [1.0] * len(c_terminals)
    elif isinstance(init_c, (int, float)):
        init_c = [init_c] * len(c_terminals)
    else:
        assert len(init_c) == len(c_terminals)

    if isinstance(np_maps, dict):
        for k, v in np_maps:
            assert isinstance(v, Callable)

    if np_maps is None:
        np_maps = {}

    return expr01, x, y, init_c, terminals, c_terminals, np_maps


def add_coef_fitting(expr01, x, y, init_c=None, terminals=None, c_terminals=None, np_maps=None,
                     classification=False,built_format_input=False):
    """
    Try calculate predict y by sympy expression with coefficients.
    if except error return expr itself.

    Parameters
    ----------
    expr01:sympy.Expr
        expr for fitting.
    x: list of np.ndarray or np.ndarray
        real data with: [x1,x2,x3,...,x_n_feature].
    y: np.ndarray with shape (n_sample,)
        real data of target.
    init_c: list of float or float,None
        default 1.
    terminals: list of sympy.Symbol,None
        placeholder for xi, with the same features in expr01.
    c_terminals:list of sympy.Symbol,None
        placeholder for ci, with the same coefficients/constants in expr01.
    np_maps: dict,default is None
        for self-definition.
        1. make your function with sympy.Function and arrange in in expr01.
        >>> x1, x2, x3, c1,c2,c3,c4 = sympy.symbols("x1,x2,x3,c1,c2,c3,c4")
        >>> Seg = sympy.Function("Seg")
        >>> expr01 = Seg(x1*x2,c1)
        2. write the numpy calculation method for this function.
        >>> def np_seg(x,c):
        >>>     res = -x
        >>>     res[res>-c]=0
        >>>     return res
        3. pass the np_maps parameters.
        >>> np_maps = {"Seg":np_seg}

        In total, when parse the expr01, find the numpy function in sequence by:
        (np_maps -> numpy's function -> system -> Error)

    classification:bool
        classfication or not, default False.

    built_format_input:bool
        use format_input function to check input parameters.
        Just used for temporary test or single case, due to format_input is repetitive.

    Returns
    -------
    pre_y:
        np.array or None
    expr01: Expr
        New expr.
    """

    if np_maps is None:
        np_maps = {}

    expr00 = copy.copy(expr01)

    try:

        func0 = sympy.utilities.lambdify(terminals+c_terminals, expr01, modules=[np_maps, "numpy"])

        def func(x_, p):
            """"""

            return func0(*x_,*p)

        def res(p, x_, y_):
            """"""
            return y_ - func(x_, p)

        def res2(p, x_, y_):
            """"""
            pre_y = func(x_, p)
            ress = y_ - cla(pre_y, cl=False)

            return ress

        if not classification:
            result = least_squares(res, x0=init_c, args=(x, y),
                                   # xtol=1e-4, ftol=1e-5, gtol=1e-5,
                                            # long
                                            jac='3-point', loss='linear')
        else:
            result = least_squares(res2, x0=init_c, args=(x, y),
                                   # xtol=1e-4, ftol=1e-5, gtol=1e-5,
                                            # long
                                            jac='3-point', loss='linear')
        cof = np.round(result.x,3)

        pre_y = func0(*x, *cof)
        if classification:
            pre_y = cla(pre_y, cl=True)

        for ai, choi in zip(c_terminals, cof):
            expr01 = expr01.xreplace({ai: choi})

        if isinstance(pre_y,float):
            pre_y = None
        elif all(np.isfinite(pre_y)):
            pass
        else:
            pre_y=None

    except (ValueError, KeyError, NameError, TypeError, ZeroDivisionError,IndexError):
    # except ImportError:

        expr01 = expr00
        pre_y = None

    return pre_y, expr01,

#
# x1, x2, x3, c1,c2,c3 = sympy.symbols("x1,x2,x3,c1,c2,c3")
# expr = x1 * x2 + x3 + c1
# expr = c3*x1 + c2* x2 + x3 + c1
#
# x = np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5], [4, 5, 6, 5, 4]]).T
# y = np.array([3, 3, 5, 6, 5])
#
# expr01, x, y, init_c, terminals, c_terminals, np_maps = format_input(expr, x, y)
#
# pre_y, expr02 = add_coef(expr01, x, y, init_c, terminals, c_terminals)





