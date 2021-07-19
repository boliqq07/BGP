#!/usr/bin/python
# coding:utf-8

# @author: wangchangxin
# @contact: 986798607@qq.com
# @software: PyCharm
# @file: scores.py
# @License: GNU Lesser General Public License v3.0
"""
Notes:
    score method.
"""

import copy
import itertools
import warnings

import numpy as np
# import sympy
from sklearn import metrics
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils import check_array
from sympy import utilities, Expr, diff
from sympy.core import Function

from bgp.calculation.coefficient import try_add_coef, cla
from bgp.calculation.translate import compile_context
from bgp.functions.dimfunc import dim_map, dless, dnan, Dim
from bgp.functions.gsymfunc import NewArray


def calculate_y(expr01, x, y, terminals, add_coef=True, x_test=None, y_test=None,
                filter_warning=True, inter_add=True, inner_add=False, vector_add=False, out_add=False, flat_add=False,
                np_maps=None, classification=False):
    if filter_warning:
        warnings.filterwarnings("ignore")
    try:
        if add_coef:
            pre_y, expr01 = try_add_coef(expr01, x, y, terminals,
                                         filter_warning=filter_warning,
                                         inter_add=inter_add, inner_add=inner_add,
                                         vector_add=vector_add, out_add=out_add, flat_add=flat_add,
                                         np_maps=np_maps, classification=classification)
        else:
            func0 = utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
            pre_y = func0(*x)
            if classification:
                pre_y = cla(pre_y)

        if x_test is not None and y_test is not None:
            func0 = utilities.lambdify(terminals, expr01, modules=[np_maps, "numpy"])
            pre_y = func0(*x_test)
            if classification:
                pre_y = cla(pre_y)

            pre_y = pre_y.ravel()
            assert y_test.shape == pre_y.shape
            pre_y = check_array(pre_y, ensure_2d=False)
        else:
            pre_y = pre_y.ravel()
            assert y.shape == pre_y.shape
            pre_y = check_array(pre_y, ensure_2d=False)

    except (DataConversionWarning, TypeError,AssertionError, ValueError, AttributeError, KeyError, ZeroDivisionError):
        pre_y = None

    return pre_y, expr01


def calculate_y_unpack(expr01, x, terminals, classification=False):
    try:
        func0 = utilities.lambdify(terminals, expr01)
        pre_y = func0(*x)
        if classification:
            pre_y = cla(pre_y)
        pre_y = pre_y.ravel()
        pre_y = check_array(pre_y, ensure_2d=False)

    except (DataConversionWarning, AssertionError, ValueError, AttributeError, KeyError, ZeroDivisionError):
        pre_y = None
    return pre_y


def uniform_score(score_pen=1):
    """return the worse score"""
    if score_pen >= 0:
        return -np.inf
    elif score_pen <= 0:
        return np.inf
    elif score_pen == 0:
        return 0
    else:
        return score_pen


def calculate_score(expr01, x, y, terminals, scoring=None, score_pen=(1,),
                    add_coef=True, filter_warning=True, inter_add=True,
                    inner_add=False, vector_add=False, out_add=False, flat_add=False, np_maps=None,
                    classification=False, score_object="y", details=False):
    """

    Parameters
    ----------
    vector_add
    expr01: Expr
        sympy expression.
    x: list of np.ndarray
        list of xi
    y: np.ndarray
        y value
    terminals: list of sympy.Symbol
        features and constants
    scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
        See Also sklearn.metrics
    score_pen: tuple of  1 or -1
        1 : best is positive, worse -np.inf
        -1 : best is negative, worse np.inf
        0 : best is positive , worse 0
    add_coef: bool
        bool
    filter_warning: bool
        bool
    inter_add: bool
        bool
    inner_add: bool
        bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    score:float
        score
    expr01: Expr
        New expr.
    pre_y: np.ndarray or float
        np.array or None
    """
    if filter_warning:
        warnings.filterwarnings("ignore")
    if not scoring:
        scoring = [r2_score, ]
    if not isinstance(scoring, (tuple, list)):
        scoring = [scoring, ]
    if isinstance(score_pen, int):
        score_pen = [score_pen, ]

    assert len(scoring) == len(score_pen), "the scoring and score_pen with same size"

    pre_y, expr01 = calculate_y(expr01, x, y, terminals, add_coef=add_coef,
                                filter_warning=filter_warning, inter_add=inter_add, inner_add=inner_add,
                                vector_add=vector_add, out_add=out_add, flat_add=flat_add,
                                np_maps=np_maps, classification=classification)

    if score_object == "y" or classification is True:
        try:
            sc_all = []
            for si, sp in zip(scoring, score_pen):
                sc = si(y, pre_y)
                if np.isnan(sc):
                    sc = uniform_score(score_pen=sp)
                sc_all.append(sc)

        except (ValueError, RuntimeWarning, TypeError):

            sc_all = [uniform_score(score_pen=i) for i in score_pen]

    else:

        ######dy under test########
        pre_dy, dy = calculate_derivative_y(expr01, x, terminals, np_maps=np_maps)
        if pre_dy is None:
            sc_all = [uniform_score(score_pen=i) for i in score_pen]
        else:
            try:
                sc_all = []
                for si, sp in zip(scoring, score_pen):
                    sc = []
                    for i, j in zip(dy, pre_dy):
                        index = np.isfinite(i) * np.isfinite(j)
                        sci = si(i[index], j[index])
                        sc.append(sci)
                    sc = [uniform_score(score_pen=sp) if np.isnan(i) or i is None else i for i in sc]
                    sc = sum(sc) / len(sc)
                    sc_all.append(sc)

            except (ValueError, RuntimeWarning, TypeError):

                sc_all = [uniform_score(score_pen=i) for i in score_pen]
    if details:
        return sc_all, expr01, pre_y
    else:
        return sc_all, str(expr01), 0


def calculate_derivative_y(expr01, x, terminals, np_maps=None):
    """
    Something error for reference:

    M. Schmidt, H. Lipson, Distilling free-form natural laws from experimental data, Science, 324 (2009),
    81–85.

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    x: list of np.ndarray
        list of xi
    terminals: list of sympy.Symbol
        features and constants
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    pre_dy_all:np.ndarray or float
        pre-dy
    dy_all: np.ndarray or float
        dy
    """
    warnings.filterwarnings("ignore")
    if not isinstance(expr01, (Expr, NewArray)):
        return None, None
    if len(expr01.free_symbols) < 2:
        return None, None

    free_symbols = list(expr01.free_symbols)[:4] if len(expr01.free_symbols) > 4 else list(expr01.free_symbols)

    free_symbols.sort(key=lambda x: x.name)
    par = list(itertools.combinations(free_symbols, 2))

    # free_symbols = [] # no dependence
    free_symbols = par  # all denpendence
    # free_symbols = [par[0]]
    # free_symbols = par[-1:]
    # free_symbols = par[1:]

    try:
        pre_dy_all = []
        dy_all = []
        for pari in par:
            pari = list(pari)
            pari.sort(key=lambda x: x.name)
            i, j = pari
            subbb = {}
            subbb1 = {k: Function("{}f".format(k.name))(v) for k, v in free_symbols if k is not i and v is i}
            subbb2 = {k: Function("{}f".format(k.name))(v) for v, k in free_symbols if k is not i and v is i}
            subbb.update(subbb1)
            subbb.update(subbb2)
            subb_re = dict(zip(subbb.values(), subbb.keys()))

            fdv1 = diff(expr01.subs(subbb), i, evaluate=True)
            fdv1 = fdv1.subs(subb_re)
            subbb = {}
            subbb3 = {k: Function("{}f".format(k.name))(v) for k, v in free_symbols if k is not j and v is j}
            subbb4 = {k: Function("{}f".format(k.name))(v) for v, k in free_symbols if k is not j and v is j}
            subbb.update(subbb3)
            subbb.update(subbb4)
            subb_re = dict(zip(subbb.values(), subbb.keys()))
            fdv2 = diff(expr01.subs(subbb), j, evaluate=True).subs(subb_re)
            fdv = fdv2 / fdv1

            func0 = utilities.lambdify(terminals, fdv, modules=[np_maps, "numpy"])
            pre_dy = func0(*x)

            ff = diff(i, j, evaluate=False)
            func0 = utilities.lambdify(terminals, ff, modules=[np_maps, "numpy"])
            dy = func0(*x)

            pre_dy_all.append(pre_dy)
            dy_all.append(dy)

        pre_dy_all = np.array(pre_dy_all)
        dy_all = np.array(dy_all)

    except (ValueError, RuntimeWarning, TypeError):
        pre_dy_all, dy_all = None, None

    return pre_dy_all, dy_all


def calculate_cv_score(expr01, x, y, terminals, scoring=None, score_pen=(1,), cv=5, refit=True,
                       add_coef=True, filter_warning=True, inter_add=True,
                       inner_add=False, vector_add=False, out_add=False, flat_add=False, np_maps=None,
                       classification=False, score_object="y", details=False):
    """
    Use cv spilt for score, return the mean_test_score.
    Use cv spilt for predict, return the cv_predict_y.(have not be used)

    Notes:
        if cv and refit, all the data is refit to determine the coefficients.
        Thus the expression is not compact with the this scores, when re-calculated by this expression.

    Parameters
    ----------

    score_object:
        score by y or delta y
    classification:
        classification or not
    refit: True:
        use forced, refit the coefficient use all data.
    cv:sklearn.model_selection._split._BaseKFold,int
        the shuffler must be False
    vector_add
    expr01: Expr
        sympy expression.
    x: list of np.ndarray
        list of xi
    y: np.ndarray
        y value
    terminals: list of sympy.Symbol
        features and constants
    scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
        See Also sklearn.metrics
    score_pen: tuple of  1 or -1
        1 : best is positive, worse -np.inf
        -1 : best is negative, worse np.inf
        0 : best is positive , worse 0
    add_coef: bool
        bool
    filter_warning: bool
        bool
    inter_add: bool
        bool
    inner_add: bool
        bool
    flat_add:bool
        bool
    out_add:bool
        bool
    np_maps: Callable
        user np.ndarray function

    Returns
    -------
    score:float
        score
    expr01: Expr
        New expr.
    pre_y: np.ndarray or float
        np.array or None
    """
    if filter_warning:
        warnings.filterwarnings("ignore")

    if cv == 1:
        sc_all, expr01, pre_y = calculate_score(expr01, x, y, terminals, scoring=scoring, score_pen=score_pen,
                                                add_coef=add_coef, filter_warning=filter_warning, inter_add=inter_add,
                                                inner_add=inner_add, vector_add=vector_add, out_add=out_add,
                                                flat_add=flat_add, np_maps=np_maps, classification=classification,
                                                score_object=score_object, details=details)
        return sc_all, expr01, pre_y

    else:
        if score_object != "y":
            raise TypeError('cv>1 and score_object !="y" are not compatible')

        if isinstance(cv, int):
            cv = KFold(cv, shuffle=False)

        if True:
            _, expr01, _ = calculate_score(expr01, x, y, terminals, scoring=scoring, score_pen=score_pen,
                                           add_coef=add_coef, filter_warning=filter_warning,
                                           inter_add=inter_add,
                                           inner_add=inner_add, vector_add=vector_add, out_add=out_add,
                                           flat_add=flat_add, np_maps=np_maps, classification=classification,
                                           score_object=score_object, details=details)

        cv_sc_all = []
        # cv_expr01 = []
        cv_pre_y = []
        xx = [xi for xi in x if isinstance(xi, np.ndarray)]
        c = [xi for xi in x if not isinstance(xi, np.ndarray)]
        xx = [xi.reshape((-1, 1)) if xi.ndim == 1 else xi.T for xi in xx]

        for train_index, test_index in cv.split(xx[0], y):

            X_train = [xi[train_index] for xi in xx]
            X_test = [xi[test_index] for xi in xx]
            y_train, y_test = y[train_index], y[test_index]

            X_train.reverse()
            X_test.reverse()
            nc = copy.deepcopy(c)
            nc.reverse()
            X_train = [X_train.pop() if isinstance(xi, np.ndarray) else nc.pop() for index, xi in enumerate(x)]
            nc = copy.deepcopy(c)
            nc.reverse()
            X_test = [X_test.pop() if isinstance(xi, np.ndarray) else nc.pop() for index, xi in enumerate(x)]

            pre_y, expr01 = calculate_y(expr01, X_train, y_train, terminals,
                                        x_test=X_test, y_test=y_test,
                                        add_coef=False,
                                        filter_warning=filter_warning, inter_add=inter_add,
                                        inner_add=inner_add,
                                        vector_add=vector_add, out_add=out_add, flat_add=flat_add,
                                        np_maps=np_maps, classification=classification)

            try:
                sc_all = []
                for si, sp in zip(scoring, score_pen):
                    sc = si(y_test, pre_y)
                    if np.isnan(sc):
                        sc = uniform_score(score_pen=sp)
                    sc_all.append(sc)

            except (ValueError, RuntimeWarning, TypeError):

                sc_all = [uniform_score(score_pen=i) for i in score_pen]

            cv_sc_all.append(sc_all)
            # cv_expr01.append(expr01)
            cv_pre_y.append(pre_y)

        sc_all = list(np.mean(np.array(cv_sc_all), axis=0))
        try:
            pre_y = np.concatenate(cv_pre_y)
        except ValueError:
            pre_y = None

        if refit is True:
            "the refit only use for see the detial of calcualtion after loop"
            sc_all0, expr01, pre_y0 = calculate_score(expr01, x, y, terminals, scoring=scoring, score_pen=score_pen,
                                                      add_coef=add_coef, filter_warning=filter_warning,
                                                      inter_add=inter_add,
                                                      inner_add=inner_add, vector_add=vector_add, out_add=out_add,
                                                      flat_add=flat_add, np_maps=np_maps, classification=classification,
                                                      score_object=score_object, details=details)

        return sc_all, expr01, pre_y


def score_dim(dim_, dim_type, fuzzy=False):
    if dim_type is None:
        return 1
    elif isinstance(dim_type, str):
        if dim_type == 'integer':
            return 1 if dim_.isinteger() else 0
        elif dim_type == 'coef':
            return 1 if not dim_.anyisnan() else 0
        elif dim_type == 'ignore':
            return 1
        else:
            raise TypeError("dim_type should be None,'coef', 'integer', special Dim or list of Dim")
    elif isinstance(dim_type, list):
        return 1 if dim_ in dim_type else 0
    elif isinstance(dim_type, Dim):
        if fuzzy:
            return 1 if dim_.is_same_base(dim_type) else 0
        else:
            return 1 if dim_ == dim_type else 0

    else:
        print(dim_type)
        raise TypeError("dim_type should be None,'coef','integer', special Dim or list of Dim")


def calcualte_dim(expr01, terminals, dim_list, dim_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    terminals: list of sympy.Symbol
        features and constants
    dim_list: list of Dim
        dims of features and constants
    dim_maps: Callable
        user dim_maps

    Returns
    -------
    Dim:
        dimension
    dim_score
        is target dim or not
    """
    terminals = [str(i) for i in terminals]
    if not dim_maps:
        dim_maps = dim_map()
    func0 = utilities.lambdify(terminals, expr01, modules=[dim_maps])
    try:
        dim_ = func0(*dim_list)
    except (ValueError, TypeError, ZeroDivisionError, NameError):
        dim_ = dnan
    if isinstance(dim_, (float, int)):
        dim_ = dless
    if not isinstance(dim_, Dim):
        dim_ = dnan

    return dim_


def calcualte_dim_score(expr01, terminals, dim_list, dim_type, fuzzy, dim_maps=None):
    """

    Parameters
    ----------
    expr01: Expr
        sympy expression.
    terminals: list of sympy.Symbol
        features and constants
    dim_list: list of Dim
        dims of features and constants
    dim_maps: Callable
        user dim_maps
    dim_type:list of Dim
        target dim
    fuzzy:
        fuzzy dim or not

    Returns
    -------
    Dim:
        dimension
    dim_score
        is target dim or not
    """
    dim_ = calcualte_dim(expr01, terminals, dim_list, dim_maps=dim_maps)

    dim_score = score_dim(dim_, dim_type, fuzzy)
    return dim_, dim_score


def calculate_collect_(ind, context, x, y, terminals_and_constants_repr, gro_ter_con,
                       dim_ter_con_list, dim_type, fuzzy, cv=1, refit=True,
                       scoring=None, score_pen=(1,),
                       add_coef=True, filter_warning=True, inter_add=True, inner_add=False,
                       vector_add=False, out_add=False, flat_add=False,
                       np_maps=None, classification=False, dim_maps=None, cal_dim=True, score_object="y",
                       details=False):
    expr01 = compile_context(ind, context, gro_ter_con)

    if cal_dim:
        dim, dim_score = calcualte_dim_score(expr01, terminals_and_constants_repr,
                                             dim_ter_con_list, dim_type, fuzzy,
                                             dim_maps=dim_maps)
    else:
        dim, dim_score = dless, 1

    score, expr01, pre_y = calculate_cv_score(expr01, x, y, terminals_and_constants_repr,
                                              cv=cv, refit=refit,
                                              add_coef=add_coef, inter_add=inter_add,
                                              inner_add=inner_add, vector_add=vector_add, out_add=out_add,
                                              flat_add=flat_add,
                                              scoring=scoring, score_pen=score_pen,
                                              filter_warning=filter_warning,
                                              np_maps=np_maps, classification=classification, score_object=score_object,
                                              details=details
                                              )

    return score, dim, dim_score, expr01, pre_y


score_collection = {'explained_variance': metrics.explained_variance_score,
                    'max_error': metrics.max_error,
                    'neg_mean_absolute_error': metrics.mean_absolute_error,
                    'neg_mean_squared_error': metrics.mean_squared_error,
                    'neg_root_mean_squared_error': metrics.mean_squared_error,
                    'r2': metrics.r2_score,
                    'accuracy': metrics.accuracy_score,
                    'precision': metrics.precision_score,
                    'f1': metrics.f1_score,
                    'balanced_accuracy': metrics.balanced_accuracy_score,
                    'average_precision': metrics.average_precision_score, }
