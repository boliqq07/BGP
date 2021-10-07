"""
vector coef and vector const, which is a UndefinedFunction to excape the auto calculation of numpy to sympy.
"""
import copy
import warnings
from collections import Counter

import numpy as np
import sympy
from scipy import optimize
from sklearn.utils import resample
from sympy.core.numbers import NegativeOne
from sympy.core.function import UndefinedFunction, Function


class Coef(UndefinedFunction):
    """
    generate metaclass, the type of identity is .arr,.tp ,rather isinstance.
    """

    def __new__(mcs, name, arr):

        def lfun(x):
            if isinstance(x, np.ndarray):
                return arr * x
            else:
                return arr.ravel() * x

        implementation = lfun
        f = super(Coef, mcs).__new__(mcs, name=name, _imp_=staticmethod(implementation))
        f.arr = arr
        f.name = name
        f.tp = "Coef"
        return f

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Coef):
            return np.all(np.all((self.arr, other.arr)))
        else:
            return False

    def __hash__(self):
        return hash((self.name, str(self.arr)))


class Const(UndefinedFunction):
    """
    generate metaclass, the type of identity is .arr,.tp ,rather isinstance
    """

    def __new__(mcs, name, arr):

        def lfun(x):
            if isinstance(x, np.ndarray):
                return arr + x
            else:
                return arr.ravel() + x

        implementation = lfun
        f = super(Const, mcs).__new__(mcs, name=name, _imp_=staticmethod(implementation))
        f.arr = arr
        f.name = name
        f.tp = "Const"
        return f

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Coef):
            return np.all(np.all((self.arr, other.arr)))
        else:
            return False

    def __hash__(self):
        return hash((self.name, str(self.arr)))


def get_args(expr, sole=True):
    """

    Parameters
    ----------
    expr:sympy.Expr
    sole:only find unique term

    Returns
    -------
    list
    """

    def _get_args(expr_):
        """"""
        list_arg = []
        for argi in expr_.args:
            list_arg.append(argi)
            if argi.args:
                re = _get_args(argi)
                list_arg.extend(re)

        return list_arg

    list_a = _get_args(expr)
    if sole:
        count = Counter(list_a)
        term = [i for i, j in count.items() if j == 1]
        list_a = term

    return list_a


def find_args(expr_, patten):
    """find the term of patten, judge by hash rather type"""
    if len(expr_.args) > 0:
        if patten in expr_.args:
            return expr_.args
        else:
            for argi in expr_.args:
                d = find_args(argi, patten)
                if d is not None:
                    return d


def replace_args(expr_, old, new):
    """find the term of patten, judge by hash rather type"""
    if len(expr_.args) > 0:
        if old in expr_.args:
            compo = list(expr_.args)
            indexs = compo.index(old)
            compo.remove(old)
            compo.insert(indexs, new)
            return expr_.func(*compo)
        else:

            compo = [replace_args(argi, old, new) for argi in expr_.args]
            return expr_.func(*compo)
    else:
        return expr_


def _replace_args_first(expr_, old, new, keep=False):
    """find the term of patten, judge by hash rather type"""
    if not keep:

        if len(expr_.args) > 0:

            if old in expr_.args:
                compo = list(expr_.args)
                indexs = compo.index(old)
                compo.remove(old)
                compo.insert(indexs, new)
                return expr_.func(*compo), True
            else:
                compo = list(expr_.args)
                for i, argi in enumerate(expr_.args):
                    argi, keep = _replace_args_first(argi, old, new)
                    if keep:
                        compo[i] = argi
                        break
                    else:
                        pass
                return expr_.func(*compo), True

        else:
            return expr_, False
    else:
        return expr_, keep


def replace_args_first(expr_, old, new):
    """a"""
    return _replace_args_first(expr_, old, new)[0]


M3 = (Function("MAdd"), Function("MSub"), Function("Conv"))


def _flatten_add_f(expr01, cof_list, cof_dict, vector_add):
    if len(expr01.args) > 0:

        if isinstance(expr01, sympy.Add):
            arg_list = []
            for i, argi in enumerate(expr01.args):
                argi_new = _flatten_add_f(argi, cof_list, cof_dict, vector_add)
                if isinstance(argi_new, M3):
                    if vector_add:
                        arg_list.append(argi_new)
                        # except with both W and V
                    else:
                        wi = sympy.Symbol("W%s" % len(cof_list))
                        arg_list.append(sympy.Mul(wi, argi_new))
                        cof_list.append(wi)
                elif not argi_new.is_number:
                    wi = sympy.Symbol("W%s" % len(cof_list))
                    arg_list.append(sympy.Mul(wi, argi_new))
                    cof_list.append(wi)
                else:
                    arg_list.append(argi_new)
            return sympy.Add(*arg_list)

        elif isinstance(expr01, sympy.Mul):
            if NegativeOne() in expr01.args and len(expr01.args) == 2:
                argi = [i for i in expr01.args if i is not NegativeOne()][0]
                argi_new = _flatten_add_f(argi, cof_list, cof_dict, vector_add)
                return sympy.Mul(NegativeOne(), argi_new)
            else:
                return expr01

        elif isinstance(expr01, M3):
            if vector_add:
                conu = expr01.conu
                if conu > 1:

                    Wi = sympy.Symbol("V%s" % len(cof_dict))
                    argi = expr01.args[0]
                    argi_new = _flatten_add_f(argi, cof_list, cof_dict, vector_add)
                    expr02 = expr01.func(sympy.Mul(Wi, argi_new))
                    cof_dict[Wi] = conu
                    return expr02
                else:
                    return expr01
            else:
                return expr01
        else:
            return expr01
    else:
        return expr01


def flatten_add_f(expr01, cof_list, cof_dict, vector_add):
    expr01 = _flatten_add_f(expr01, cof_list, cof_dict, vector_add)

    if isinstance(expr01, sympy.Add):
        pass
    elif isinstance(expr01, M3):
        if vector_add:
            pass
        else:
            A = sympy.Symbol("A")
            expr01 = sympy.Mul(expr01, A)
            cof_list.append(A)
    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)
    return expr01


def out_add_f(expr01, cof_list, cof_dict, vector_add):
    if isinstance(expr01, sympy.Add):
        wiss = [sympy.Symbol("W%s" % i) for i, j in enumerate(expr01.args)]
        args_new = [(wi, sympy.Mul(wi, ei)) if not ei.is_number else (None, ei) for ei, wi in
                    zip(expr01.args, wiss)]
        wis, we = zip(*args_new)
        cof_list.extend([wi for wi in wis if wi is not None])
        argss = sympy.Add(*we)
        expr01 = argss

    elif isinstance(expr01, M3):

        exprin1 = expr01.args[0]
        conu = expr01.conu
        if isinstance(exprin1, sympy.Add):
            wiss = [sympy.Symbol("W%s" % i) for i, j in enumerate(exprin1.args)]
            args_new = [(wi, sympy.Mul(wi, ei)) if not ei.is_number else (None, ei) for ei, wi in
                        zip(exprin1.args, wiss)]
            wis, we = zip(*args_new)
            cof_list.extend([wi for wi in wis if wi is not None])
            argss = sympy.Add(*we)
            expr01 = expr01.func(argss)

        if vector_add:
            if conu > 1:
                Wi = sympy.Symbol("V")
                arg = expr01.args[0]
                expr02 = expr01.func(sympy.Mul(Wi, arg))
                cof_dict[Wi] = conu
                expr01 = expr02
        else:
            A = sympy.Symbol("A")
            expr01 = sympy.Mul(expr01, A)
            cof_list.append(A)

    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)
    return expr01


def inner_add_f(expr01, cof_list, cof_dict, vector_add):
    arg_list = get_args(expr01)

    cho = []
    cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
    cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
    [cho.extend(i) for i in cho_add]

    a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]

    for ai, choi in zip(a_cho, cho):
        expr02 = expr01.xreplace({choi: sympy.Mul(ai, choi)})
        cof_list.append(ai)
        expr01 = expr02

    if vector_add:
        cho_add2 = [i for i in arg_list if isinstance(i, M3)
                    if hasattr(i, "conu") and i.conu > 1]

        for i, j in enumerate(cho_add2):
            Wi = sympy.Symbol("V%s" % i)
            arg = j.args[0]
            arg_new = j.func(sympy.Mul(Wi, arg))
            expr02 = expr01.xreplace({j: arg_new})
            cof_dict[Wi] = j.conu
            expr01 = expr02

    if isinstance(expr01, sympy.Add):
        pass
    elif isinstance(expr01, M3):
        if vector_add:
            pass
        else:
            A = sympy.Symbol("A")
            expr01 = sympy.Mul(expr01, A)
            cof_list.append(A)
    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)

    return expr01


def add_coefficient(expr01, inter_add=True, inner_add=False, vector_add=False, out_add=False, flat_add=False):
    """
    Try add the placeholder coefficient to sympy expression.
    1. Add Wi,A,B normal coefficients to expression.
    2. Add V, Vi vector coefficients to expression, for this type of coefficient ,
    there should be with expr01.conu for Function("MAdd"), Function("MSub").
    more details can be found in ..translate.simple

    Parameters
    ----------
    expr01: Expr
        sympy expressions
    inter_add: bool
        bool
    inner_add: bool
        bool
    vector_add:bool
        bool
    flat_add:bool
        bool
    out_add:bool
        bool
    Returns
    -------
    expr
    """
    cof_list = []
    cof_dict = {}
    assert len([i for i in [inner_add, out_add, flat_add] if i is True]) <= 1, \
        "For the 'out_add','flat_add'and 'out_add', you should just choose one of them."

    if out_add:  # out layer
        expr01 = out_add_f(expr01, cof_list, cof_dict, vector_add)

    elif inner_add:  # out and inner layer each add and Madd...
        expr01 = inner_add_f(expr01, cof_list, cof_dict, vector_add)

    elif flat_add:  # out and inner layer each add and Madd... just open by add.
        expr01 = flatten_add_f(expr01, cof_list, cof_dict, vector_add)
    else:
        A = sympy.Symbol("A")
        expr01 = sympy.Mul(expr01, A)
        cof_list.append(A)

    if inter_add:  # intercept
        B = sympy.Symbol("B")
        expr01 = sympy.Add(expr01, B)
        cof_list.append(B)

    return expr01, cof_list, cof_dict


class CheckCoef(object):
    """
    group the coef and pack the calculate part out of loop.
    """

    def __init__(self, cof_list, cof_dict):
        """

        Parameters
        ----------
        cof_list: Sized
        cof_dict:dict
        """
        self.cof_list = cof_list
        self.cof_dict = cof_dict
        self.cof_dict_keys = list(cof_dict.keys())
        self.cof_dict_values = list(cof_dict.values())
        lt = []
        lt.extend(cof_list)
        lt.extend(list(self.cof_dict_keys))
        self.name = lt
        self.num = len(cof_list) + sum(list(self.cof_dict_values))

    def __len__(self):
        return len(self.name)

    @property
    def ind(self):
        lsa = list(range(len(self.cof_list)))
        n = len(lsa)
        for k in self.cof_dict_values:
            lsi = np.arange(k)+n
            lsa.append(lsi)
            n = lsi[-1] + 1

        return lsa

    def group(self, p, decimals=False):
        """change the p to grpup"""
        p = np.array(p)

        ls = [p[i] if isinstance(i, int) else p[i].reshape((-1, 1)) for i in self.ind]

        if decimals:
            return self.dec(ls)
        else:
            return ls
        # return ls

    def dec(self, ls):
        cof_ = []
        for a_listi, cofi in zip(self.name, ls):

            if not isinstance(cofi, np.ndarray):
                cof_.append(float("%.3e" % cofi))
            else:
                cof_.append(np.array([float("%.3e" % i) for i in cofi]).reshape((-1, 1)))
        return cof_


def cla(pre_y, cl=True):
    pre_y = 1.0 / (1.0 + np.exp(-pre_y))
    if cl:
        pre_y[np.where(pre_y >= 0.5)] = 1
        pre_y[np.where(pre_y < 0.5)] = 0
    return pre_y


def try_add_coef(expr01, x, y, terminals, grid_x=None,
                 filter_warning=True, inter_add=True, inner_add=False, vector_add=False, out_add=False, flat_add=False,
                 np_maps=None, classification=False):
    """
    Try calculate predict y by sympy expression with coefficients.
    if except error return expr itself.

    Parameters
    ----------
    flat_add:bool
        add flat coefficient or not
    out_add:
        add outcoefficientt or not
    vector_add: bool
        add vectorcoefficientt or not
    expr01: sympy.Expr
        sympy expressions
    x: list of np.ndarray
        list of xi
    y: np.ndarray
        y value
    grid_x:
        new x to predict
    terminals: list of sympy.Symbol
        features and constants
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
    pre_y:
        np.array or None
    expr01: Expr
        New expr.
    """
    if filter_warning:
        warnings.filterwarnings("ignore")

    expr00 = copy.copy(expr01)

    expr01, a_list, a_dict = add_coefficient(expr01, inter_add=inter_add, inner_add=inner_add, vector_add=vector_add,
                                             out_add=out_add, flat_add=flat_add)

    cc = CheckCoef(a_list, a_dict)

    ter = []
    ter.extend(terminals)
    ter.extend(cc.name)

    try:

        func0 = sympy.utilities.lambdify(ter, expr01, modules=[np_maps, "numpy", "math"])

        def func(x_, p):
            """"""
            # num_list = []
            # num_list.extend(x_)
            if vector_add:
                p = cc.group(p)
            # num_list.extend(p)

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
            result = optimize.least_squares(res, x0=[1.0] * cc.num, args=(x, y), xtol=1e-4, ftol=1e-5, gtol=1e-5,
                                            # long
                                            jac='3-point', loss='linear')
        else:
            result = optimize.least_squares(res2, x0=[1.0] * cc.num, args=(x, y), xtol=1e-4, ftol=1e-5, gtol=1e-5,
                                            # long
                                            jac='3-point', loss='linear')
        cof = result.x

        cof = cc.group(cof)

        if grid_x is None:
            grid_x = x

        pre_y = func0(*grid_x, *cof)
        if classification:
            pre_y = cla(pre_y, cl=True)

        cof = cc.dec(cof)

        for ai, choi in zip(cc.name, cof):
            if ai in cc.cof_dict_keys:

                fun = Coef(ai.name, choi)
                # replace the Vi to Vi()
                olds0 = find_args(expr01, ai)
                if olds0 is None:
                    raise KeyError("0*wi is 0,and make the placeholder fade out")
                olds = [old for old in olds0 if old is not ai]
                olds = sympy.Mul(*olds)
                expr01 = expr01.xreplace({sympy.Mul(ai, olds): fun(olds)})
            else:
                expr01 = expr01.xreplace({ai: choi})

    except (ValueError, KeyError, NameError, TypeError, ZeroDivisionError,IndexError):

        expr01 = expr00
        pre_y = None

    return pre_y, expr01


def try_add_coef_times(expr01, x, y, terminals, grid_x=None,
                       filter_warning=True, inter_add=True, inner_add=False, vector_add=False, out_add=False,
                       flat_add=False,
                       np_maps=None, classification=False, random_state=0, return_expr=False, resample_number=500):
    if filter_warning:
        warnings.filterwarnings("ignore")

    if grid_x is None:
        grid_x = x

    expr01, a_list, a_dict = add_coefficient(expr01, inter_add=inter_add, inner_add=inner_add, vector_add=vector_add,
                                             out_add=out_add, flat_add=flat_add)

    cc = CheckCoef(a_list, a_dict)

    ter = []
    ter.extend(terminals)
    ter.extend(cc.name)

    func0 = sympy.utilities.lambdify(ter, expr01, modules=[np_maps, "numpy", "math"])

    def func(x_, p):
        """"""
        num_list = []
        num_list.extend(x_)

        p = cc.group(p)
        num_list.extend(p)

        return func0(*num_list)

    def res(p, x_, y_):
        """"""
        ress = y_ - func(x_, p)
        return ress

    def res2(p, x_, y_):
        """"""
        pre_y = func(x_, p)
        pre_y = cla(pre_y, cl=False)
        ress = y_ - pre_y

        return ress

    pre_y_all = []
    expr_all = []
    for times in range(resample_number):
        *x, y = resample(*x, y, replace=True, random_state=times)

        if not classification:
            result = optimize.least_squares(res, x0=[1.0] * cc.num, args=(x, y), xtol=1e-4, ftol=1e-4, gtol=1e-4,
                                            # max_nfev=200,
                                            jac='3-point', loss='linear')
        else:
            result = optimize.least_squares(res2, x0=[1.0] * cc.num, args=(x, y), xtol=1e-4, ftol=1e-4, gtol=1e-4,
                                            # max_nfev=200,
                                            jac='3-point', loss='linear')
        cof = result.x
        cof = cc.group(cof)

        try:

            pre_y = func0(*grid_x + cof)

            if classification:
                pre_y = cla(pre_y, cl=True)

                expr00 = expr01
            if return_expr:

                expr00 = copy.deepcopy(expr01)

                cof = cc.dec(cof)
                for ai, choi in zip(cc.name, cof):
                    if ai in cc.cof_dict_keys:

                        fun = Coef(ai.name, choi)
                        # replace the Vi to Vi()
                        olds0 = find_args(expr00, ai)
                        if olds0 is None:
                            raise KeyError("0*wi is 0,and make the placeholder fade out")
                        olds = [old for old in olds0 if old is not ai]
                        olds = sympy.Mul(*olds)
                        expr00 = expr00.xreplace({sympy.Mul(ai, olds): fun(olds)})
                    else:
                        expr00 = expr00.xreplace({ai: choi})
            else:
                pass

        except (ValueError, KeyError, NameError, TypeError, ZeroDivisionError):

            pre_y = None

        if isinstance(pre_y, np.ndarray):

            if np.all(np.isfinite(pre_y)):
                pre_y_all.append(pre_y)

                if return_expr:
                    expr_all.append(expr00)

    pre_y_all = np.array(pre_y_all)

    if return_expr:
        return pre_y_all, expr_all
    else:

        return pre_y_all
