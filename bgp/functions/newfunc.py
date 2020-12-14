import numpy as np
import sympy
from sympy import Function

from bgp.functions.dimfunc import Dim
from bgp.functions.gsymfunc import NewArray


def newfuncV(operation, arity=1, name="Fc"):
    """

    Parameters
    ----------
    operation:callable
        the detail of operation only accept +,-,*,/,abs,-(negative),x^n.
    arity: int
        the arity of operation.
    name:str
        name.
    """
    func = Function(name)

    def npf(*x):
        return operation(*x)

    def gsymf(*x):
        return operation(*x)

    def dimf(*dim):
        return operation(*dim)

    return {"func": func, "arity": arity, "name": name, "np_func": npf,
            "dim_func": dimf, "sym_func": gsymf}


def newfuncD(operation, name="Fc", keep=True, is_jump=False, check=True):
    """

    Parameters
    ----------
    operation : callable
        the detail of opearation only accept +,-,*,/,-(negative),x^n
    name:str
        name
    keep:bool
        the group size after this function. true is the input size,and false is 1.
    is_jump:bool
        the bool means the rem and rem_dim can be treat 2+ domension problems or not.
    check:bool
        check the function building
    """

    func = Function(name)
    func.is_jump = is_jump
    func.keep = keep

    def npf(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    res = operation(x[0], x[1])
                    if keep:
                        assert isinstance(res, tuple)
                    return np.array(res)
                else:  # >=3
                    if is_jump:
                        return x
                    else:
                        return np.array(operation(*x))
            else:  # 1
                return x
        else:  # number
            return x

    def gsymf(x):
        if isinstance(x, (np.ndarray, NewArray)):
            if x.shape[0] == 2:
                res = operation(x[0], x[1])
                if keep:
                    return NewArray(res)
                else:
                    return res
            else:  # >=3
                if is_jump:
                    return x
                else:
                    res = operation(*x)
                    if keep:
                        return NewArray(res)
                    else:
                        return res
        else:  # 1
            return x

    def dimf(dim):

        if isinstance(dim, Dim):
            if dim.ndim == 1:
                return dim
            elif dim.shape[0] == 2:
                if keep:
                    return Dim(operation(*dim))
                else:
                    return Dim(operation(*dim))
            else:
                if is_jump:
                    return dim
                else:
                    return Dim(operation(*dim))
        else:  # number
            return dim

    res = {"func": func, "arity": 1, "name": name, "np_func": npf, "dim_func": dimf, "sym_func": gsymf}

    if check:
        check_funcD(res)

    return res


def check_funcD(funcs, self_grpup=2):
    """self_group>=2"""

    def test_npf():
        a = 1
        b = np.array([1, 2, 3, 4, 5])
        c = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        d = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        e = np.array([[1, 2, 3, 4, 5] * self_grpup])
        np_func = funcs["np_func"]
        np_func(a)
        np_func(b)
        np_func(c)
        s = np_func(d)
        s = np_func(e)

    def test_dim_func():
        a = 1
        b = Dim(np.array([1, 2, 3, 4, 5]))
        c = Dim(np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))
        d = Dim(np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))
        e = Dim(np.array([[1, 2, 3, 4, 5] * self_grpup]))
        dim_func = funcs["dim_func"]
        dim_func(a)
        dim_func(b)
        dim_func(c)
        s = dim_func(d)
        s = dim_func(e)

    def test_sym_func():
        a = 1
        b = sympy.Symbol("x")
        c = NewArray((sympy.Symbol("x"), sympy.Symbol("y")))
        d = NewArray((sympy.Symbol("x"), sympy.Symbol("x"), sympy.Symbol("z")))
        e = NewArray([sympy.Symbol("x")] * self_grpup)
        sym_func = funcs["sym_func"]
        sym_func(a)
        sym_func(b)
        sym_func(c)
        s = sym_func(d)
        s = sym_func(e)

    test_npf()
    test_dim_func()
    test_sym_func()


if __name__ == "__main__":
    def funcs(*arg):
        return sum(arg)


    newfuncD(funcs, name="Fc", keep=False, is_jump=False, check=True)


    def funcs(*arg):
        return arg[0] ** 2 + arg[1]


    newfuncD(funcs, name="Fc", keep=False, is_jump=True, check=True)


    def funcs(*arg):
        return tuple([2.5 * argi ** 2 for argi in arg])


    newfuncD(funcs, name="Fc", keep=True, is_jump=False, check=True)
