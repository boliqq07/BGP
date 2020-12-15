import copy
import sys

import numpy as np
import sympy
from sympy import Number, Expr
from sympy.core.numbers import ComplexInfinity, NumberSymbol

from bgp.calculation.coefficient import get_args

"""lambidfy"""


def general_expr_dict(self, expr_init_map, free_symbol, gsym_map, simplifying=False):
    """gen expr"""

    init_sub = expr_init_map.items()
    for i, j in init_sub:
        self = self.xreplace({sympy.Symbol(i): j})

    fs = free_symbol
    func = sympy.lambdify(fs[0], self, modules=(gsym_map, "sympy"))

    res = func(*fs[1])
    if simplifying:
        res = sympy.simplify(res)

    return res


def general_expr(self, pset, simplifying=False):
    """

    Parameters
    ----------
    simplifying: bool
    self:sympy.Expr
    pset:SymbolSet

    Returns
    -------

    """

    res = general_expr_dict(self, pset.expr_init_map, pset.free_symbol,
                            pset.gsym_map, simplifying=simplifying)

    return res


def group_str(self, pset, feature_name=False):
    """
    return expr just build by input feature name.

    Parameters
    ----------
    self:sympy.Expr or SymbolTree
    pset:SymbolSet
    feature_name:Bool

    Returns
    -------

    """
    #####get expr
    if isinstance(self, Expr):
        expr = copy.deepcopy(self)
        expr = simple(expr, pset.gro_ter_con)[0]
    else:
        if hasattr(self, "coef_expr"):
            expr = self.coef_expr
        elif hasattr(self, "expr") and self.expr is not None:
            expr = self.expr
        else:
            if hasattr(self, "capsule"):
                self = self.capsule
            try:
                expr = compile_context(self, pset.context, pset.gro_ter_con)
            except TypeError:
                raise TypeError("the first inupt must be sympy.expr or SymbolTree,"
                                "the second is SymbolSet")

    ### replace newi to (xi+xj*xk...)
    e_map_va1 = list(pset.expr_init_map.items())
    e_map_va1.reverse()
    for i, j in e_map_va1:
        expr = expr.subs(i, j)

    ### to str
    name_subd = str(expr)

    ### replace Vi() to Vi*(),Vi+()
    arg_list = get_args(expr, sole=False)
    V_map1 = {ar.name: str([str(_) for _ in ar.arr.ravel()]) for ar in arg_list if
              hasattr(ar, "arr") and ar.tp == "Coef"}
    V_map2 = {ar.name: str([str(_) for _ in ar.arr.ravel()]) for ar in arg_list if
              hasattr(ar, "arr") and ar.tp == "Const"}

    V_map_va1 = list(V_map1.items())
    V_map_va2 = list(V_map2.items())
    V_map_va1.reverse()
    V_map_va2.reverse()

    for i, j in V_map_va1:
        name_subd = name_subd.replace(i, "%s*" % j)
    for i, j in V_map_va2:
        name_subd = name_subd.replace(i, "%s+" % j)

    ### replace gxi to [xi,xj]
    t_map_va1 = list(pset.terminals_init_map.items())
    t_map_va1.reverse()
    for i, j in t_map_va1:
        name_subd = name_subd.replace(i, j)

    ### replace ci to (float,)
    c_map_va1 = list(pset.data_x_dict.keys())
    c_map_va1 = [i for i in c_map_va1 if "c" in i]
    c_map_va1.reverse()
    for i in c_map_va1:
        name_subd = name_subd.replace(i, "%.3e" % float(pset.data_x_dict[i]))

    ### replace represent xi to (latex,)
    if feature_name:
        if pset.terminals_fea_map:
            for j1, j2 in pset.terminals_fea_map.values():
                name_subd = name_subd.replace(j1, j2)
        else:
            print("Don not assign the feature_name to pset when pest.add_features")
    print(name_subd)
    return name_subd


def simple(expr01, groups):
    """
    str to sympy.Expr function.
    add conv to MMdd and MMul.
    the calcualte conv need conform with np_func()!!
    
    is_jump: jump the calculate >= 3 (group_size).
    keep: the calculate is return then input group_size or 1.
    """

    def max_method(expr):

        new = [calculate_number(i) for i in expr.args]
        try:
            exprarg_new = list(zip(*new))[0]
            n = list(list(zip(*new))[1])
            expr = expr.func(*exprarg_new)
            n.append(1)
            le = len(set(n))
            if le >= 3:
                return expr, np.nan
            else:
                return expr, max(n)
        except IndexError:
            print(expr)
            return expr, np.nan

    def calculate_number(expr):

        if isinstance(expr, sympy.Symbol):
            return expr, groups[expr.name]
        elif isinstance(expr, (Number, ComplexInfinity)):
            return expr, 1
        elif isinstance(expr, NumberSymbol):
            return expr, 1
        else:

            if hasattr(expr.func, "keep"):
                expr_arg, ns = calculate_number(expr.args[0])

                if expr.func.keep:  ###["Self,Conv"]
                    if ns == 1:
                        expr = expr_arg
                        return expr, ns
                    elif ns == 2:
                        expr = expr.func(expr_arg)
                        expr.conu = ns
                        return expr, ns
                    elif ns >= 3:
                        if expr.func.is_jump:
                            expr = expr_arg
                            return expr, ns
                        else:
                            expr = expr.func(expr_arg)
                            expr.conu = ns
                            return expr, ns
                    else:
                        expr = expr_arg
                        return expr, ns

                else:
                    if ns == 1:
                        expr = expr_arg
                        return expr, ns
                    elif ns == 2:
                        expr = expr.func(expr_arg)
                        expr.conu = ns
                        return expr, 1
                    elif ns >= 3:
                        if expr.func.is_jump:
                            expr = expr_arg
                            return expr, ns
                        else:
                            expr = expr.func(expr_arg)  ###["MAdd", "MMul"]
                            expr.conu = ns
                            return expr, 1
                    else:
                        # expr = expr_arg
                        expr = expr.func(expr_arg)
                        expr.conu = ns
                        return expr, ns

            elif hasattr(expr, "arr"):
                #### expr, ns = max_method(expr)
                #### assert expr.arr.shape[0] == ns  #
                return max_method(expr)

            else:
                return max_method(expr)

    expr01 = calculate_number(expr01)
    return expr01


def compile_context(expr, context, gro_ter_con, simplify=True):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param context: dict
    :param simplify: bool
    :param gro_ter_con: list if group_size
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.

    """
    if isinstance(expr, str):
        code = expr
    else:
        code = repr(expr)

    try:
        expr = eval(code, context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)
    if simplify:
        expr = simple(expr, gro_ter_con)[0]
    return expr


def compile_(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    if isinstance(expr, str):
        code = expr
    else:
        code = repr(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)
