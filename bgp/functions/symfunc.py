#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

import functools

import sympy
from sympy import Function


def sym_vector_map():
    """str to sympy.Expr function"""

    def Div(left, right):
        return left / right

    def Sub(left, right):
        return left - right

    def rem(ax):
        return 1 - ax

    der = Function("Der")

    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'ln': sympy.ln, "log": sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0), "Self": lambda x: x,
                  "Rem": rem}
    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div, "Der": der}

    return functions1, functions2


def sym_dispose_map():
    """user's str to sympy.expr function"""
    Flat = Function("MAdd")
    Comp = Function("MMul")
    Diff = Function("MSub")
    Quot = Function("MDiv")
    Self = lambda x: x
    Conv = Function("Conv")
    Flat.is_jump = False
    Comp.is_jump = False
    Diff.is_jump = True
    Quot.is_jump = True
    Conv.is_jump = True
    Self.is_jump = True

    Flat.keep = False
    Comp.keep = False
    Diff.keep = False
    Quot.keep = False
    Conv.keep = True
    Self.keep = True

    return {"MAdd": Flat, "MMul": Comp, "MSub": Diff, "MDiv": Quot, "Conv": Conv, "Self": Self}
