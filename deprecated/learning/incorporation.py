#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/7/28 16:26
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause


import copy
import random
import warnings
from itertools import product, chain

import numpy as np
import sympy
from deap import creator, gp
from mgetool.tool import time_this_function
from sympy.physics.units import Dimension

from deprecated.learning.incorporationbase import Dim, dim_func, fea_compile, spilt_couple

warnings.filterwarnings("ignore")


######################
# generate_index pop
######################


class Prop(list):
    def __init__(self, name):
        super().__init__()
        self.name = name


class Comp(list):
    def __init__(self, name):
        super().__init__()
        self.name = name


def generate(pset0, dim_list):
    """generate_index core method"""

    same = False
    # same = True
    comp = pset0.terminals_comp
    prop = pset0.terminals_prop

    expr = []
    if dim_list[3] != Dim(Dimension(1)):
        prop_count = [pset0.mapping['powre'], pset0.mapping['pow1'],
                      pset0.mapping['pow2'], pset0.mapping['pow3']]
    else:
        prop_count = [pset0.mapping['exp'], pset0.mapping['log'],
                      pset0.mapping['sqrt'], pset0.mapping['powre'], pset0.mapping['pow1'],
                      pset0.mapping['pow2'], pset0.mapping['pow3']]

    comp_count = [pset0.mapping['exp'], pset0.mapping['log'],
                  pset0.mapping['sqrt'], pset0.mapping['powre'], pset0.mapping['pow1'],
                  pset0.mapping['pow2'], pset0.mapping['pow3']]
    out1_count = [pset0.mapping['pow1'], ]
    link_count = [pset0.mapping['Add'], pset0.mapping['Div'], ]
    out2_count = [pset0.mapping['exp'], pset0.mapping['log'],
                  pset0.mapping['sqrt'], pset0.mapping['pow1'],
                  ]

    for i in prop_count:
        for j in comp_count:
            for l in out1_count:
                expr.append([prop[0], i, comp[0], j, pset0.mapping['Mul'], l])

    expr_list0 = []

    for k in range(len(comp)):
        exprco = copy.deepcopy(expr)
        for expri in exprco:
            expri[0] = prop[k]
            expri[2] = comp[k]
        expr_list0.append(exprco)
    if same is True:
        expr_list = list(zip(*expr_list0))
    else:
        expr_list = list(product(*expr_list0))

    expr_list2 = []
    for m in out2_count:
        for t in link_count:
            exprr = [t] * (len(comp) - 1)
            exprr.append(m)
            for exp1 in expr_list:
                ex = list(chain(*exp1))
                expp = ex + exprr
                expp.reverse()
                expr_list2.append(expp)
    return expr_list2


def generate2(pset0):
    """generate_index core method"""
    comp = pset0.terminals_comp
    prop = pset0.terminals_prop

    prop_count = [
        pset0.mapping['sqrt'], pset0.mapping['powre'], pset0.mapping['pow1'],
        pset0.mapping['pow2'], pset0.mapping['pow3'], pset0.mapping['Abs']]

    comp_count = [pset0.mapping['exp'], pset0.mapping['log'],
                  pset0.mapping['sqrt'], pset0.mapping['powre'], pset0.mapping['pow1'],
                  pset0.mapping['pow2'], pset0.mapping['pow3']]

    link_count = [pset0.mapping['Add'], pset0.mapping['Sub']]

    out2_count = [
        pset0.mapping['pow2'], pset0.mapping['powre'], pset0.mapping['pow1'],
        pset0.mapping['sqrt'], pset0.mapping['powre3'], pset0.mapping['Abs']]

    exprc = [[[k, j, pset0.mapping['Mul']] for j in comp_count] for k in comp]
    exprp = [[[p, i, q] for p in prop] for i, q in zip(prop_count, out2_count)]

    expr_list2 = []
    for t in link_count:
        for ec in product(*exprc):
            for ep in exprp:
                seq = []
                p = pset0.mapping['pow1']
                for c, p in zip(ec, ep):
                    seq.extend(chain(p[:-1], c))
                seq.extend([t] * (len(comp) - 1))
                seq.extend((pset0.mapping['Abs'], p[-1]))

                seq.reverse()
                expr_list2.append(seq)

    return expr_list2


def sympy_prim_set(sym_list):
    """refer to deap"""

    def sub(left, right):
        return left - right

    def protectdiv(left, right):
        return left / right

    def gene_pows(b):
        return sympy.Pow(b, -1)

    def gene_pows1(b):
        return sympy.Pow(b, 1)

    def gene_pows2(b):
        return sympy.Pow(b, 2)

    def gene_pows3(b):
        return sympy.Pow(b, 3)

    def gene_powsre2(b):
        return sympy.Pow(b, -2)

    def gene_powsre3(b):
        return sympy.Pow(b, -3)

    pset0 = gp.PrimitiveSet("MAIN", arity=0)

    pset0.addPrimitive(sympy.Add, name="Add", arity=2)

    pset0.addPrimitive(sub, name='Sub', arity=2)

    pset0.addPrimitive(sympy.Mul, name="Mul", arity=2)

    pset0.addPrimitive(protectdiv, name="Div", arity=2)

    pset0.addPrimitive(sympy.exp, name="exp", arity=1)

    pset0.addPrimitive(sympy.log, name="log", arity=1)

    pset0.addPrimitive(sympy.Abs, name="Abs", arity=1)

    pset0.addPrimitive(sympy.sqrt, name="sqrt", arity=1)

    pset0.addPrimitive(gene_pows, name="powre", arity=1)

    pset0.addPrimitive(gene_pows1, name="pow1", arity=1)

    pset0.addPrimitive(gene_pows2, name="pow2", arity=1)

    pset0.addPrimitive(gene_pows3, name="pow3", arity=1)

    pset0.addPrimitive(gene_powsre2, name="powre2", arity=1)

    pset0.addPrimitive(gene_powsre3, name="powre3", arity=1)

    pset0.primitives_all = pset0.primitives[object]
    pset0.primitives_1 = [i for i in pset0.primitives_all if i.arity == 1]
    pset0.primitives_2 = [i for i in pset0.primitives_all if i.arity == 2]
    pset0.primitives_i = [i for i in pset0.primitives_all if i.arity > 2]

    le = int(len(sym_list) / 2)
    for sym in sym_list[:le]:
        pset0.addTerminal(sym, name=str(sym))

    for sym in sym_list[le:]:
        pset0.addTerminal(sym, name=str(sym))

    pset0.terminals_comp = pset0.terminals[object][:le]
    pset0.terminals_prop = pset0.terminals[object][le:]
    assert len(pset0.terminals_comp) == len(pset0.terminals_prop), "components and propritys must be same size "
    pset0.terminals_all = pset0.terminals[object]
    pset0.ter_name = [i.name for i in pset0.terminals_all]
    pset0.pri_name = [i.name for i in pset0.primitives_all]

    return pset0


def add_expr(sym_):
    """add some experience formulas"""
    from sympy import exp, log, Abs
    x1, x2, x3, x4 = sym_
    listt = [

        x1 * x3 + x2 * x4,
        x4 / x2 + x3 / x1,

        exp(x1) * x3 * exp(x2) * x4,
        exp(x1) / x3 * exp(x2) / x4,
        exp(1 - x1) / x3 * exp(1 - x2) / x4,
        exp(1 - x1) * x3 * exp(1 - x2) * x4,
        x1 ** 2 * x3 * x2 ** 2 * x4,
        x1 ** 2 / x3 * x2 ** 2 / x4,
        x1 ** 0.5 * x3 * x2 ** 0.5 * x4,
        x1 ** 0.5 / x3 * x2 ** 0.5 / x4,
        x1 ** 3 * x3 * x2 ** 3 * x4,
        x1 ** 3 / x3 * x2 ** 3 / x4,
        log(x1) * x3 * log(x2) * x4,
        log(x1) / x3 * log(x2) / x4,
        log(1 - x1) * x3 * log(1 - x2) * x4,
        log(1 - x1) / x3 * log(1 - x2) / x4,
        x1 ** -1 * x3 * x2 ** -1 * x4,
        x1 ** -1 / x3 * x2 ** -1 / x4,

        x2 / x1 * exp(x3 / x4),
        Abs(x2 - x1) * exp(x3 / x4),
        x1 / x2 * exp(x3 / x4),
        (x2 + x1) * exp(x3 / x4),

        x2 / x1 * log(x3 / x4),
        Abs(x2 - x1) * log(x3 / x4),
        x1 / x2 * log(x3 / x4),
        Abs(x2 + x1) * log(x3 / x4),
        x2 / x1 * log(1 - x3 / x4),

        x3 ** x1 + x4 ** x2,
        x1 ** x3 + x2 ** x4,
        (x3 - x4) ** abs(x2 - x1),
        (x3 / x4) ** abs(x2 - x1),
        (x3 - x4) ** -abs(x2 - x1),
        (x3 / x4) ** -abs(x2 - x1),

        (x1 + x2) / x1 / x2 * (x3 + x4),
        (x1 - x2) / x1 / x2 * (x3 - x4),

    ]
    return listt


######################
# check and count part
######################

def check_term(ind, pset):
    """check term by deap method"""
    ter_name = [i.name for i in ind]
    if set(pset.ter_name).issubset(set(ter_name)):
        return ind


def check_term2(expr, sym_list):
    """check term by sympy method"""
    # if set(expr.free_symbols).isdisjoint(sym_list):
    if set(sym_list).issubset(set(expr.free_symbols)):
        return expr
    else:
        return None


def check_expr_unit(expr, sym_list, dim_list):
    """check unit and mark the inaccessible  feature"""
    expr = sympy.simplify(expr)
    func0 = sympy.utilities.lambdify(sym_list, expr, modules=dim_func())
    try:
        state = func0(*dim_list)

    except (TypeError, NameError, ZeroDivisionError):

        state = None

    return state


def count_y(expr, x1, sym_list, sym_unify_list):
    """count y and deal with error"""
    num_list = [*x1.T]
    combine = zip(sym_list, sym_unify_list)
    expr1 = expr.subs(combine)

    try:
        func0 = sympy.utilities.lambdify(sym_list, expr1)
        import warnings
        warnings.filterwarnings("ignore")

        fit_yy = func0(*num_list)
    except (ValueError, ZeroDivisionError, OverflowError, NameError, AttributeError):
        fit_yy = None
    return fit_yy


def check_y_value(expr, x1, sym_list, sym_unify_list):
    """check value and mark the inaccessible and low std feature"""
    expr = sympy.simplify(expr)
    fit_y0 = count_y(expr, x1, sym_list, sym_unify_list)

    if isinstance(fit_y0, np.ndarray):
        if fit_y0.shape[0] == x1.shape[0] and fit_y0.dtype == np.float:
            if np.isfinite(fit_y0).any():
                if fit_y0.std() / fit_y0.mean() >= 0.001 or fit_y0.mean() == 0:
                    return str(expr), fit_y0


def check_corr(pop0):
    """count corrcoef"""
    data = np.array([expr[1] for expr in pop0])
    cov = np.corrcoef(data)

    return cov


def count_cof(cof):
    """check cof and count the number"""
    list_count2 = []
    list_coef = []
    g = np.where(abs(cof) >= 0.99)
    for i in range(cof.shape[0]):
        e = np.where(g[0] == i)
        com = list(g[1][e])
        # ele_ratio.remove(i)
        list_count2.append(com)
        list_coef.append([cof[i, j] for j in com])
    return list_coef, list_count2


def remove_coef(cof__list_all):
    """delete the index of feature with repeat coef """
    random.seed(0)
    reserve = []
    for i in cof__list_all:
        if not cof__list_all:
            reserve.append(i)

    for cof_list in cof__list_all:
        if not cof_list:
            pass
        else:
            if reserve:
                candi = []
                for j in cof_list:

                    con = any([[False, True][j in cof__list_all[k]] for k in reserve])
                    if not con:
                        candi.append(j)
                if any(candi):
                    a = random.choice(candi)
                    reserve.append(a)
                else:
                    pass
            else:
                a = random.choice(cof_list)
                reserve.append(a)
            cof_list_t = copy.deepcopy(cof_list)
            for dela in cof_list_t:
                for cof_list2 in cof__list_all:
                    if dela in cof_list2:
                        cof_list2.remove(dela)
    return reserve


######################
# flow
######################

@time_this_function
def gen_filter(pset, x_feai, xi):
    """
    generate_index pop flow
    --------
    pop1: deap pop
    pop2: sympy pop
    pop3: str pop and str filter
    pop4: check value filer
    pop5: check unit filer, Note dim was transformed to str
    pop6: list of dict

    :rtype: dict

    """
    sym_unify_list, sym_list, dim_list, unit_list = x_feai
    creator.create("Individual", gp.PrimitiveTree, pset=pset)
    random.seed(0)
    pop1 = [creator.Individual(i) for i in generate2(pset)]
    # pop1 = list(filter(None, [check_term(i, pset) for i in pop1]))
    # pop1 = list(set([str(_) for _ in pop1]))
    print(len(pop1))

    pop2 = [sympy.simplify(gp.compile(str(expr), pset)) for expr in pop1]
    pop2.extend(add_expr(sym_list))
    # pop2 = list(filter(None, [check_term2(i, sym_list) for i in pop2]))
    print(len(pop2))

    pop3 = list(set([str(_) for _ in pop2]))
    print(len(pop3))

    pop4 = list(filter(None, [check_y_value(expr, xi, sym_list, sym_unify_list) for expr in pop3]))
    pop4.sort()
    print(len(pop4))

    pop5 = []
    for expr in pop4:
        unit = check_expr_unit(expr[0], sym_list, dim_list)
        if unit:
            pop5.append((*expr, str(unit)))
    pop5.sort()
    print(len(pop5))

    cov = check_corr(pop5)
    listt, listtt = count_cof(cov)
    list2 = remove_coef(listtt)
    list2.sort()
    pop6 = [pop5[_] for _ in list2]
    pop6 = dict(zip(range(len(pop6)), pop6))

    print(len(pop1), len(pop2), len(pop3), len(pop4), len(pop5), len(pop6))
    return pop6


def produce(x, x_unit, x_name, y, store_path=r'C:\Users\Administrator\Desktop\gap\inter_data'):
    """
    generate_index combination feature from all element feature.
    --------

    :param store_path: str, path
    :param x_name: list of str
    :param x_unit:list of sympy.quantity
    :param x: np.ndarray
    :param y: np.ndarray

    :rtype: dict, binary files in path
    """

    x_fea_all = fea_compile(x_unit, s=False, p=False, symbols_name=x_name, pre=True)
    x_fea_i, x_i = spilt_couple(x_fea_all, x, compount_index=None, pro_index=None, n=2)

    # inner cycle
    # for numeric in range(0, len(x_fea_i)):
    for N in range(40, len(x_fea_i)):
        sym_unify_list, sym_list, dim_list, unit_list = x_feai = x_fea_i[N]
        xi = x_i[N]
        pset = sympy_prim_set(sym_list)
        pop = gen_filter(pset, x_feai, xi)
        filename = "_".join([str(_) for _ in sym_list])
