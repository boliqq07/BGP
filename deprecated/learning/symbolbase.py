# -*- coding: utf-8 -*-

# @Time   : 2019/6/8 21:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: 4.symbollearing.py
# @Software: PyCharm

"""
All part are copy from deap (https://github.com/DEAP/deap)
"""
import copy
import functools
import operator
import random
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial
from inspect import isclass
from operator import attrgetter

import numpy as np
import pandas as pd
import sympy
from deap import creator
from deap.base import Fitness, Toolbox
from deap.gp import PrimitiveSet, PrimitiveTree, staticLimit
from deap.tools import HallOfFame, MultiStatistics, Statistics, initIterate, initRepeat, Logbook
from mgetool.exports import Store
from mgetool.tool import time_this_function, parallelize
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import explained_variance_score, make_scorer, r2_score
from sklearn.utils import assert_all_finite, check_array

warnings.filterwarnings("ignore")

__type__ = object


class FixedTerminal(object):
    """

    """
    __slots__ = ('name', 'value', 'conv_fct', 'arity')

    def __init__(self, terminal):
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str
        self.arity = 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    __repr__ = __str__


class FixedPrimitive(object):
    """

    """
    __slots__ = ('name', 'arity', 'args', 'seq')

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self.args = []
        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.name

    __repr__ = __str__


class FixedPrimitiveSet(object):
    """

    """

    def __init__(self, name):
        self.terminals = []
        self.primitives = []
        self.terms_count = 0
        self.prims_count = 0
        self.arguments = []
        self.context = {"__builtins__": None}
        self.dimtext = {"__builtins__": None}
        self.mapping = dict()
        self.name = name

    def addPrimitive(self, primitive, arity, name=None):

        if name is None:
            name = primitive.__name__

        prim = FixedPrimitive(name, arity)

        assert name not in self.context, "Primitives are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second '%s' primitive." % (name,)

        self.primitives.append(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, name=None):

        if name is None and callable(terminal):
            name = str(terminal)

        assert name not in self.context, "Terminals are required to have a unique x_name. " \
                                         "Consider using the argument 'x_name' to rename your " \
                                         "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal

        prim = FixedTerminal(terminal)
        self.terminals.append(prim)
        self.terms_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class FixedExpressionTree(list):
    """

    """
    hasher = str

    def __init__(self, content):
        list.__init__(self, content)

        assert sum(_.arity - 1 for _ in self.primitives) + 1 >= len(self.terminals)
        assert len(self.terminals) >= 2

    @property
    def root(self):
        """

        Returns
        -------
        start site number of tree
        """
        len_ter = len(self.terminals) - 1
        num_pri = list(range(len(self.primitives)))
        num_pri.reverse()
        i = 0
        for i in num_pri:
            if len_ter == 0:
                break
            elif len_ter <= 0:
                raise ("Add terminals or move back the {}".format(self[i - 1]),
                       "because the {} have insufficient terminals, need {},but get {}".format(self[i - 1],
                                                                                               self[i - 1].arity,
                                                                                               len_ter - self[
                                                                                                   i - 1].arity)
                       )
            len_ter = len_ter - self[i].arity + 1
        # return i  # for wencheng
        if self[i].arity == 1:
            return i
        else:
            return i + 1

    def __deepcopy__(self, memo):

        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return self.str_run_index(self.root)

    __repr__ = __str__

    @property
    def pri_site(self):
        return tuple([p for p, primitive in enumerate(self) if primitive.arity >= 1 and p >= self.root])

    @property
    def ter_site(self):
        return tuple([t for t, primitive in enumerate(self) if primitive.arity == 0])

    @property
    def primitives(self):
        """Return primitives that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity >= 1]

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    @classmethod
    def search_y_name(cls, name):
        """

        Parameters
        ----------
        name

        Returns
        -------

        """
        list_name = []
        for i in range(len(cls)):
            if cls[i].name == name:
                list_name.append(i)
        return list_name

    @property
    def number_branch(self):
        """

        Returns
        -------
        dict,a site tree
        """

        def cal():
            coup_dict = {}
            coup = []
            for _ in range(pri_i.arity):
                coup.append(terminals.pop())
            coup.reverse()
            coup_dict[len_pri - i - 1] = coup
            terminals_new.append(coup_dict)

        primitives = self.primitives
        primitives.reverse()
        len_pri = len(primitives)
        terminals = list(range(len_pri, len(self.terminals) + len_pri))
        terminals_new = []
        for i, pri_i in enumerate(primitives):
            if len(terminals) >= pri_i.arity:
                cal()
            else:
                terminals_new.reverse()
                terminals.extend(terminals_new)
                terminals_new = []
                if len(terminals) >= pri_i.arity:
                    cal()
                else:
                    break
        result = terminals_new or terminals
        return result[0]

    def number_branch_index(self, index):
        """
        Returns
        -------
        number_branch for given index
        """
        if index < self.root or index > len(self.primitives):
            raise IndexError("not a primitives index")
        else:
            def run_index(number_branch=None):
                if number_branch is None:
                    number_branch = self.number_branch
                jk = list(number_branch.keys())[0]
                ji = list(number_branch.values())[0]
                if jk == index:
                    return number_branch
                else:
                    repr1 = []
                    for jii in ji:
                        if isinstance(jii, dict):
                            repr1 = run_index(jii)
                        else:
                            repr1 = []
                        if repr1:
                            break
                    return repr1
        set1 = run_index()
        # set1.sort()
        return set1

    def str_run(self, number_branch=None):
        """
        Returns
        -------
        str of tree by given number_branch,default is the str of root number_branch
        """
        if number_branch is None:
            number_branch = self.number_branch
        # print(number_branch)
        args = []
        jk = list(number_branch.keys())[0]
        ji = list(number_branch.values())[0]

        for jii in ji:
            if isinstance(jii, dict):
                repr1 = self.str_run(jii)
            else:
                repr1 = self[jii].name
            args.append(repr1)
        repr1 = self[jk].format(*args)

        return repr1

    def str_run_index(self, i):
        return self.str_run(self.number_branch_index(i))

    def indexs_in_node(self, i):
        """

        Returns
        -------
        indexs in node branch
        """

        def run_index(number_branch=None):
            if number_branch is None:
                number_branch = self.number_branch
            jk = list(number_branch.keys())[0]
            ji = list(number_branch.values())[0]
            sub_index = []
            for jii in ji:
                if isinstance(jii, dict):
                    repr1 = run_index(jii)
                else:
                    repr1 = [jii]
                sub_index.extend(repr1)
            sub_index.append(jk)

            return sub_index

        res = run_index(number_branch=self.number_branch_index(i))
        res = list(set(res))
        res.sort()
        return res


class ExpressionTree(PrimitiveTree):
    """

    """
    hasher = str

    def __init__(self, content):
        super(ExpressionTree, self).__init__(content)

    def __repr__(self):
        """Symbolic representation of the expression tree."""
        repr1 = ''
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                repr1 = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(repr1)
        return repr1

    def __hash__(self):
        return hash(self.hasher(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    @property
    def pri_site(self):
        return [i for i, primitive in enumerate(self) if primitive.arity >= 1]

    @property
    def ter_site(self):
        return [i for i, primitive in enumerate(self) if primitive.arity == 0]

    @property
    def primitives(self):
        """Return primitives that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity >= 1]


def sympyPrimitiveSet(rep_name, types="Fixed", categories=("Add", "Mul", "Abs", "exp"), power_categories=None,
                      partial_categories=None, self_categories=None, name=None, dim=None, max_=5,
                      definate_operate=None, definate_variable=None, operate_linkage=None, variable_linkage=None):
    """

    Parameters
    ----------
        :type partial_categories: double list
        partial_categories = [["Add","Mul"],["x4"]]
        :type power_categories: list
        index_categories=[0.5,1,2,3]
        :type dim: list,tuple
        :type name: list,tuple
        :type rep_name: list,tuple
        :type categories: list,tuple
        :param self_categories:
        def rem(a):
            return 1-a
        self_categories = [[rem, 1, 'rem']]
        :type definate_variable: list,tuple
        definate_variable = [(-1, [1, ]), ]
        :type definate_operate: list,tuple
        definate_operate = [(-1, [0, ]), ]
        :param types
        :max
    """

    def Div(left, right):
        return left / right

    def Sub(left, right):
        return left - right

    def zeroo(_):
        return 0

    def oneo(_):
        return 1

    def rem(a):
        return 1 - a

    def self(_):
        return _

    functions2 = {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div, 'Max': sympy.Max, "Min": sympy.Min}
    functions1 = {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'log': sympy.ln,
                  'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),
                  "Rec": functools.partial(sympy.Pow, e=-1.0),
                  'Zeroo': zeroo, "Oneo": oneo, "Rem": rem, "Self": self}

    pset0 = FixedPrimitiveSet('main') if types in ["fix", "fixed", "Fix", "Fixed"] else PrimitiveSet('main', 0)

    if power_categories:
        for j, i in enumerate(power_categories):
            pset0.addPrimitive(functools.partial(sympy.Pow, e=i), arity=1, name='pow%s' % j)

    for i in categories:
        if i in functions1:
            pset0.addPrimitive(functions1[i], arity=1, name=i)
        if i in functions2:
            pset0.addPrimitive(functions2[i], arity=2, name=i)

    if partial_categories:
        for partial_categoriesi in partial_categories:
            for i in partial_categoriesi[0]:
                for j in partial_categoriesi[1]:
                    if i in ["Mul", "Add"]:
                        pset0.addPrimitive(functools.partial(functions2[i], sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
                    else:
                        pset0.addPrimitive(functools.partial(functions2[i], right=sympy.Symbol(j)), arity=1,
                                           name="{}_{}".format(i, j))
    if self_categories:
        for i in self_categories:
            pset0.addPrimitive(i[0], i[1], i[2])

    # define terminal
    if isinstance(rep_name[0], str):
        rep_name = [sympy.Symbol(i) for i in rep_name]
    if dim is None:
        dim = [1] * len(rep_name)
    if name is None:
        name = rep_name

    assert len(dim) == len(name) == len(rep_name)

    for sym in rep_name:
        pset0.addTerminal(sym, name=str(sym))

    if types in ["fix", "fixed", "Fix", "Fixed"]:
        dict_pri = dict(zip([_.name for _ in pset0.primitives], range(len(pset0.primitives))))
        dict_ter = dict(zip([_.name for _ in pset0.terminals], range(len(pset0.terminals))))
    else:
        dict_pri = dict(zip([_.name for _ in pset0.primitives[object]], range(len(pset0.primitives))))
        dict_ter = dict(zip([_.name for _ in pset0.terminals[object]], range(len(pset0.terminals))))

    if max_ is None:
        max_ = len(pset0.terminals)

    # define limit
    def link_check(checking_linkage):
        """

        Parameters
        ----------
        checking_linkage

        Returns
        -------

        """
        if checking_linkage is None:
            checking_linkage = [[]]
        assert isinstance(checking_linkage, (list, tuple))
        if not isinstance(checking_linkage[0], (list, tuple)):
            checking_linkage = [checking_linkage, ]
        return checking_linkage

    operate_linkage = link_check(operate_linkage)
    variable_linkage = link_check(variable_linkage)

    operate_linkage = [[j - max_ for j in i] for i in operate_linkage]
    linkage = operate_linkage + variable_linkage

    if definate_operate:
        definate_operate = [list(i) for i in definate_operate]
        for i, j in enumerate(definate_operate):
            j = list(j)
            definate_operate[i][1] = [dict_pri[_] if _ in dict_pri else _ for _ in j[1]]
    if definate_variable:
        definate_variable = [list(i) for i in definate_variable]
        for i, j in enumerate(definate_variable):
            j = list(j)
            definate_variable[i][1] = [dict_ter[_] if _ in dict_ter else _ for _ in j[1]]

    pset0.definate_operate = definate_operate
    pset0.definate_variable = definate_variable
    pset0.linkage = linkage
    pset0.rep_name_list = rep_name
    pset0.name_list = name
    pset0.dim_list = dim
    pset0.max_ = max_
    print(dict_pri)
    print(dict_ter)

    return pset0


def produce(container, generator):
    """

    Parameters
    ----------
    container
    generator

    Returns
    -------

    """
    return container(generator())


def generate_(pset, min_=None, max_=None):
    """

    Parameters
    ----------
    pset
    min_
    max_

    Returns
    -------

    """
    if max_ is None:
        max_ = len(pset.terminals_and_constant)
    if min_ is None:
        min_ = max_

    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]

    max_varibale_set_long = max_
    varibale_set_long = random.randint(min_, max_varibale_set_long)
    '''random'''
    trem_set = random.sample(pset.terminals_and_constant, varibale_set_long) * 20
    '''sequence'''
    # trem_set = pset.terminals[:varibale_set_long] * 20

    init_operator_long = max_varibale_set_long * 3
    individual1 = []
    for i in range(init_operator_long):
        trem = random.choice(pri2) if random.random() > 0.5 * len(pri1) / len(
            pset.primitives) else random.choice(pri1)
        individual1.append(trem)
    individual2 = []
    for i in range(varibale_set_long):
        trem = trem_set[i]
        individual2.append(trem)
    # define protect primitives
    pri2 = [i for i in pset.primitives if i.arity == 2]
    protect_individual = []
    for i in range(varibale_set_long):
        trem = random.choice(pri2)
        protect_individual.append(trem)

    definate_operate = pset.definate_operate
    definate_variable = pset.definate_variable
    linkage = pset.linkage

    if definate_operate:
        for i in definate_operate:
            individual1[i[0]] = pset.primitives[random.choice(i[1])]

    if definate_variable:
        for i in definate_variable:
            individual2[i[0]] = pset.terminals_and_constant[random.choice(i[1])]

    individual_all = protect_individual + individual1 + individual2
    if linkage:
        for i in linkage:
            for _ in i:
                individual_all[_] = individual_all[i[0]]

    return individual_all


def compile_(expr_, pset):
    """

    Parameters
    ----------
    expr_
    pset

    Returns
    -------

    """
    code = str(expr_)
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


def sub(expr01, subed, subs):
    """"""
    listt = list(zip(subed, subs))
    return expr01.subs(listt)


def addCoefficient(expr01, inter_add=None, iner_add=None, random_add=None):
    """

    Parameters
    ----------
    expr01
    inter_add
    iner_add
    random_add

    Returns
    -------

    """

    def get_args(expr_):
        """

        Parameters
        ----------
        expr_

        Returns
        -------

        """
        list_arg = []
        for i in expr_.args:
            list_arg.append(i)
            if i.args:
                re = get_args(i)
                list_arg.extend(re)
        return list_arg

    arg_list = get_args(expr01)
    arg_list = [i for i in arg_list if i not in expr01.args]
    cho = []
    a_list = []
    #
    if isinstance(expr01, sympy.Add):
        for i, j in enumerate(expr01.args):
            Wi = sympy.Symbol("W%s" % i)
            expr01 = expr01.subs(j, Wi * j)
            a_list.append(Wi)
    else:
        A = sympy.Symbol("A")
        expr01 = expr01 * A
        a_list.append(A)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        a_list.append(B)

    if iner_add:
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        [cho.extend(i) for i in cho_add]

    if random_add:

        lest = [i for i in arg_list if i not in cho]
        if len(lest) != 0:
            cho2 = random.sample(lest, 1)
            cho.extend(cho2)
    # #
    a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
    for ai, choi in zip(a_cho, cho):
        expr01 = expr01.subs(choi, ai * choi)
    a_list.extend(a_cho)

    return expr01, a_list


def my_custom_loss_func(y_true, y_pred):
    """"""
    diff = - np.abs(y_true - y_pred) / y_true + 1
    return np.mean(diff)


mre_score = make_scorer(my_custom_loss_func, greater_is_better=True)


def calculateExpr(expr01, pset, x, y, score_method=r2_score, add_coeff=True,
                  del_no_important=False, filter_warning=True, terminals=None, **kargs):
    """

    Parameters
    ----------
    expr01
    pset
    x
    y
    score_method
    add_coeff
    del_no_important
    filter_warning
    terminals
    kargs

    Returns
    -------

    """
    if not terminals:
        terminals = pset.terminals_and_constant[object] if isinstance(pset.terminals_and_constant,
                                                                      defaultdict) else pset.terminals_and_constant
        terminals = [_.value for _ in terminals]
    if filter_warning:
        warnings.filterwarnings("ignore")

    expr00 = deepcopy(expr01)

    if not score_method:
        score_method = r2_score
    if add_coeff:
        expr01, a_list = addCoefficient(expr01, **kargs)
        try:
            func0 = sympy.utilities.lambdify(terminals + a_list, expr01)

            def func(x_, p):
                """

                Parameters
                ----------
                x_
                p

                Returns
                -------

                """
                num_list = []
                num_list.extend([*x_.T])
                num_list.extend(p)
                return func0(*num_list)

            def res(p, x_, y_):
                """

                Parameters
                ----------
                p
                x_
                y_

                Returns
                -------

                """
                return y_ - func(x_, p)

            result = optimize.least_squares(res, x0=[1] * len(a_list), args=(x, y), loss='huber', ftol=1e-4)

            cof = result.x
            cof_ = []
            for a_listi, cofi in zip(a_list, cof):
                if "A" or "W" in a_listi.name:
                    cof_.append(cofi)
                else:
                    cof_.append(np.round(cofi, decimals=3))
            cof = cof_
            for ai, choi in zip(a_list, cof):
                expr01 = expr01.subs(ai, choi)
        except (ValueError, NameError, TypeError, KeyError):
            expr01 = expr00
    else:
        pass

    try:
        if del_no_important and isinstance(expr01, sympy.Add) and len(expr01.args) >= 3:
            re_list = []
            for expri in expr01.args:
                if not isinstance(expri, sympy.Float):
                    func0 = sympy.utilities.lambdify(terminals, expri)
                    re = np.mean(func0(*x.T))
                    if abs(re) > abs(0.001 * np.mean(y)):
                        re_list.append(expri)
                else:
                    re_list.append(expri)
            expr01 = sum(re_list)
        else:
            pass

        func0 = sympy.utilities.lambdify(terminals, expr01)
        re = func0(*x.T)
        assert y.shape == re.shape
        assert_all_finite(re)
        check_array(re, ensure_2d=False)
        score = score_method(y, re)
    except (ValueError, DataConversionWarning, TypeError, NameError, KeyError, AssertionError, AttributeError):
        score = -0
    else:
        if np.isnan(score):
            score = -0
    return score, expr01


def calculate(individual, pset, x, y, score_method=r2_score, add_coeff=True, filter_warning=True,
              **kargs):
    """

    Parameters
    ----------
    individual
    pset
    x
    y
    score_method
    add_coeff
    filter_warning
    kargs

    Returns
    -------

    """
    # '''1 not expand'''
    expr_no = sympy.sympify(compile_(individual, pset))
    # '''2 expand by sympy.expand,long expr is slow, use if when needed'''
    # expr_no = sympy.expand(compile_(individual, pset), deep=False, power_base=False, power_exp=False, mul=True,
    #                        log=False, multinomial=False)
    # '''3 expand specially '''
    if isinstance(expr_no, sympy.Mul) and len(expr_no.args) == 2:
        if isinstance(expr_no.args[0], sympy.Add) and expr_no.args[1].args == ():
            expr_no = sum([i * expr_no.args[1] for i in expr_no.args[0].args])
        if isinstance(expr_no.args[1], sympy.Add) and expr_no.args[0].args == ():
            expr_no = sum([i * expr_no.args[0] for i in expr_no.args[1].args])
        else:
            pass
    score, expr = calculateExpr(expr_no, pset, x, y, score_method=score_method, add_coeff=add_coeff,
                                filter_warning=filter_warning, **kargs)
    return score, expr


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, pset=None, store=True):
    """

    Parameters
    ----------
    population
    toolbox
    cxpb
    mutpb
    ngen
    stats
    halloffame
    verbose
    pset
    store

    Returns
    -------

    """
    len_pop = len(population)
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    random_seed = random.randint(1, 1000)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fitnesses = parallelize(n_jobs=4, func=toolbox.evaluate, iterable=invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0],
        ind.expr = fit[1]

    if halloffame is not None:
        halloffame.update(population)
    random.seed(random_seed)
    record = stats.compile_(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    data_all = {}
    # Begin the generational process
    for gen in range(1, ngen + 1):

        if store:
            if pset:
                subp = partial(sub, subed=pset.rep_name_list, subs=pset.name_list)
                data = [{"score": i.fitness.values[0], "expr": subp(i.expr)} for i in halloffame.items[-5:]]
            else:
                data = [{"score": i.fitness.values[0], "expr": i.expr} for i in halloffame.items[-5:]]
            data_all['gen%s' % gen] = data
        # select_gs the next generation individuals
        offspring = toolbox.select_gs(population, len_pop)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        if halloffame is not None:
            offspring.extend(halloffame)

        random_seed = random.randint(1, 1000)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fitnesses = parallelize(n_jobs=4, func=toolbox.evaluate, iterable=invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.expr = fit[1]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

            if halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                break
        random.seed(random_seed)
        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile_(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    store = Store()
    store.to_txt(data_all)
    return population, logbook


def multiEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
                  halloffame=None, verbose=__debug__, pset=None, store=True, alpha=1):
    """

    Parameters
    ----------
    population
    toolbox
    cxpb
    mutpb
    ngen
    stats
    halloffame
    verbose
    pset
    store
    alpha

    Returns
    -------

    """
    logbook = Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    random_seed = random.randint(1, 1000)
    # fitnesses = list(toolbox.map(toolbox.evaluate, [str(_) for _ in invalid_ind]))
    # fitnesses2 = toolbox.map(toolbox.evaluate2, [str(_) for _ in invalid_ind])
    fitnesses = parallelize(n_jobs=6, func=toolbox.evaluate, iterable=[str(_) for _ in invalid_ind])
    fitnesses2 = parallelize(n_jobs=6, func=toolbox.evaluate2, iterable=[str(_) for _ in invalid_ind])

    def funcc(a, b):
        """

        Parameters
        ----------
        a
        b

        Returns
        -------

        """
        return (alpha * a + b) / 2

    for ind, fit, fit2 in zip(invalid_ind, fitnesses, fitnesses2):
        ind.fitness.values = funcc(fit[0], fit2[0]),
        ind.values = (fit[0], fit2[0])
        ind.expr = (fit[1], fit2[1])
    if halloffame is not None:
        halloffame.update(population)
    random.seed(random_seed)
    record = stats.compile_(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    data_all = {}
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # select_gs the next generation individuals
        offspring = toolbox.select_gs(population, len(population))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        if halloffame is not None:
            offspring.extend(halloffame.items[-2:])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        random_seed = random.randint(1, 1000)
        # fitnesses = toolbox.map(toolbox.evaluate, [str(_) for _ in invalid_ind])
        # fitnesses2 = toolbox.map(toolbox.evaluate2, [str(_) for _ in invalid_ind])
        fitnesses = parallelize(n_jobs=6, func=toolbox.evaluate, iterable=[str(_) for _ in invalid_ind])
        fitnesses2 = parallelize(n_jobs=6, func=toolbox.evaluate2, iterable=[str(_) for _ in invalid_ind])

        for ind, fit, fit2 in zip(invalid_ind, fitnesses, fitnesses2):
            ind.fitness.values = funcc(fit[0], fit2[0]),
            ind.values = (fit[0], fit2[0])
            ind.expr = (fit[1], fit2[1])

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            if halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                print(halloffame.items[-1].values[0])
                print(halloffame.items[-1].values[1])
                break

        if store:
            if pset:
                subp = partial(sub, subed=pset.rep_name_list, subs=pset.name_list)
                data = [{"score": i.values[0], "expr": subp(i.expr[0])} for i in halloffame.items[-2:]]
                data2 = [{"score": i.values[1], "expr": subp(i.expr[1])} for i in halloffame.items[-2:]]
            else:
                data = [{"score": i.values[0], "expr": i.expr} for i in halloffame.items[-2:]]
                data2 = [{"score": i.values[1], "expr": i.expr[2]} for i in halloffame.items[-2:]]
            data_all['gen%s' % gen] = list(zip(data, data2))
        random.seed(random_seed)
        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile_(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    if store:
        store1 = Store()
        store1.to_txt(data_all)

    return population, logbook


def getName(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    if isinstance(x, pd.DataFrame):
        name = x.columns.values
        name = [sympy.Symbol(i) for i in name]
        rep_name = [sympy.Symbol("x%d" % i) for i in range(len(name))]

    elif isinstance(x, np.ndarray):
        check_array(x)
        name = x.shape[1]
        name = [sympy.Symbol("x%d" % i) for i in range(name)]
        rep_name = [sympy.Symbol("x%d" % i) for i in range(len(name))]
    else:
        raise TypeError("just support np.ndarray and pd.DataFrame")

    return name, rep_name


def cxOnePoint_index(ind1, ind2, pset):
    """

    Parameters
    ----------
    ind1
    ind2
    pset

    Returns
    -------

    """
    linkage = pset.linkage
    root = max(ind1.root, ind2.root)
    index = random.randint(root, root + len(ind1.pri_site))
    ind10 = copy.copy(ind1)
    ind20 = copy.copy(ind2)
    ind10[index:] = ind2[index:]
    ind20[index:] = ind1[index:]
    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]
                ind20[_] = ind20[i[0]]
    return ind10, ind20


def mutUniForm_index(ind1, pset, ):
    """

    Parameters
    ----------
    ind1
    pset

    Returns
    -------

    """

    ind10 = copy.copy(ind1)
    linkage = pset.linkage
    pri2 = [i for i in pset.primitives if i.arity == 2]
    pri1 = [i for i in pset.primitives if i.arity == 1]
    index = random.choice(ind10.pri_site)
    ind10[index] = random.choice(pri2) if random.random() > 0.5 * len(pri1) / len(
        pset.primitives) else random.choice(pri1)

    definate_operate = pset.definate_operate
    ranges = list(range(ind1.pri_site[-1]))
    if definate_operate:
        for i in definate_operate:
            ind10[ranges[i[0]]] = pset.primitives[random.choice(i[1])]

    if linkage:
        for i in linkage:
            for _ in i:
                ind10[_] = ind10[i[0]]
    return ind10,


def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals_and_constant[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The symbol.generate function tried to add " \
                                 "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The symbol.generate function tried to add " \
                                 "a primitive of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def cxOnePoint(ind1, ind2):
    """Randomly select_gs in each individual and exchange each subtree with the
    point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.symbol.genGrow`,
    the other half, the expression is generated with :func:`~deap.symbol.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        term = random.choice(pset.terminals_and_constant[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


def selRandom(individuals, k):
    """select_gs *k* individuals at random from the input0 *individuals* with
    replacement. The list returned contains references to the input0
    *individuals*.

    :param individuals: A list of individuals to select_gs from.
    :param k: The number of individuals to select_gs.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """select_gs the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input0 *individuals*.

    :param individuals: A list of individuals to select_gs from.
    :param k: The number of individuals to select_gs.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input0 population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


@time_this_function
def mainPart(x_, y_, pset, pop_n=100, random_seed=1, cxpb=0.8, mutpb=0.1, ngen=5, alpha=1,
             tournsize=3, max_value=10, double=False, score=None, **kargs):
    """

    Parameters
    ----------
    score
    double
    x_
    y_
    pset
    pop_n
    random_seed
    cxpb
    mutpb
    ngen
    alpha
    tournsize
    max_value
    kargs

    Returns
    -------

    """
    max_ = pset.max_
    if score is None:
        score = [r2_score, explained_variance_score]
    random.seed(random_seed)
    toolbox = Toolbox()
    if isinstance(pset, PrimitiveSet):
        PTrees = ExpressionTree
        Generate = genHalfAndHalf
        mutate = mutNodeReplacement
        mate = cxOnePoint
    elif isinstance(pset, FixedPrimitiveSet):
        PTrees = FixedExpressionTree
        Generate = generate_
        mate = partial(cxOnePoint_index, pset=pset)
        mutate = mutUniForm_index
    else:
        raise NotImplementedError("get wrong pset")
    if double:
        creator.create("Fitness_", Fitness, weights=(1.0, 1.0))
    else:
        creator.create("Fitness_", Fitness, weights=(1.0,))
    creator.create("PTrees_", PTrees, fitness=creator.Fitness_)
    toolbox.register("generate_", Generate, pset=pset, min_=None, max_=max_)
    toolbox.register("individual", initIterate, container=creator.PTrees_, generator=toolbox.generate_)
    toolbox.register('population', initRepeat, container=list, func=toolbox.individual)
    # def selection
    toolbox.register("select_gs", selTournament, tournsize=tournsize)
    # def mate
    toolbox.register("mate", mate)
    # def mutate
    toolbox.register("mutate", mutate, pset=pset)
    if isinstance(pset, PrimitiveSet):
        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    # def elaluate
    toolbox.register("evaluate", calculate, pset=pset, x=x_, y=y_, score_method=score[0], **kargs)
    toolbox.register("evaluate2", calculate, pset=pset, x=x_, y=y_, score_method=score[1], **kargs)

    stats1 = Statistics(lambda ind: ind.fitness.values[0])
    stats = MultiStatistics(score1=stats1, )
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop = toolbox.population(n=pop_n)

    haln = 5
    hof = HallOfFame(haln)

    if double:
        population, logbook = multiEaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, alpha=alpha,
                                            halloffame=hof, pset=pset)
    else:
        population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                       halloffame=hof, pset=pset)

    return population, logbook, hof
