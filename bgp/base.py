#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: GNU Lesser General Public License v3.0

"""
Base objects for symbolic regression.

Contains:
  - Class: ``SymbolSet``

  - Class: ``CalculatePrecisionSet``

  - Class: ``SymbolTree``

  - others
"""

import copy
import functools
from collections.abc import Iterable

import numpy as np
import sympy
from mgetool.tool import parallelize, batch_parallelize
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y, check_array

from bgp.calculation.coefficient import try_add_coef_times
from bgp.calculation.scores import calcualte_dim_score, compile_context, calculate_cv_score, score_collection, \
    calculate_collect_
from bgp.calculation.translate import group_str
from bgp.functions.dimfunc import dim_map, Dim, dnan, dless
from bgp.functions.gsymfunc import gsym_map, NewArray
from bgp.functions.npfunc import np_map
from bgp.functions.symfunc import sym_vector_map, sym_dispose_map
from bgp.gp import genGrow, genFull, depart
from bgp.probability.preference import PreMap


class SymbolTerminal:
    """General feature type, do not use directly.\n
    The name for show (str) and calculation (repr) are set to different string for
    avoiding repeated calculations.
    """

    def __init__(self, name, init_name=None):
        """

        Parameters
        ----------
        name: str
            Represent name. Default "xi".
        init_name: str
            Just for show, rather than calculate.\n
            Examples:
                init_name=[x1, x2] , if is compact features, need[].\n
                init_name=(x1*x4-x3), if is expr, need ().
        """
        self.name = str(name)
        self.conv_fct = str
        self.arity = 0
        if init_name is None:
            self.init_name = None
        else:
            self.init_name = str(init_name)

    def format_repr(self):
        # short, repr
        """representing name"""
        return self.conv_fct(self.name)

    def format_str(self):
        # long.str
        """represented name"""
        if self.init_name:
            return self.init_name
        else:
            return self.name

    def __str__(self):
        """represented name"""
        if self.init_name:
            return self.init_name
        else:
            return self.name

    def __repr__(self):
        """represent name"""
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(repr(self))


class SymbolTerminalDetail(SymbolTerminal):
    """General feature type.\n
    The name for show (str) and calculation (repr) are set to different string for
    avoiding repeated calculations.
    """

    def __init__(self, values, name, dim=None, prob=None, init_sym=None, init_name=None):
        """

        Parameters
        ----------
        values: None, number or np.ndarray
            xi value, the shape can be (n, ) or (n_x, n), 
            n is number of samples, n_x is numbers of feature.
        name: str
            Represent name. Default "xi"
        dim: bgp.dim.Dim or None
            None.
        prob: float or None
            None.
        init_sym: list, sympy.Expr
            list.
        init_name: str or None 
            Just for show, rather than calculate.\n
            Examples:
                init_name="[x1, x2]" , if is compact features, need[].\n
                init_name="(x1*x4-x3)", if is expr, need ().
        """
        super(SymbolTerminalDetail, self).__init__(name, init_name)
        if prob is None:
            prob = 1
        if dim is None:
            dim = dless
        self.value = values
        self.sym = sympy.Symbol(str(name))
        self.init_sym = init_sym
        self.dim = dim
        self.prob = prob

    def capsule(self):
        return SymbolTerminal(self.name, self.init_name)


def _tsum(*ters, name="gx0"):
    """

    Parameters
    ----------
    ters: tuple of SymbolTerminalDetail
        SymbolTerminalDetail
    name: str
        specific the name of results.

    Returns
    -------
    SymbolTerminalDetail
    """

    for i in ters:
        assert isinstance(i, SymbolTerminalDetail), "only the SymbolTerminals can be added"
    assert all(ters[0].dim == i.dim for i in ters)
    values = np.array([i.value for i in ters])
    dim = Dim(np.array([i.dim for i in ters]))
    sym = NewArray([i.sym for i in ters], shape=(len(ters),))
    name = str(name)
    prob = sum([i.prob for i in ters]) / len(ters)
    res = SymbolTerminalDetail(values, name, dim=dim, prob=prob,
                               init_sym=sym, init_name=str(list(sym)))
    return res


class SymbolPrimitive:
    """General operator type, do not use directly,
    but use SymbolPrimitiveDetail."""

    def __init__(self, name, arity):
        """

        Parameters
        ----------
        name: str
            function name.
        arity: int
            input parameters numbers of function.
            such as ``+`` with 2, ``ln`` with 1.

        """
        self.name = str(name)
        self.arity = arity
        self.args = list(range(arity))

        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format_str(self, *args):
        return self.seq.format(*args)

    format_repr = format_str  # for function the format for machine and user is the same.

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return self.name

    __repr__ = __str__  # for function the format for machine and user is the same.


class SymbolPrimitiveDetail(SymbolPrimitive):
    """
    General operator type with more details.
    """

    def __init__(self, name, arity, func, prob, np_func=None, dim_func=None, sym_func=None):
        """
        Parameters
        ----------

        func: Callable
            function. better using sympy.Function Type.\n
            For Maintainer:
                If self function and can not be simplified to sympy.Function or elementary function, 
                the function for function.np_map() and dim.dim_map() should be defined.
        name: str
            function name.
        arity: int
            function input numbers.
        prob: float
            default 1.
        """
        super(SymbolPrimitiveDetail, self).__init__(name, arity)

        if prob is None:
            prob = 1
        self.func = func
        self.prob = prob
        self.np_func = np_func
        self.dim_func = dim_func
        self.sym_func = sym_func

    def capsule(self):
        """return short one."""
        return SymbolPrimitive(self.name, self.arity)


class SymbolSet(object):
    """
    Definite the preparation set of operations, features, and fixed constants.
    """

    def __init__(self, name="PSet"):
        """

        Parameters
        ----------
        name: str
            name.
        """
        self.arguments = []  # for translate
        self.name = name
        self.y = None  # data y
        self.y_dim = dless  # dim y

        self.data_x_dict = {}  # data x

        self.new_num = 0

        self.terms_count = 0
        self.prims_count = 0
        self.constant_count = 0
        self.dispose_count = 0

        self.context = {"__builtins__": None}  # all elements map

        self.dim_map = dim_map()
        self.np_map = np_map()
        self.gsym_map = gsym_map()

        self.primitives_dict = {}
        self.prob_pri = {}  # probability of operation default is 1

        self.dispose_dict = {}
        self.prob_dispose = {}  # probability of  structure operation, default is 1/n

        self.ter_con_dict = {}  # term and const
        self.dim_ter_con = {}  # Dim of and features and constants
        self.prob_ter_con = {}  # probability of and features and constants

        self.gro_ter_con = {}  # for group size calculation and simple

        self.terminals_init_map = {}  # for Tree show
        # terminals representing name "gx0" to represented name "[x1, x2]", 
        # or "newx0" to represented name "Add(Mul(x2, x4)+[x1, x2])".

        self.terminals_symbol_map = {}  # for Tree show
        # terminals representing name "gx0" to represented name "[x1, x2]", 
        # or "newx0" to represented name "Add(Mul(x2, x4)+[x1, x2])".

        self.expr_init_map = {}  # for expr show
        # terminals representing name "newx0" to represented name "(x2*x4+gx0)"
        self.terminals_fea_map = {}  # for terminals Latex feature name show.

        self.premap = PreMap.from_shape(3)
        self.x_group = [[]]

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def _add_primitive(self, func, name, arity, prob=None, np_func=None,
                       dim_func=None, sym_func=None):

        """
        Parameters
        ----------
        name: str
            function name
        func: Callable
            function. Better for sympy.Function Type.
            If self function and can not be simplified to sympy.Function or elementary function, 
            the function for np_func and dim_func should be defined.
        arity: int
            function input numbers
        prob: float
            default 1
        np_func: Callable
            numpy function or function constructed by numpy function
        dim_func: Callable
            function to calculate Dim
        sym_func: Callable
            function to calculate group sympy.Expr
        """

        if prob is None:
            prob = 1

        if name is None:
            name = func.__name__

        assert name not in self.primitives_dict, "Primitives are required to have a unique func. " \
                                                 "Consider  rename your second '%s' primitive." % (name,)

        self.primitives_dict[name] = SymbolPrimitiveDetail(name, arity, func, prob=prob,
                                                           np_func=np_func, dim_func=dim_func,
                                                           sym_func=sym_func)

        self.prims_count += 1

    def _add_dispose(self, func, name, arity=1, prob=None, np_func=None, dim_func=None, sym_func=None):
        """
        Parameters
        ----------
        name: str
            function name
        func: Callable
            function. Better for sympy.Function Type.
            If self function and can not be simplified to sympy.Function or elementary function, 
            the function for np_func and dim_func should be defined.
        arity: 1
            function input numbers, must be 1
        prob: float
            default 1/n, n is structure function number.
        np_func: Callable
            numpy function or function constructed by numpy function
        dim_func: Callable
            function to calculate Dim
        sym_func: Callable
            function to calculate group sympy.Expr
        """

        if prob is None:
            prob = 1

        if name is None:
            name = func.__name__

        self.dispose_dict[name] = SymbolPrimitiveDetail(name, arity, func, prob=prob,
                                                        np_func=np_func, dim_func=dim_func,
                                                        sym_func=sym_func)
        self.dispose_count += 1

    def _add_terminal(self, value, name, dim=None, prob=None, init_sym=None, init_name=None):
        """
        Parameters
        ----------
        name: str
            function name
        value: numpy.ndarray, ndarray
            xi value
        prob: float
            default 1
        init_name: str
            True name can be found of input. Just for show, rather than calculate.
            Examples:
            init_name="[x1, x2]" , if is compact features, need[]
            init_name="(x1*x4-x3)", if is expr, need ()
        dim: Dim
            xi Dim
        """

        if prob is None:
            prob = 1
        if dim is None:
            dim = dless

        self.ter_con_dict[name] = SymbolTerminalDetail(value, name, dim=dim, prob=prob,
                                                       init_sym=init_sym,
                                                       init_name=init_name)
        self.terms_count += 1

    def register(self, primitives_dict="all", dispose_dict="all", ter_con_dict="all"):
        """
        Register and capsule for simplify.
        
        Parameters
        ----------
        primitives_dict:None, str, dict
        
        dispose_dict:None, str, dict
        
        ter_con_dict:None, str, dict

        """
        if primitives_dict == "all":
            primitives_dict = self.primitives_dict
        if dispose_dict == "all":
            dispose_dict = self.dispose_dict
        if ter_con_dict == "all":
            ter_con_dict = self.ter_con_dict

        if primitives_dict is not None:
            for p in primitives_dict.values():
                if p.np_func is not None:
                    self.np_map[p.name] = p.np_func
                if p.sym_func is not None:
                    self.gsym_map[p.name] = p.sym_func
                if p.dim_func is not None:
                    self.dim_map[p.name] = p.dim_func
                self.prob_pri[p.name] = p.prob
                self.context[p.name] = p.func
                self.primitives_dict[p.name] = self.primitives_dict[p.name].capsule()

        if dispose_dict is not None:
            for p in dispose_dict.values():
                if p.np_func is not None:
                    self.np_map[p.name] = p.np_func
                if p.sym_func is not None:
                    self.gsym_map[p.name] = p.sym_func
                if p.dim_func is not None:
                    self.dim_map[p.name] = p.dim_func
                self.prob_dispose[p.name] = p.prob
                self.context[p.name] = p.func
                self.dispose_dict[p.name] = self.dispose_dict[p.name].capsule()

        if ter_con_dict is not None:
            for t in ter_con_dict.values():

                if t.dim.ndim == 1:
                    ng = 1
                elif t.dim.ndim == 2:
                    ng = t.dim.shape[0]
                else:
                    ng = 1

                self.gro_ter_con[t.name] = ng
                self.dim_ter_con[t.name] = t.dim
                self.prob_ter_con[t.name] = t.prob
                self.data_x_dict[t.name] = t.value

                self.context[t.name] = sympy.Symbol(t.name)

                if t.init_name:
                    self.terminals_init_map[t.name] = t.init_name
                    if isinstance(t.init_sym, (np.ndarray, NewArray)):
                        self.terminals_symbol_map[t.name] = t.init_sym
                    else:
                        self.expr_init_map[t.name] = t.init_sym
                self.ter_con_dict[t.name] = self.ter_con_dict[t.name]

    def _group(self, x_group=None):
        if not x_group:
            x_group = self.x_group
        if isinstance(x_group, int):
            assert self.terms_count > x_group > 1, "the len of group should in (2, x.shape[1]]"
            indexes = [_ for _ in range(self.terms_count)]
            x_group = [indexes[i:i + x_group] for i in range(0, len(indexes), x_group)]

        x_group = [x_groupi for x_groupi in x_group if len(x_groupi) >= 2]

        for i, gi in enumerate(x_group):
            len_gi = len(gi)
            if len_gi > 0:
                init_ter_name = ["x%s" % j for j in gi]
                init_ter = [self.ter_con_dict.pop(i) for i in init_ter_name]
                name = "gx%s" % i
                ter = _tsum(*init_ter, name=name)
                self.ter_con_dict[name] = ter

                self.terms_count -= (len(init_ter) - 1)

    def _add_constant(self, value, name=None, dim=None, prob=None):
        """
        Parameters
        ----------
        name: None, str
            function name
        value: numpy.ndarray or float
            ci value
        prob: float
            default 0.5
        dim: Dim
            ci Dim
        """

        if prob is None:
            prob = 1
        if dim is None:
            dim = dless

        if name is None:
            name = "c%s" % self.constant_count

        self.ter_con_dict[name] = SymbolTerminalDetail(value, sympy.Symbol(name), dim=dim, prob=prob)

        self.constant_count += 1

    def add_operations(self, power_categories=None, categories=None, self_categories=None,
                       power_categories_prob="balance", categories_prob="balance", special_prob=None):
        """
        Add operations with probability.

        Parameters
        ----------
        power_categories: Sized, tuple, None
            Examples:
                (0.5, 2, 3)
        categories: tuple of str
            map table:
                {'Add': sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}
                {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'ln': sympy.ln, }
                {'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0), }
                "Rec": functools.partial(sympy.Pow, e=-1.0)}

                Others:  \n
                "Rem":  f(x)=1-x, if x true \n
                "Self":  f(x)=x, if x true \n
        categories_prob: "balance", float
            probability of categories, except (+, -*, /), in (0, 1].
            "balance" is 1/n_categories.
            The (+, -*, /) are set as 1 to be a standard.
        special_prob: None, dict
            prob for special name.\n
            Examples:{"Mul":0.6, "Add":0.4, "exp":0.1}
        power_categories_prob: "balance", float
            float in (0, 1].
            probability of power categories, "balance" is 1/power_categories_prob.
        self_categories:list of dict, None
            the dict can be generate from newfuncV or definition self.\n
            the function at least containing:
            {"func": func, "name": name, "arity":2, "np_func": npf, "dim_func": dimf, "sym_func": gsymf}

            1.func:sympy.Function(name) object

            2.name:name

            3.arity:int, the number of parameter

            4.np_func:numpy function

            5.dim_func:dimension function

            6.sym_func:NewArray function. (unpack the group, used just for shown)

            See Also bgp.newfunc.newfuncV
        """

        """        



"""
        if categories is None:
            categories = ("Add", "Mul", "Self", "exp")

        def change(n, p):
            if isinstance(special_prob, dict):
                if n in special_prob:
                    p = special_prob[n]
            return p

        if "MAdd" not in self.dispose_dict or "Self" not in self.dispose_dict:
            self.add_accumulative_operation()

        functions1, functions2 = sym_vector_map()
        if power_categories:
            if power_categories_prob == "balance":
                prob = 1 / len(power_categories)
            elif isinstance(power_categories_prob, float):
                prob = power_categories_prob
            else:
                raise TypeError("power_categories_prob accept float from (0, 1] or 'balance'.")
            for j, i in enumerate(power_categories):
                name = 'pow%s' % j
                prob = change(name, prob)
                self._add_primitive(functools.partial(sympy.Pow, e=i),
                                    arity=1, name='pow%s' % j, prob=prob)

        for i in categories:
            if categories_prob == "balance":
                ca_new = [_ for _ in categories if _ not in ("Add", 'Sub' 'Mul', 'Div')]
                if len(ca_new) >= 1:
                    prob1 = 1 / len(ca_new)
                else:
                    prob1 = 1
            elif isinstance(categories_prob, float):
                prob1 = categories_prob
            else:
                raise TypeError("categories_prob accept float from (0, 1] or 'balance'.")
            if i in functions1:
                prob1 = change(i, prob1)
                self._add_primitive(functions1[i], arity=1, name=i, prob=prob1)
            if i in functions2:
                prob2 = change(i, 1)
                self._add_primitive(functions2[i], arity=2, name=i, prob=prob2)

        if self_categories:
            for i in self_categories:
                prob = change(i, 0.2)
                i["prob"] = prob
                i["arity"] = 1
                self._add_primitive(**i)
        self.register(primitives_dict="all", dispose_dict=None, ter_con_dict=None)
        return self

    def add_accumulative_operation(self, categories=None, categories_prob="balance",
                                   self_categories=None, special_prob=None):
        """
        add accumulative operation.

        Parameters
        ----------
        categories: tuple of str
            categories=("Self", "MAdd", "MSub", "MMul", "MDiv")
        categories_prob: None, "balance" or float.
            probility of categories in (0, 1], except ("Self", "MAdd", "MSub", "MMul", "MDiv"),

            "balance" is 1/n_categories.

            "MSub", "MMul", "MDiv" are only worked on the size of group is 2, else work like "Self".

            Notes:
                the  ("Self", "MAdd", "MSub", "MMul", "MDiv") are set as 1 to be a standard.
        self_categories:list of dict, None
            the dict can be generate from newfuncD or defination self.

            the function at least containing:

            {"func": func, "name": name, "np_func": npf, "dim_func": dimf, "sym_func": gsymf}

            1.func:sympy.Function(name) object, which need add attributes: is_jump, keep.

            2.name:name

            3.np_func:numpy function

            4.dim_func:dimension function

            5.sym_func:NewArray function. (unpack the group, used just for shown)

            See Also bgp.newfunc.newfuncV

        special_prob: None or dict
            Examples:
                {"MAdd":0.5, "Self":0.5}
        """

        def change(n, pp):
            if isinstance(special_prob, dict):
                if n in special_prob:
                    pp = special_prob[n]
            return pp

        if not categories:
            if self.types == 1:
                categories = ["Self"]
            elif self.types == 2:  # classification detail
                categories = ["Self", "MAdd", "MSub", "MMul", "MDiv", "Conv"]
            else:
                categories = ["Self", "MAdd", "MMul"]
            # else:
            #     categories = ["Self", "MAdd", "MSub", "MMul", "MDiv", "Conv"]
        if isinstance(categories, str):
            categories = [categories, ]

        if categories_prob == "balance":
            ca_new = [_ for _ in categories if _ not in ("Self", 'Flat', "MSub", "MMul",
                                                         "MDiv", "Conv")]
            if len(ca_new) >= 1:
                prob1 = 1.0 / len(ca_new)
            else:
                prob1 = 1.0
        elif isinstance(categories_prob, float):
            prob1 = categories_prob
        else:
            raise TypeError("categories_prob accept float from (0, 1] or 'balance'.")

        for i in categories:

            if i == "Self":
                p = change(i, 0.5)
                self._add_dispose(sym_dispose_map()[i], arity=1, name=i, prob=p)
            elif i in ("MAdd", "MSub", "MMul", "MDiv"):
                p = change(i, 0.1)
                self._add_dispose(sym_dispose_map()[i], arity=1, name=i, prob=p)
            elif i == "Conv":
                p = change(i, 0.05)
                self._add_dispose(sym_dispose_map()[i], arity=1, name=i, prob=p)
            else:
                # to be add for future
                p = change(i, prob1)
                self._add_dispose(sym_dispose_map()[i], arity=1, name=i, prob=p)

        if self_categories:
            for i in self_categories:
                prob = change(i, 0.2)
                i["prob"] = prob
                i["arity"] = 1
                self._add_dispose(*i)
        self.register(primitives_dict=None, dispose_dict="all", ter_con_dict=None)
        return self

    def add_tree_to_features(self, Tree, prob=0.3):
        """
        Add the individual as a new feature to initial features.
        not sure add success, because the value and name should be check and
        different to exist.

        Parameters
        ----------
        Tree: SymbolTree
            individual or expression
        prob: int
            probability of this individual
        """
        try:
            value = Tree.pre_y
            check_array(value, ensure_2d=False)

            eq = any([np.all(np.equal(value, i)) for i in self.data_x_dict.values()])
            if eq:
                raise ValueError

            assert Tree.y_dim is not dnan
            dim = Tree.y_dim

            init_name0 = str("(%s)" % Tree)

            t_map_va = list(self.terminals_init_map.keys())
            t_map_va.reverse()
            for i in t_map_va:
                init_name0 = init_name0.replace(i, self.terminals_init_map[i])

            if init_name0 in self.terminals_init_map.values():
                raise NameError

            init_name1 = Tree.expr

            # self.expr are not passed
            t_map_va = list(self.expr_init_map.keys())
            t_map_va.reverse()
            for i in t_map_va:
                init_name1 = init_name1.subs(sympy.Symbol(i), self.expr_init_map[i])

        except(AssertionError, NameError, ValueError, TypeError):
            pass
        else:
            name = "new%s" % self.new_num
            Tree.p_name = name
            self._add_terminal(value, name, dim=dim, prob=prob,
                               init_sym=init_name1, init_name=init_name0)
            self.premap = self.premap.add_new()
            self.new_num += 1
            self.register(primitives_dict=None, dispose_dict=None,
                          ter_con_dict={name: self.ter_con_dict[name]})

    def add_features(self, X, y, x_dim=1, y_dim=1, x_prob=None, x_group=None,
                     feature_name=None, ):

        """
        Add features with dimension and probability.

        Parameters
        ----------
        X: np.ndarray
            2D data.
        y: np.ndarray
            1D data.
        feature_name: None, list of str
            the same size wih x.shape[1].
        x_dim: 1 or list of Dim
            the same size wih x.shape[1], default 1 is dless for all x.
        y_dim: 1, Dim
            dim of y.
        x_prob: None, list of float
            the same size wih x.shape[1].
        x_group: None or list of list, int
            features group.

        """
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X, y = check_X_y(X, y)

        # define terminal
        n = X.shape[1]
        self.y = y.ravel()

        if y_dim == 1:
            y_dim = dless
        self.y_dim = y_dim

        if x_dim == 1:
            x_dim = [dless for _ in range(n)]

        if x_prob is None:
            x_prob = [1.0 for _ in range(n)]
        elif isinstance(x_prob, (float, int)):
            x_prob = [float(x_prob) for _ in range(n)]
        if isinstance(feature_name, list):
            assert n == len(x_dim) == len(feature_name) == len(x_prob)
        else:
            assert n == len(x_dim) == len(x_prob)

        for i in range(n):
            self._add_terminal(np.array(X.T[i]), name="x%s" % i, dim=x_dim[i], prob=x_prob[i])

        if isinstance(feature_name, list):
            for i, j in enumerate(feature_name):
                self.terminals_fea_map["x%s" % i] = j
        if x_group is None:
            x_group = [[]]
        self.x_group = x_group
        self._group(x_group)

        self.register(primitives_dict=None, dispose_dict=None, ter_con_dict="all")
        self.premap = PreMap.from_shape(len(self.ter_con_dict))
        return self

    def replace(self, X, y=None, tree_X=None):
        X = X.astype(np.float32)
        if y is not None:
            y = y.astype(np.float32)
            X, y = check_X_y(X, y)

        self.y = y.ravel()

        old = self.terms_count
        self.terms_count = 0
        # self.ter_con_dict={}
        n = X.shape[1]
        for i in range(n):
            self._add_terminal(np.array(X.T[i]), name="x%s" % i)

        self._group(self.x_group)

        if isinstance(tree_X, np.ndarray):
            tree_X = tree_X.astype(np.float32)
            n = tree_X.shape[1]
            for i in range(n):
                self._add_terminal(np.array(tree_X.T[i]), name="new%s" % i)

        elif isinstance(tree_X, dict):
            for k, v in tree_X.items():
                v = v.astype(np.float32)
                self._add_terminal(np.array(v), name=v)

        assert old == self.terms_count, "the new X (test, predict) should be with the " \
                                        "same shape[1] with old X (fit, train). when use re_tree, " \
                                        "the tree_X should be offered"

        self.register(primitives_dict=None, dispose_dict=None, ter_con_dict="all")
        if y is not None:
            self.y = y
        return self

    def add_constants(self, c, c_dim=1, c_prob=None):
        """
        Add features with dimension and probability.

        Parameters
        ----------
        c_dim: 1, list of Dim
            the same size wih c.
        c: float, list
            list of float.
        c_prob: None, float, list of float
            the same size with c.
        """
        if isinstance(c, float):
            c = [c, ]

        n = len(c)

        if c_dim == 1:
            c_dim = [dless for _ in range(n)]

        if c_prob is None:
            c_prob = [0.1 for _ in range(n)]
        elif isinstance(c_prob, (float, int)):
            c_prob = [c_prob for _ in range(n)]

        assert len(c) == len(c_dim) == len(c_prob)

        for v, dimi, probi in zip(c, c_dim, c_prob):
            self._add_constant(v, name=None, dim=dimi, prob=probi)

        self.register(primitives_dict=None, dispose_dict=None, ter_con_dict="all")
        self.premap = PreMap.from_shape(len(self.ter_con_dict))
        # re-generate each time.
        return self

    def add_features_and_constants(self, X, y, c=None, x_dim=1, y_dim=1, c_dim=1, x_prob=None,
                                   c_prob=None, x_group=None, feature_name=None):
        """combination of add_constant and add_features."""
        self.add_features(X, y, x_dim=x_dim, y_dim=y_dim, x_prob=x_prob, x_group=x_group,
                          feature_name=feature_name, )
        if c is not None:
            self.add_constants(c, c_dim=c_dim, c_prob=c_prob)

    def set_personal_maps(self, pers):
        """
        personal preference add to permap. more control can be found by pset.premap.***\n
        Just set couples of points and don't chang others.

        Parameters
        ----------
        pers : list of list
            Examples:
                [[index1, index2, prob]],
                the prob in [0, 1).
        """
        for i in pers:
            self.premap.set_sigle_point(*i)

    def bonding_personal_maps(self, pers):
        """
        Personal preference add to permap more control can be found by pset.premap\n
        Bond the points with ratio. the others would be penalty.\n
        For example set the [1, 2, 0.9], 
        the others bond such as (1, 2), (1, 3), (1, 4),...,(2, 3), (2, 4)...would be with small prob.

        Parameters
        ----------
        pers : list of list
            Examples:
                [[index1, index2, prob][...]]
                the prob is [0, 1), 1 means the force binding.
        """
        for i in pers:
            self.premap.down_other_point(*i)

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)

    @staticmethod
    def get_values(v, mean=False):
        """get list of dict values"""
        v = list(v.values())
        if mean:
            v = np.array(v)
            return list(v / sum(v))
        else:
            return v

    @property
    def prob_ter_con_list(self):
        return self.get_values(self.prob_ter_con, mean=True)

    @property
    def prob_pri_list(self):
        return self.get_values(self.prob_pri, mean=True)

    @property
    def prob_dispose_list(self):
        return self.get_values(self.prob_dispose, mean=True)

    @property
    def dim_ter_con_list(self):
        return self.get_values(self.dim_ter_con, mean=False)

    @property
    def primitives(self):
        """operators"""
        return self.get_values(self.primitives_dict, mean=False)

    @property
    def types(self):
        if self.gro_ter_con:
            ln = list(self.gro_ter_con.values())
            ln.append(1)
            ln = set(ln)
            if len(ln) == 1:
                return 1
            elif len(ln) == 2:
                ln = list(ln)
                ln.remove(1)
                return ln[0]
            else:
                return None
        else:
            raise NotImplementedError("the question type are defined by 'group' "
                                      "parameters in .add_features"
                                      "please add features before add operations.")

    @property
    def dispose(self):
        """accumulate operators"""
        return self.get_values(self.dispose_dict, mean=False)

    @property
    def terminals_and_constants(self):
        """terminals_and_constants"""
        return self.get_values(self.ter_con_dict, mean=False)

    @property
    def terminals_and_constants_repr(self):
        return [sympy.Symbol(repr(i)) for i in self.terminals_and_constants]

    @property
    def data_x(self):
        return self.get_values(self.data_x_dict, mean=False)

    @property
    def free_symbol(self):
        init_sub = self.terminals_symbol_map.values()
        name = self.terminals_symbol_map.keys()
        old = [sympy.Symbol(si) for si in name]

        new = list(init_sub)
        s = self.terminals_and_constants_repr
        s = [i for i in s if i not in old]
        fea_zip = old + s
        fea_unpack = new + s
        return fea_zip, fea_unpack

    @property
    def init_free_symbol(self):
        symbols = self.free_symbol[1]
        symbol = []
        [symbol.extend(tuple(i)) if isinstance(i, Iterable) else symbol.append(i) for i in symbols]

        xn = [i for i in symbol if "x" in i.name]
        cn = [i for i in symbol if "c" in i.name]
        terminals = xn + cn
        return terminals


class _ExprTree(list):
    """
    Tree of expression
    """

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo=None):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__))
        return new

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5, "
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")

        list.__setitem__(self, key, val)

    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            if node.name == "Self":
                pass
            else:
                stack.append((node, []))
                while len(stack[-1][1]) == stack[-1][0].arity:
                    prim, args = stack.pop()
                    string = prim.format_str(*args)
                    if len(stack) == 0:
                        break  # If stack is empty, all nodes should have been seen
                    stack[-1][1].append(string)

        return string

    def __repr__(self):
        """Return the expression in a machine readable string for calculating.
        """
        string = ""
        stack = []
        for node in self:
            if node.name == "Self":
                pass
            else:
                stack.append((node, []))
                while len(stack[-1][1]) == stack[-1][0].arity:
                    prim, args = stack.pop()
                    string = prim.format_repr(*args)
                    if len(stack) == 0:
                        break  # If stack is empty, all nodes should have been seen
                    stack[-1][1].append(string)

        return string

    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """

        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def length(self):
        return self.__len__()

    @property
    def h_bgp(self):
        return (self.height - 1) / 2

    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def top(self):
        """accumulative operation"""
        return self[::2]

    def bot(self):
        """operation and terminals"""
        return self[1::2]

    def cut(self, index=2):
        slice_ = self.searchSubtree(index)
        new_inds = self[slice_]
        self.clear()
        self.extend(new_inds)


class SymbolTree(_ExprTree):
    """ Individual Tree, each tree is one expression.
    The SymbolTree is only generated by method: ``genGrow`` and ``genFull``.
    """

    def __init__(self, *arg, **kwargs):
        super(SymbolTree, self).__init__(*arg, **kwargs)
        self.p_name = None
        self.y_dim = dnan
        self.pre_y = None
        self.expr = None
        self.dim_score = 0

    def reset(self):
        """keep these attribute refreshed"""
        self.p_name = None
        self.y_dim = dnan
        self.pre_y = None
        self.expr = None
        self.dim_score = 0

    def __repr__(self):
        if self.p_name:
            return self.p_name
        else:
            return _ExprTree.__repr__(self)

    def __str__(self):

        return _ExprTree.__str__(self)

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def compress(self):
        """drop unnecessary attributes"""

        [_ExprTree.__delattr__(self, i) for i in ("coef_expr", "coef_pre_y",
                                                  "coef_score", "pure_expr", "pure_pre_y")
         if hasattr(self, i)]

    def terminals(self):
        """Return terminals that occur in the expression tree."""
        return [primitive for primitive in self if primitive.arity == 0]

    def ter_site(self):
        """site for feature and constants node"""
        return [i for i, primitive in enumerate(self) if primitive.arity == 0]

    def depart(self):
        """take part the expression"""
        return depart(self)

    @property
    def capsule(self):
        """return the short one"""
        return ShortStr(self)

    @classmethod
    def genGrow(cls, pset, min_, max_, per=False, ):
        """details in genGrow function"""
        return cls(genGrow(pset, min_, max_, per))

    @classmethod
    def genFull(cls, pset, min_, max_, per=False, ):
        """details in genGrow function"""
        return cls(genFull(pset, min_, max_, per, ))

    def to_expr(self, pset):
        """transformed to sympy.Expr"""
        return compile_context(self, pset.context, pset.gro_ter_con)

    def ppprint(self, pset, feature_name=False):
        """get a user friendly version"""
        return group_str(self, pset, feature_name=feature_name)


class ShortStr:
    """short version of tree, just left name to simplify the store and transmit."""

    def __init__(self, st):
        self.reprst = repr(st)
        self.strst = str(st)

    def __str__(self):
        return self.strst

    def __repr__(self):
        return self.reprst

    def __hash__(self):
        return hash(self.__repr__())


class CalculatePrecisionSet(SymbolSet):
    """
    Add score method to SymbolSet.
    The object can get from a worked ``SymbolSet`` object.
    """
    hasher = str

    def __hash__(self):
        return hash(self.hasher(self))

    def __init__(self, pset, scoring=None, score_pen=(1,), filter_warning=True, cv=1,
                 cal_dim=True, dim_type=None, fuzzy=False, add_coef=True, inter_add=True,
                 inner_add=False, vector_add=False, out_add=False, flat_add=False, n_jobs=1, batch_size=20,
                 tq=True, details=False, classification=False, score_object="y", batch_para=False):
        """

        Parameters
        ----------
        pset:SymbolSet
            SymbolSet.
        scoring: Callbale, default is sklearn.metrics.r2_score.
            See Also sklearn.metrics.
        filter_warning:bool
            bool.
        score_pen: tuple of float
            1 : best is positive, worse -np.inf. \n
            -1 : best is negative, worse np.inf. \n
            0 : best is positive , worst is 0. \n
        cal_dim: bool
            calculate dim or not, if not return dless.
        add_coef: bool
            bool.
        inter_add: bool
            bool.
        inner_add: bool
            bool.
        fuzzy : bool
            fuzzy or not.
        dim_type : object
            if None, use the y_dim.
        n_jobs:int
            running core.
        batch_size:int
            batch size, advice batch_size*n_jobs = inds.
        tq:bool
            bool.
        cv:sklearn.model_selection._split._BaseKFold, int
            the shuffler must be False!

            use cv spilt for score, return the mean_test_score.

            use cv spilt for predict, return the cv_predict_y.(not be used)

            Notes:
                if cv and refit, all the data is refit to determination the coefficients.\n
                Thus the expression is not compact with the this scores, when re-calculated by this expression

        details:bool
            return the expr and predict y cor not.

        classification: bool
            classification or not.

        score_object:
            score by y or delta y (for implicit function).

        """
        super(CalculatePrecisionSet, self).__init__()
        self.__dict__.update(copy.deepcopy(pset.__dict__))
        self.name = "CPSet"
        self.cal_dim = cal_dim
        self.score_pen = score_pen
        self.filter_warning = filter_warning

        self.add_coef = add_coef
        self.inter_add = inter_add
        self.inner_add = inner_add
        self.vector_add = vector_add
        self.out_add = out_add
        self.flat_add = flat_add
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.batch_para = batch_para
        self.tq = tq
        self.fuzzy = fuzzy
        self.dim_type = dim_type if dim_type is not None else self.y_dim
        self.refit = True
        self.cv = cv
        self.details = details
        self.classification = classification
        self.score_object = score_object

        if not scoring:
            scoring = [r2_score, ]
        if not isinstance(scoring, (tuple, list)):
            scoring = [scoring, ]

        scoring = [score_collection[i] if isinstance(i, str) else i for i in scoring]

        self.scoring = scoring

    def update(self, pset):
        """updata self by input pset."""
        self.__dict__.update(copy.deepcopy(pset.__dict__))

    def update_with_X_y(self, X, y):
        """replace x, y data."""
        if self.expr_init_map:
            tree_x = []
            n = len(self.expr_init_map)
            self.replace(X, y=y, tree_X=np.zeros((2, n)))

            for k, v in self.expr_init_map.items():
                v = self.calculate_expr(v)
                pre_y = v["pre_y"]
                tree_x.append(pre_y)
            lens = len(tree_x)
            tree_x = np.array(tree_x).T
            tree_x = tree_x.reshape(-1, lens)
            self.replace(X, y=y, tree_X=tree_x)

        else:
            self.replace(X, y=y)

    def compile_context(self, ind):
        """transform SymbolTree to sympy.Expr."""
        if isinstance(ind, SymbolTree):
            expr = compile_context(ind.capsule, self.context, self.gro_ter_con)
        else:
            expr = compile_context(ind, self.context, self.gro_ter_con)
        return expr

    def calculate_cv_score(self, ind):
        """just used for calculating single one or check."""
        if isinstance(ind, SymbolTree):
            expr = compile_context(ind.capsule, self.context, self.gro_ter_con)
        elif isinstance(ind, sympy.Expr):
            expr = ind
        else:
            raise TypeError("must be SymbolTree or sympy.Expr")
        score, expr01, pre_y = calculate_cv_score(expr, self.data_x, self.y,
                                                  self.terminals_and_constants_repr,
                                                  cv=self.cv, refit=self.refit,
                                                  add_coef=self.add_coef, inter_add=self.inter_add,
                                                  inner_add=self.inner_add, vector_add=self.vector_add,
                                                  out_add=self.out_add, flat_add=self.flat_add,
                                                  scoring=self.scoring, score_pen=self.score_pen,
                                                  filter_warning=self.filter_warning,
                                                  np_maps=self.np_map, classification=self.classification,
                                                  score_object=self.score_object, details=True)
        return score, expr01, pre_y

    def calculate_score(self, ind):
        """
        just used for calculating single one or check with cv=1.

        Parameters
        ----------
        ind:SymbolTree
        """
        self.cv = 1
        return self.calculate_cv_score(ind)

    def calculate_detail(self, ind):
        """
        just used for calculated final best one result for showing.

        calculate the best expression.

        Parameters
        ----------
        ind: SymbolTree
            best expression.
        """
        ind = self.calculate_simple(ind)

        score, expr01, pre_y = calculate_cv_score(ind.expr, self.data_x, self.y,
                                                  self.terminals_and_constants_repr,
                                                  cv=self.cv, refit=self.refit,
                                                  add_coef=self.add_coef, inter_add=self.inter_add,
                                                  inner_add=self.inner_add, vector_add=self.vector_add,
                                                  out_add=self.out_add, flat_add=self.flat_add,
                                                  scoring=self.scoring, score_pen=self.score_pen,
                                                  filter_warning=self.filter_warning,
                                                  np_maps=self.np_map, classification=self.classification,
                                                  score_object=self.score_object, details=True)

        # this group should be get onetime and get all.
        ind.coef_expr = expr01
        ind.coef_pre_y = pre_y
        ind.coef_score = score

        ind.pure_expr = ind.expr
        ind.pure_pre_y = ind.pre_y

        return ind

    def calculate_simple(self, ind):
        """
        just used for re_Tree, and showing.

        calculate the best expression.

        Parameters
        ----------
        ind:SymbolTree
        """
        if isinstance(ind, SymbolTree):
            expr = compile_context(ind.capsule, self.context, self.gro_ter_con)
        elif isinstance(ind, sympy.Expr):
            expr = ind
        else:
            raise TypeError("must be SymbolTree or sympy.Expr")

        if self.cal_dim:
            dim, dim_score = calcualte_dim_score(expr, self.terminals_and_constants_repr,
                                                 self.dim_ter_con_list, self.dim_type,
                                                 self.fuzzy, self.dim_map)
        else:
            dim, dim_score = dless, 1

        score, expr01, pre_y = calculate_cv_score(expr, self.data_x, self.y,
                                                  self.terminals_and_constants_repr,
                                                  cv=self.cv, refit=self.refit,
                                                  add_coef=False, inter_add=False,
                                                  inner_add=False, vector_add=False, out_add=False, flat_add=False,
                                                  scoring=self.scoring, score_pen=self.score_pen,
                                                  filter_warning=self.filter_warning,
                                                  np_maps=self.np_map, classification=self.classification,
                                                  score_object=self.score_object, details=True)

        ind.y_dim = dim
        ind.expr = expr01
        ind.pre_y = pre_y
        ind.dim_score = dim_score

        return ind

    def calculate_expr(self, expr):
        """
        just used for calculated final result for showing.

        Parameters
        ----------
        ind:sympy.Expr

        """

        if isinstance(expr, sympy.Expr):
            pass
        else:
            raise TypeError("must be sympy.Expr")

        if self.cal_dim:
            dim, dim_score = calcualte_dim_score(expr, self.terminals_and_constants_repr,
                                                 self.dim_ter_con_list, self.dim_type,
                                                 self.fuzzy, self.dim_map)
        else:
            dim, dim_score = dless, 1

        score, expr01, pre_y = calculate_cv_score(expr, self.data_x, self.y,
                                                  self.terminals_and_constants_repr,
                                                  cv=self.cv, refit=self.refit,
                                                  add_coef=False, inter_add=False,
                                                  inner_add=False, vector_add=False, out_add=False, flat_add=False,
                                                  scoring=self.scoring, score_pen=self.score_pen,
                                                  filter_warning=self.filter_warning,
                                                  np_maps=self.np_map, classification=self.classification,
                                                  score_object=self.score_object, details=True)

        result = {"score": score, "expr": expr01, "pre_y": pre_y, "dim": dim, "dim_score": dim_score}

        return result

    def parallelize_calculate_expr(self, exprs):
        """just used for final results, calculate exprs."""

        calls = functools.partial(calculate_cv_score, x=self.data_x, y=self.y,
                                  terminals_and_constants_repr=self.terminals_and_constants_repr,
                                  cv=self.cv, refit=self.refit,
                                  add_coef=False, inter_add=False,
                                  inner_add=False, vector_add=False, out_add=False, flat_add=False,
                                  scoring=self.scoring, score_pen=self.score_pen,
                                  filter_warning=self.filter_warning,
                                  np_maps=self.np_map, classification=self.classification,
                                  score_object=self.score_object, details=self.details)

        score_dim_list = parallelize(func=calls, iterable=exprs, n_jobs=self.n_jobs,
                                     respective=False,
                                     tq=self.tq, batch_size=self.batch_size)
        # (sc_all, expr01, pre_y)
        return score_dim_list

    def try_add_coef_times(self, expr, grid_x=None):
        """just used for best result, try add coefficient to expr."""

        pre_y_all_expr01 = try_add_coef_times(expr, self.data_x, self.y, self.terminals_and_constants_repr, grid_x,
                                              filter_warning=self.filter_warning, inter_add=self.inter_add,
                                              inner_add=self.inner_add, vector_add=self.vector_add,
                                              out_add=self.out_add,
                                              flat_add=self.flat_add,
                                              np_maps=self.np_map, classification=self.classification,
                                              random_state=0,
                                              return_expr=False
                                              )

        return pre_y_all_expr01

    def parallelize_try_add_coef_times(self, exprs, grid_x=None, resample_number=500):
        """to be continued"""
        if isinstance(grid_x, np.ndarray):
            grid_x = list(grid_x.T)
        calls = functools.partial(try_add_coef_times, x=self.data_x, y=self.y,
                                  terminals=self.terminals_and_constants_repr,
                                  grid_x=grid_x,
                                  filter_warning=self.filter_warning, inter_add=self.inter_add,
                                  inner_add=self.inner_add, vector_add=self.vector_add,
                                  out_add=self.out_add,
                                  flat_add=self.flat_add,
                                  np_maps=self.np_map, classification=self.classification,
                                  random_state=0,
                                  return_expr=False,
                                  resample_number=resample_number,
                                  )

        pre_y_all_list = parallelize(func=calls, iterable=exprs, n_jobs=1, respective=False,
                                     tq=self.tq)

        return pre_y_all_list

    def parallelize_score(self, inds):
        """
        The main score in each generation of GP!

        Parameters
        ----------
        inds:list of SymbolTree
            list of expressions

        """

        indss = [i.capsule for i in inds]

        calculate_collect_use = calculate_collect_

        calls = functools.partial(calculate_collect_use, context=self.context, x=self.data_x, y=self.y,
                                  terminals_and_constants_repr=self.terminals_and_constants_repr,
                                  gro_ter_con=self.gro_ter_con, cv=self.cv, refit=self.refit,
                                  dim_ter_con_list=self.dim_ter_con_list, dim_type=self.dim_type,
                                  fuzzy=self.fuzzy,
                                  scoring=self.scoring, score_pen=self.score_pen,
                                  vector_add=self.vector_add,
                                  add_coef=self.add_coef, inter_add=self.inter_add,
                                  out_add=self.out_add, flat_add=self.flat_add,
                                  inner_add=self.inner_add, np_maps=self.np_map, classification=self.classification,
                                  filter_warning=self.filter_warning,
                                  dim_maps=self.dim_map, cal_dim=self.cal_dim, score_object=self.score_object,
                                  details=self.details
                                  )

        if isinstance(self.batch_size, int) and self.batch_para:
            score_dim_list = batch_parallelize(func=calls, iterable=indss, n_jobs=self.n_jobs,
                                               respective=False,
                                               tq=self.tq, batch_size=self.batch_size)
        else:
            score_dim_list = parallelize(func=calls, iterable=indss, n_jobs=self.n_jobs,
                                         respective=False,
                                         tq=self.tq, batch_size=self.batch_size)

        return score_dim_list
