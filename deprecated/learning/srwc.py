import operator
import random
from copy import deepcopy
from functools import partial

import numpy as np
import sympy
from bgp.combination import creator
from bgp.combination.common import _compile, varAnd
from bgp.combination.common import selKbestDim
from bgp.combination.deapbase import ExpressionSetFill
from bgp.combination.deapbase import ExpressionTree, ExpressionSet
from bgp.combination.dictbase import FixedSet, FixedTree, generate_index, cxOnePoint_index, mutUniForm_index
from bgp.combination.dim import Dim, dnan, dless
from deap.base import Fitness, Toolbox
from deap.gp import staticLimit, cxOnePoint, mutNodeReplacement, genHalfAndHalf
from deap.tools import HallOfFame, MultiStatistics, Statistics, initIterate, initRepeat, selTournament, Logbook
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import parallelize
from scipy import optimize
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.utils import check_array
from sympy import sympify


def addCoefficient(expr01, inter_add=True, iner_add=True, random_add=None):
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
        """"""
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
        expr01 = sympy.Mul(expr01, A)

        a_list.append(A)

    if inter_add:
        B = sympy.Symbol("B")
        expr01 = expr01 + B
        a_list.append(B)

    if iner_add:
        cho_add = [i.args for i in arg_list if isinstance(i, sympy.Add)]
        cho_add = [[_ for _ in cho_addi if not _.is_number] for cho_addi in cho_add]
        [cho.extend(i) for i in cho_add]

    if random_add:
        pass
    #     lest = [i for i in arg_list if i not in cho]
    #     if len(lest) != 0:
    #         cho2 = random.sample(lest, 1)
    #         cho.extend(cho2)

    a_cho = [sympy.Symbol("k%s" % i) for i in range(len(cho))]
    for ai, choi in zip(a_cho, cho):
        expr01 = expr01.subs(choi, ai * choi)
    a_list.extend(a_cho)

    return expr01, a_list


def calculateExpr(expr01, x, y, terminals, scoring=None, add_coeff=True,
                  del_no_important=False, filter_warning=True, inter_add=True, iner_add=True, random_add=None):
    """

    Parameters

    """

    def split_x(x):
        if x.ndim == 1:
            return [x]
        else:
            return [*x.T]

    # if filter_warning:
    #     warnings.filterwarnings("ignore")
    if not scoring:
        scoring = r2_score

    expr00 = deepcopy(expr01)  #

    if add_coeff:

        expr01, a_list = addCoefficient(expr01, inter_add=inter_add, iner_add=iner_add, random_add=random_add)

        try:
            func0 = sympy.utilities.lambdify(terminals + a_list, expr01)

            def func(x_, p):
                """"""
                num_list = []

                num_list.extend(split_x(x))

                num_list.extend(p)
                return func0(*num_list)

            def res(p, x_, y_):
                """"""
                return y_ - func(x_, p)

            result = optimize.least_squares(res, x0=[1] * len(a_list), args=(x, y), loss='linear', ftol=1e-3)

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
        except (ValueError, KeyError, NameError, TypeError):
            expr01 = expr00

    else:
        pass  #

    try:
        if del_no_important and isinstance(expr01, sympy.Add) and len(expr01.args) >= 3:
            re_list = []
            for expri in expr01.args:
                if not isinstance(expri, sympy.Float):
                    func0 = sympy.utilities.lambdify(terminals, expri)
                    re = np.mean(func0(*split_x(x)))
                    if abs(re) > abs(0.001 * np.mean(y)):
                        re_list.append(expri)
                else:
                    re_list.append(expri)
            expr01 = sum(re_list)
        else:
            pass

        func0 = sympy.utilities.lambdify(terminals, expr01)
        re = func0(*split_x(x))
        re = re.ravel()
        assert y.shape == re.shape
        # assert_all_finite(re)
        re = check_array(re, ensure_2d=False)
        score = scoring(y, re)

    except (ValueError, DataConversionWarning, NameError, KeyError, AssertionError, AttributeError):
        score = -0
    else:
        if np.isnan(score):
            score = -0
    return score, expr01


def calculatePrecision(individual, pset, x, y, scoring=None, add_coeff=True, filter_warning=True, cal_dim=False,
                       inter_add=True, iner_add=True, random_add=None):
    """

    Parameters
    ----------

    """

    if scoring is None:
        scoring = r2_score
    # '''1 not expand'''
    expr_no = sympify(_compile(individual, pset))
    if isinstance(expr_no, sympy.Add):
        expr_no = sum([_ * sympy.Symbol("x1") for _ in expr_no.args])
    else:
        expr_no = expr_no * sympy.Symbol("x1")
    t = np.corrcoef(y, input_x[:, 0] * input_x[:, 1])

    t2 = r2_score(y, input_x[:, 0] * input_x[:, 1])

    score, expr = calculateExpr(expr_no, x=x, y=y, terminals=[sympy.Symbol("x0"), sympy.Symbol("x1")], scoring=scoring,
                                add_coeff=add_coeff,
                                filter_warning=filter_warning, inter_add=inter_add, iner_add=iner_add,
                                random_add=random_add)
    if cal_dim:
        pass

    return score, expr, dless, 1


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
    rst = random.getstate()
    len_pop = len(population)
    logbook = Logbook()
    logbook.header = [] + (stats.fields if stats else [])
    data_all = {}
    random.setstate(rst)

    for gen in range(1, ngen + 1):
        "评价"
        rst = random.getstate()
        """score"""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.parallel(iterable=population)
        for ind, fit, in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0],
            ind.expr = fit[1]
            ind.y_dim = fit[2]
            ind.withdim = fit[3]
        random.setstate(rst)

        rst = random.getstate()
        """elite"""
        add_ind = []
        add_ind1 = toolbox.select_kbest_target_dim(population, K_best=0.05 * len_pop)
        add_ind += add_ind1
        elite_size = len(add_ind)
        random.setstate(rst)

        rst = random.getstate()
        """score"""

        random.setstate(rst)

        rst = random.getstate()
        """record"""
        if halloffame is not None:
            halloffame.update(add_ind1)
            if len(halloffame.items) > 0 and halloffame.items[-1].fitness.values[0] >= 0.9999:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                break
        random.setstate(rst)

        rst = random.getstate()
        """Dynamic output"""

        record = stats.compile_(population) if stats else {}
        logbook.record(gen=gen, pop=len(population), **record)

        if verbose:
            print(logbook.stream)
        random.setstate(rst)

        """crossover, mutate"""
        offspring = toolbox.select_gs(population, len_pop - elite_size)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        rst = random.getstate()
        """re-run"""
        offspring.extend(add_ind)
        population[:] = offspring
        random.setstate(rst)

    store = Store()
    store.to_csv(data_all)
    return population, logbook


def mainPart(x_, y_, pset, max_=5, pop_n=100, random_seed=2, cxpb=0.8, mutpb=0.1, ngen=5,
             tournsize=3, max_value=10, double=False, score=None, cal_dim=True, target_dim=None,
             inter_add=True, iner_add=True, random_add=False, store=True):
    """

    Parameters
    ----------
    target_dim
    max_
    inter_add
    iner_add
    random_add
    cal_dim
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
    tournsize
    max_value

    Returns
    -------

    """
    if score is None:
        score = [r2_score, explained_variance_score]

    if cal_dim:
        assert all([isinstance(i, Dim) for i in pset.dim_list]), "all import dim of pset should be Dim object"

    random.seed(random_seed)
    toolbox = Toolbox()

    if isinstance(pset, ExpressionSet):
        PTrees = ExpressionTree
        Generate = genHalfAndHalf
        mutate = mutNodeReplacement
        mate = cxOnePoint
    elif isinstance(pset, FixedSet):
        PTrees = FixedTree
        Generate = generate_index
        mutate = mutUniForm_index
        mate = partial(cxOnePoint_index, pset=pset)

    else:
        raise NotImplementedError("get wrong pset")
    if double:
        Fitness_ = creator.create("Fitness_", Fitness, weights=(1.0, 1.0))
    else:
        Fitness_ = creator.create("Fitness_", Fitness, weights=(1.0,))

    PTrees_ = creator.create("PTrees_", PTrees, fitness=Fitness_, dim=dnan, withdim=0)
    toolbox.register("generate", Generate, pset=pset, min_=1, max_=max_)
    toolbox.register("individual", initIterate, container=PTrees_, generator=toolbox.generate)
    toolbox.register('population', initRepeat, container=list, func=toolbox.individual)
    # def selection
    toolbox.register("select_gs", selTournament, tournsize=tournsize)
    toolbox.register("select_kbest_target_dim", selKbestDim, dim_type=target_dim, fuzzy=True)
    toolbox.register("select_kbest_dimless", selKbestDim, dim_type="integer")
    toolbox.register("select_kbest", selKbestDim, dim_type='ignore')
    # def mate
    toolbox.register("mate", mate)
    # def mutate
    toolbox.register("mutate", mutate, pset=pset)
    if isinstance(pset, ExpressionSet):
        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=max_value))
    # def elaluate
    toolbox.register("evaluate", calculatePrecision, pset=pset, x=x_, y=y_, scoring=score[0], cal_dim=cal_dim,
                     inter_add=inter_add, iner_add=iner_add, random_add=random_add)
    toolbox.register("evaluate2", calculatePrecision, pset=pset, x=x_, y=y_, scoring=score[1], cal_dim=cal_dim,
                     inter_add=inter_add, iner_add=iner_add, random_add=random_add)
    toolbox.register("parallel", parallelize, n_jobs=1, func=toolbox.evaluate, respective=False)
    toolbox.register("parallel2", parallelize, n_jobs=1, func=toolbox.evaluate2, respective=False)

    pop = toolbox.population(n=pop_n)

    haln = 10
    hof = HallOfFame(haln)

    stats1 = Statistics(lambda ind: ind.fitness.values[0] if ind and ind.y_dim in target_dim else 0)
    stats1.register("max", np.max)

    stats2 = Statistics(lambda ind: ind.y_dim in target_dim if ind else 0)
    stats2.register("countable_number", np.sum)
    stats = MultiStatistics(score1=stats1, score2=stats2)

    population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                   halloffame=hof, pset=pset, store=store)

    return population, hof

    # import sympy
    #
    # data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    #
    # X_frame = data216_import[select]
    # y_frame = data216_import['exp_gap']
    #
    # X = X_frame.values
    # y = y_frame.values
    #
    # x0 = sympy.Symbol("x0")
    # x1 = sympy.Symbol("x1")
    # x2 = sympy.Symbol("x2")
    # x3 = sympy.Symbol("x3")
    #
    # #
    # expr01 = (x0 ** 0.5 - x1 ** 0.5 + 1) ** 2 * sympy.log(x2 / x3) ** 2
    #
    # terminals = [x0, x1, x2, x3]
    # score, expr01 = calculateExpr(expr01, X, y, terminals, scoring=None, add_coeff=True,
    #                               del_no_important=False, filter_warning=True, inter_add=False, iner_add=True,
    #                               random_add=None)
    # x = X
    # x0 = x[:, 0]
    # x1 = x[:, 1]
    # x2 = x[:, 2]
    # x3 = x[:, 3]
    #
    # t = expr01
    # func0 = sympy.utilities.lambdify(terminals, t)
    # re = func0(*x.T)
    # p = BasePlot(font=None)
    # p.scatter(y, re, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    # import matplotlib.pyplot as plt
    #
    # plt.show()


if __name__ == '__main__':
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\4.symbol')
    data = Call(r'C:\Users\Administrator\Desktop\c', index_col=None)
    data_import = data.xlsx().sr

    X = data_import["delt_x"].values
    input_x = data_import[["delt_x", "G"]].values

    Pexp = data_import["Pexp"].values
    Pmix = data_import["Pmix"].values

    G = data_import["G"].values
    y = data_import["PG_y"].values
    y = y * G
    testfunc = input_x[:, 0] * input_x[:, 1]
    t = np.corrcoef(y, input_x[:, 0] * input_x[:, 1])

    dim1 = Dim([0, 0, 0, 0, 0, 0, 0])
    target_dim = [Dim([0, 0, 0, 0, 0, 0, 0])]
    dim_list = [dim1]


    def my_func1(y, y_pre):
        return 1 - np.mean(np.abs((y + Pmix) - (y_pre + Pmix)) / Pexp)


    def my_func3(y, y_pre):
        return 1 - mean_absolute_error(y, y_pre) / Pexp


    def my_func2(y, y_pre):
        return r2_score(y + Pmix, y_pre + Pmix) ** 0.5


    pset = ExpressionSetFill(x_name=["x0"], power_categories=[1 / 3, 1 / 2, 2, 3, 2 / 3, 3 / 2, 4 / 3],
                             categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Self", "Rem"),
                             partial_categories=None, self_categories=None, dim=dim_list)
    result = mainPart(input_x, y, pset, pop_n=500, random_seed=0, cxpb=1, mutpb=0.6, ngen=10, tournsize=3, max_value=2,
                      max_=2,
                      double=False, score=[my_func2, my_func2], inter_add=False, iner_add=True, target_dim=target_dim,
                      cal_dim=False)
    for i in [i.expr for i in result[1].items]:
        print(i)
    for i in [i.values for i in result[1].keys][::-1]:
        print(i)
