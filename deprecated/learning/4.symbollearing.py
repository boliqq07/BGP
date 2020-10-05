# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 16:10
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

# @Time    : 2019/11/12 15:13
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import functools
import operator
import random
from functools import partial

import numpy as np
from bgp.combination import creator
from bgp.combination.common import calculatePrecision, selKbestDim
from bgp.combination.deapbase import ExpressionSetFill
from bgp.combination.deapbase import ExpressionTree, ExpressionSet
from bgp.combination.dictbase import FixedSet, FixedTree, generate_index, cxOnePoint_index, mutUniForm_index
from bgp.combination.dim import Dim, dnan
from deap.base import Fitness, Toolbox
from deap.gp import staticLimit, cxOnePoint, mutNodeReplacement, genHalfAndHalf
from deap.tools import HallOfFame, MultiStatistics, Statistics, initIterate, initRepeat, selTournament, Logbook
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import time_this_function, parallelize
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.utils import shuffle


def varAnd(population, toolbox, cxpb, mutpb):
    rst = random.getstate()
    offspring = [toolbox.clone(ind) for ind in population]
    random.setstate(rst)
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


def sub(expr01, subed, subs):
    """"""
    listt = list(zip(subed, subs))
    return expr01.subs(listt)


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
            ind.dim = fit[2]
            ind.withdim = fit[3]
        random.setstate(rst)

        rst = random.getstate()
        """elite"""
        add_ind = []
        add_ind1 = toolbox.select_kbest_target_dim(population, K_best=0.01 * len_pop)
        add_ind2 = toolbox.select_kbest_dimless(population, K_best=0.01 * len_pop)
        add_ind3 = toolbox.select_kbest(population, K_best=5)
        add_ind += add_ind1
        add_ind += add_ind2
        add_ind += add_ind3
        elite_size = len(add_ind)
        random.setstate(rst)

        rst = random.getstate()
        """score"""
        if store:
            subp = functools.partial(sub, subed=pset.rep_name_list, subs=pset.real_name_list)
            data = {"gen{}_pop{}".format(gen, n): {"gen": gen, "pop": n,
                                                   "score": i.fitness.values[0],
                                                   "expr": str(subp(i.expr)),
                                                   } for n, i in enumerate(population) if i is not None}
            data_all.update(data)
        random.setstate(rst)

        rst = random.getstate()
        """record"""
        if halloffame is not None:
            halloffame.update(add_ind3)
            if len(halloffame.items) > 0 and halloffame.items[-1].fitness.values[0] >= 0.95:
                print(halloffame.items[-1])
                print(halloffame.items[-1].fitness.values[0])
                break
        random.setstate(rst)

        rst = random.getstate()
        """Dynamic output"""

        record = stats.compile(population) if stats else {}
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


@time_this_function
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

    haln = 5
    hof = HallOfFame(haln)

    stats1 = Statistics(lambda ind: ind.fitness.values[0])
    stats1.register("max", np.max)

    stats2 = Statistics(lambda ind: 0 if ind else 0)
    stats2.register("countable_number", np.sum)
    stats = MultiStatistics(score1=stats1, score2=stats2)

    population, logbook = eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats,
                                   halloffame=hof, pset=pset, store=store)

    return hof


if __name__ == '__main__':
    # 输入
    store = Store(r'D:\sy')
    data = Call(r'D:\sy')

    data_import = data.xlsx().featuredata
    name_abbr = data_import.columns.values
    x_name = name_abbr[:-1]
    # data_import = data_import.iloc[np.where(data_import['f1'] <= 1)[0]]

    X_frame = data_import[x_name]
    y_frame = data_import['y']

    X = X_frame.values
    y = y_frame.values

    # 处理
    # scal = preprocessing.MinMaxScaler()
    # X = scal.fit_transform(X)

    X, y = shuffle(X, y, random_state=0)

    # 符号
    # def new(x):
    #     return x/2
    # def add(x1,x2):
    #     return x1+x2
    # self_categories= [[new, 1, 'new'],[add, 2, 'add']]

    pset = ExpressionSetFill(x_name=x_name, power_categories=[2, 3, 0.5, 1 / 3],
                             categories=("Add", "Mul", "Sub", "Div", "log", "Rec"),
                             partial_categories=None, self_categories=None)


    def custom_loss_func(y_true, y_pred):
        """"""
        diff = - np.abs(y_true - y_pred) / y_true + 1
        return np.mean(diff)


    results = mainPart(X, y, pset, pop_n=500, random_seed=1, cxpb=1, mutpb=0.6,
                       ngen=3, max_value=10, max_=4,
                       score=[custom_loss_func, custom_loss_func], cal_dim=False, store=True,
                       inter_add=False, iner_add=True)

    result = np.array([[i.fitness.values[0], str(i.expr)] for i in results.items])
