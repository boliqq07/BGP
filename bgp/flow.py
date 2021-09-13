# #!/usr/bin/python
# # -*- coding: utf-8 -*-
#
# # @Time    : 2019/11/12 15:13
# # @Email   : 986798607@qq.com
# # @Software: PyCharm
# # @License: GNU Lesser General Public License v3.0
"""
Some definition loop for genetic algorithm.
All the loop is with same run method.

Contains:

-Class: ``BaseLoop``

    one node mate and one tree mutate.

-Class: ``MultiMutateLoop``

    one node mate and (one tree mutate, one node Replacement mutate, shrink mutate, difference mutate).

-Class: ``OnePointMutateLoop``

    one node Replacement mutate: (keep height of tree)

-Class: ``DimForceLoop``

    Select with dimension : (keep dimension of tree)

"""
import copy
import operator
import os
import time

from deap.base import Fitness
from deap.tools import HallOfFame, Logbook
from mgetool import newclass
from mgetool.exports import Store
from mgetool.packbox import Toolbox
from numpy import random
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

from bgp.base import CalculatePrecisionSet
from bgp.base import SymbolSet
from bgp.base import SymbolTree
from bgp.functions.dimfunc import Dim, dless
from bgp.gp import cxOnePoint, varAnd, genGrow, staticLimit, selKbestDim, \
    selTournament, Statis_func, mutUniform, mutShrink, varAndfus, \
    mutDifferentReplacementVerbose, mutNodeReplacementVerbose, selBest, genFull


class BaseLoop(Toolbox):
    """
    Base loop for BGP.

    Examples::

        if __name__ == "__main__":
            pset = SymbolSet()
            stop = lambda ind: ind.fitness.values[0] >= 0.880963

            bl = BaseLoop(pset=pset, gen=10, pop=1000, hall=1, batch_size=40, re_hall=3, \n
            n_jobs=12, mate_prob=0.9, max_value=5, initial_min=1, initial_max=2, \n
            mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,\n
            re_Tree=0, store=False, random_state=1, verbose=True,\n
            stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},\n
            add_coef=True, inter_add=True, inner_add=False, cal_dim=True, vector_add=False,\n
            personal_map=False)

            bl.run()

    """

    def __init__(self, pset, pop=500, gen=20, mutate_prob=0.5, mate_prob=0.8, hall=1, re_hall=1,
                 re_Tree=None, initial_min=None, initial_max=3, max_value=5,
                 scoring=(r2_score,), score_pen=(1,), filter_warning=True, cv=1,
                 add_coef=True, inter_add=True, inner_add=False, vector_add=False, out_add=False, flat_add=False,
                 cal_dim=False, dim_type=None, fuzzy=False, n_jobs=1, batch_size=40,
                 random_state=None, stats=None, verbose=True, migrate_prob=0,
                 tq=True, store=False, personal_map=False, stop_condition=None, details=False, classification=False,
                 score_object="y", sub_mu_max=1, limit_type="h_bgp", batch_para=False):
        """

        Parameters
        ----------
        pset:SymbolSet
            the feature x and target y and others should have been added.
        pop: int
            number of population.
        gen: int
            number of generation.
        mutate_prob:float
            probability of mutate.
        mate_prob:float
            probability of mate(crossover).
        initial_max:int
            max initial size of expression when first producing.
        initial_min : None,int
            min initial size of expression when first producing.
        max_value:int
            max size of expression.
        limit_type: "height" or "length",","h_bgp"
            limitation type for max_value, but don't affect initial_max, initial_min.
        hall:int,>=1
            number of HallOfFame (elite) to maintain.
        re_hall:None or int>=2
            Notes: only valid when hall
            number of HallOfFame to add to next generation.
        re_Tree: int
            number of new features to add to next generation.
            0 is false to add.
        personal_map:bool or "auto"
            "auto" is using 'premap' and with auto refresh the 'premap' with individual.\n
            True is just using constant 'premap'.\n
            False is just use the prob of terminals.
        scoring: list of Callable, default is [sklearn.metrics.r2_score,]
            See Also ``sklearn.metrics``
        score_pen: tuple of  1, -1 or float but 0.
            >0 : max problem, best is positive, worse -np.inf.
            <0 : min problem, best is negative, worse np.inf.

            Notes:
                if multiply score method, the scores must be turn to same dimension in prepossessing
                or weight by score_pen. Because the all the selection are stand on the mean(w_i*score_i)

            Examples::

                scoring = [r2_score,]
                score_pen= [1,]

        cv:sklearn.model_selection._split._BaseKFold,int
            the shuffler must be False,
            default=1 means no cv.
        filter_warning:bool
            filter warning or not.
        add_coef:bool
            add coef in expression or not.
        inter_add：bool
            add intercept constant or not.
        inner_add:bool
            add inner coefficients or not.
        out_add:bool
            add out coefficients or not.
        flat_add:bool
            add flat coefficients or not.
        n_jobs:int
            default 1, advise 6.
        batch_size:int
            default 40, depend of machine.
        random_state:int
            None,int.
        cal_dim:bool
            escape the dim calculation.
        dim_type:Dim or None or list of Dim
            "coef": af(x)+b. a,b have dimension,f(x)'s dimension is not dnan. \n
            "integer": af(x)+b. f(x) is with integer dimension. \n
            [Dim1,Dim2]: f(x)'s dimension in list. \n
            Dim: f(x) ~= Dim. (see fuzzy) \n
            Dim: f(x) == Dim. \n
            None: f(x) == pset.y_dim
        fuzzy:bool
            choose the dim with same base with dim_type, such as m,m^2,m^3.
        stats:dict
            details of logbook to show. \n
            Map:\n
            values
                = {"max": np.max, "mean": np.mean, "min": np.mean, "std": np.std, "sum": np.sum}
            keys
                = {\n
                   "fitness": just see fitness[0], \n
                   "fitness_dim_max": max problem, see fitness with demand dim,\n
                   "fitness_dim_min": min problem, see fitness with demand dim,\n
                   "dim_is_target": demand dim,\n
                   "coef":  dim is True, coef have dim, \n
                   "integer":  dim is integer, \n
                   ...
                   }

            if stats is None, default is:

            for cal_dim=True:
                stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}

            for cal_dim=False
                stats = {"fitness": ("max",)}

            if self-definition, the key is func to get attribute of each ind.

            Examples::

                def func(ind):
                    return ind.fitness[0]
                stats = {func: ("mean",), "dim_is_target": ("sum",)}

        verbose:bool
            print verbose logbook or not.
        tq:bool
            print progress bar or not.
        store:bool or path
            bool or path.
        stop_condition:callable
            stop condition on the best ind of hall, which return bool,the true means stop loop.

            Examples::

                def func(ind):
                    c = ind.fitness.values[0]>=0.90
                    return c

        details:bool
            return expr and predict_y or not.

        classification: bool
            classification or not.

        score_object:
            score by y or delta y (for implicit function).
        """
        super(BaseLoop, self).__init__()

        assert initial_max <= max_value, "the initial size of expression should less than max_value limitation"
        if cal_dim:
            assert all(
                [isinstance(i, Dim) for i in pset.dim_ter_con.values()]), \
                "all import dim of pset should be Dim object."

        self.details = details
        self.max_value = max_value
        self.pop = pop
        self.gen = gen
        self.mutate_prob = mutate_prob
        self.mate_prob = mate_prob
        self.migrate_prob = migrate_prob
        self.verbose = verbose
        self.cal_dim = cal_dim
        self.re_hall = re_hall
        self.re_Tree = re_Tree
        self.store = store
        self.limit_type = limit_type
        self.data_all = []
        self.personal_map = personal_map
        self.stop_condition = stop_condition
        self.population = []
        self.rand_state = None
        self.random_state = random_state
        self.sub_mu_max = sub_mu_max
        self.population_next = []

        self.cpset = CalculatePrecisionSet(pset, scoring=scoring, score_pen=score_pen,
                                           filter_warning=filter_warning, cal_dim=cal_dim,
                                           add_coef=add_coef, inter_add=inter_add, inner_add=inner_add,
                                           vector_add=vector_add, out_add=out_add, flat_add=flat_add, cv=cv,
                                           n_jobs=n_jobs, batch_size=batch_size, tq=tq,
                                           fuzzy=fuzzy, dim_type=dim_type, details=details,
                                           classification=classification, score_object=score_object,
                                           batch_para=batch_para
                                           )

        Fitness_ = newclass.create("Fitness_", Fitness, weights=score_pen)
        self.PTree = newclass.create("PTrees", SymbolTree, fitness=Fitness_)
        # def produce
        if initial_min is None:
            initial_min = 2
        self.register("genGrow", genGrow, pset=self.cpset, min_=initial_min, max_=initial_max + 1,
                      personal_map=self.personal_map)
        self.register("genFull", genFull, pset=self.cpset, min_=initial_min, max_=initial_max + 1,
                      personal_map=self.personal_map)
        self.register("genHalf", genGrow, pset=self.cpset, min_=initial_min, max_=initial_max + 1,
                      personal_map=self.personal_map)
        self.register("gen_mu", genGrow, min_=1, max_=self.sub_mu_max + 1, personal_map=self.personal_map)
        # def selection

        self.register("select", selTournament, tournsize=2)

        self.register("selKbestDim", selKbestDim,
                      dim_type=self.cpset.dim_type, fuzzy=self.cpset.fuzzy)
        self.register("selBest", selBest)

        self.register("mate", cxOnePoint)
        # def mutate

        self.register("mutate", mutUniform, expr=self.gen_mu, pset=self.cpset)

        self.decorate("mate", staticLimit(key=operator.attrgetter(limit_type), max_value=self.max_value))
        self.decorate("mutate", staticLimit(key=operator.attrgetter(limit_type), max_value=self.max_value))

        if stats is None:
            if cal_dim:
                if score_pen[0] > 0:
                    stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}
                else:
                    stats = {"fitness_dim_min": ("min",), "dim_is_target": ("sum",)}
            else:
                if score_pen[0] > 0:
                    stats = {"fitness": ("max",)}
                else:
                    stats = {"fitness": ("min",)}

        self.stats = Statis_func(stats=stats)
        logbook = Logbook()
        logbook.header = ['gen'] + (self.stats.fields if self.stats else [])
        self.logbook = logbook

        if hall is None:
            hall = 1
        self.hall = HallOfFame(hall)

        if re_hall is None:
            self.re_hall = None
        else:
            if re_hall == 1 or re_hall == 0:
                print("re_hall should more than 1")
                re_hall = 2
            assert re_hall >= hall, "re_hall should more than hall"
            self.re_hall = HallOfFame(re_hall)

    def varAnd(self, *arg, **kwargs):
        return varAnd(*arg, **kwargs)

    def to_csv(self, data_all):
        """store to csv"""
        if self.store:
            if isinstance(self.store, str):
                path = self.store
            else:
                path = os.getcwd()
            file_new_name = "_".join((str(self.pop), str(self.gen),
                                      str(self.mutate_prob), str(self.mate_prob),
                                      str(time.time())))
            try:
                st = Store(path)
                st.to_csv(data_all, file_new_name, transposition=True)
                print("store data to ", path, file_new_name)
            except (IOError, PermissionError):
                st = Store(os.getcwd())
                st.to_csv(data_all, file_new_name, transposition=True)
                print("store data to ", os.getcwd(), file_new_name)

    def maintain_halls(self, population):
        """maintain the best p expression"""
        if self.re_hall is not None:
            maxsize = max(self.hall.maxsize, self.re_hall.maxsize)

            if self.cal_dim:
                inds_dim = self.selKbestDim(population, maxsize)
            else:
                inds_dim = self.selBest(population, maxsize)

            self.hall.update(inds_dim)
            self.re_hall.update(inds_dim)

            sole_inds = [i for i in self.re_hall.items if i not in inds_dim]
            inds_dim.extend(sole_inds)
        else:
            if self.cal_dim:
                inds_dim = self.selKbestDim(population, self.hall.maxsize)
            else:
                inds_dim = self.selBest(population, self.hall.maxsize)

            self.hall.update(inds_dim)
            inds_dim = []

        inds_dim = copy.deepcopy(inds_dim)
        return inds_dim

    def re_add(self):
        """add the expression as a feature"""
        if self.hall.items and self.re_Tree:
            it = self.hall.items
            indo = it[random.choice(len(it))]
            ind = copy.deepcopy(indo)
            inds = ind.depart()
            if not inds:
                pass
            else:
                inds = [self.cpset.calculate_detail(indi) for indi in inds]
                le = min(self.re_Tree, len(inds))
                indi = inds[random.choice(le)]
                self.cpset.add_tree_to_features(indi)

    def re_fresh_by_name(self, *arr):
        re_name = ["mutate", "genGrow", "genFull", "genHalf"]
        if len(arr) > 0:
            re_name.extend(arr)
        self.refresh(re_name, pset=self.cpset)
        # for i in re_name + ["mate"]:  # don‘t del this
        #     self.decorate(i, staticLimit(key=operator.attrgetter("height"), max_value=2 * (self.max_value + 1)))

    def top_n(self, n=10, gen=-1, key="value", filter_dim=True, ascending=False):
        """
        Return the best n results.

        Note:
            Only valid in ``store=True``.

        Parameters
        ----------
        n:int
            n.
        gen:
            the generation, default is -1.
        key: str
            sort keys, default is "values".
        filter_dim:
            filter no-dim expressions or not.
        ascending:
            reverse.

        Returns
        -------
        top n results.
        pd.DataFrame

        """
        import pandas as pd
        if self.store == "False":
            raise TypeError("Only valid with store=True")
        data = self.data_all

        data = pd.DataFrame(data)
        if gen == -1:
            gen = max(data["gen"])

        data = data[data["gen"] == gen]

        if filter_dim:
            data = data[data["dim_score"] == 1]

        data = data.drop_duplicates(['expr'], keep="first")

        if key is not None:
            data[key] = data[key].str.replace("(", "")
            data[key] = data[key].str.replace(")", "")
            data[key] = data[key].str.replace(",", "")
            try:
                data[key] = data[key].astype(float)
            except ValueError:
                raise TypeError("check this key column can be translated into float")

            data = data.sort_values(by='value', ascending=ascending).iloc[:n, :]

        return data

    def check_height_length(self, pop, site=""):
        old = len(pop)
        if self.limit_type == 'height':
            pop = [i for i in pop if i.height <= self.max_value]
        elif self.limit_type == 'length':
            pop = [i for i in pop if i.length <= self.max_value]
        else:
            pop = [i for i in pop if i.h_bgp <= self.max_value]
        new = len(pop)
        if old == new:
            pass
        else:
            if site != "":
                print(site)
            # raise TypeError
            index = random.randint(0, new, old - new)
            pop.extend([pop[i] for i in index])
        return pop

    def run(self, warm_start=False, new_gen=None):
        """

        Parameters
        ----------
        warm_start:bool
            warm_start from last result.
        new_gen:
            new generations for warm_startm, default is the initial generations.

        """
        # 1.generate###################################################################
        if warm_start is False:
            random.seed(self.random_state)
            population = [self.PTree(self.genHalf()) for _ in range(self.pop)]
            gen_i = 0
            gen = self.gen
        else:
            assert self.population_next != []
            random.set_state(self.rand_state)
            population = self.population_next
            gen_i = self.gen_i
            self.re_fresh_by_name()
            if new_gen:
                gen = gen_i + new_gen
            else:
                gen = gen_i + self.gen

        for gen_i in range(gen_i + 1, gen + 1):

            population_old = copy.deepcopy(population)

            # 2.evaluate###############################################################

            invalid_ind_score = self.cpset.parallelize_score(population_old)

            for ind, score in zip(population_old, invalid_ind_score):
                ind.fitness.values = tuple(score[0])
                ind.y_dim = score[1]
                ind.dim_score = score[2]
                ind.coef_expr = score[3]
                ind.coef_pre_y = score[4]
            population = population_old
            # 3.log###################################################################
            # 3.1.log-print##############################

            record = self.stats.compile(population) if self.stats else {}
            self.logbook.record(gen=gen_i, **record)
            if self.verbose:
                print(self.logbook.stream)
            # 3.2.log-store##############################
            if self.store:
                datas = [{"gen": gen_i, "name": str(pop_i), "expr": str([pop_i.coef_expr]),
                          "value": str(pop_i.fitness.values),
                          "dimension": str(pop_i.y_dim),
                          "dim_score": pop_i.dim_score} for pop_i in population]
                self.data_all.extend(datas)

            self.population = copy.deepcopy(population)

            # 3.3.log-hall###############################
            inds_dim = self.maintain_halls(population)

            # 4.refresh################################################################
            # 4.1.re_update the premap ##################
            if self.personal_map == "auto":
                [self.cpset.premap.update(indi, self.cpset) for indi in inds_dim]

            # 4.2.re_add_tree and refresh pset###########
            if self.re_Tree:
                self.re_add()

            self.re_fresh_by_name()

            # 6.next generation ！！！！#######################################################
            # selection and mutate,mate,migration
            population = self.select(population, int((1 - self.migrate_prob) * len(population)) - len(inds_dim))

            offspring = self.varAnd(population, self, self.mate_prob, self.mutate_prob)
            offspring.extend(inds_dim)
            migrate_pop = [self.PTree(self.genFull()) for _ in range(int(self.migrate_prob * len(population)))]
            population[:] = offspring + migrate_pop

            population = self.check_height_length(population)

            # 5.break#######################################################
            if self.stop_condition is not None:
                if self.stop_condition(self.hall.items[0]):
                    break

            # 7 freeze ###################################################

            self.rand_state = random.get_state()
            self.population_next = population
            self.gen_i = gen_i

        # final.store#####################################################################

        if self.store:
            self.to_csv(self.data_all)
        self.hall.items = [self.cpset.calculate_detail(indi) for indi in self.hall.items]

        return self.hall


class MultiMutateLoop(BaseLoop):
    """
    multiply mutate method.
    """

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(MultiMutateLoop, self).__init__(*args, **kwargs)

        self.register("mutate0", mutNodeReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

        self.register("mutate1", mutUniform, expr=self.gen_mu, pset=self.cpset)
        self.decorate("mutate1", staticLimit(key=operator.attrgetter("height"), max_value=2 * (self.max_value + 1)))

        self.register("mutate2", mutShrink, pset=self.cpset)

        self.register("mutate3", mutDifferentReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

        self.mutpb_list = [0.1, 0.5, 0.2, 0.2]

    def varAnd(self, population, toolbox, cxpb, mutpb):
        names = self.__dict__.keys()
        import re
        patt = r'mutate[0-9]'
        pattern = re.compile(patt)
        result = [pattern.findall(i) for i in names]
        att_name = []
        for i in result:
            att_name.extend(i)

        self.re_fresh_by_name(*att_name)

        fus = [getattr(self, i) for i in att_name]

        off = varAndfus(population, toolbox, cxpb, mutpb, fus, self.mutpb_list)

        return off


class OnePointMutateLoop(BaseLoop):
    """
    limitation height of population, just use mutNodeReplacementVerbose method.
    """

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(OnePointMutateLoop, self).__init__(*args, **kwargs)

        self.register("mutate0", mutNodeReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

        self.register("mutate3", mutDifferentReplacementVerbose, pset=self.cpset, personal_map=self.personal_map)

    def varAnd(self, population, toolbox, cxpb, mutpb):
        names = self.__dict__.keys()
        import re
        patt = r'mutate[0-9]'
        pattern = re.compile(patt)
        result = [pattern.findall(i) for i in names]
        att_name = []
        for i in result:
            att_name.extend(i)

        self.re_fresh_by_name(*att_name)

        fus = [getattr(self, i) for i in att_name]

        off = varAndfus(population, toolbox, cxpb, mutpb, fus)

        return off


class DimForceLoop(MultiMutateLoop):
    """Force select the individual with target dim for next generation"""

    def __init__(self, *args, **kwargs):
        """See also BaseLoop"""
        super(DimForceLoop, self).__init__(*args, **kwargs)
        assert self.cal_dim == True, "For DimForceLoop type, the 'cal_dim' must be True"

        self.register("select", selKbestDim,
                      dim_type=self.cpset.dim_type, fuzzy=self.cpset.fuzzy, force_number=True)


if __name__ == "__main__":
    # data
    data = load_boston()
    x = data["data"]
    y = data["target"]
    c = [6, 3, 4]
    # unit
    from sympy.physics.units import kg

    x_u = [kg] * 13
    y_u = kg
    c_u = [dless, dless, dless]

    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    z = time.time()

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[[1, 2], [3, 4], [5, 6]])
    pset0.add_constants(c, c_dim=c_dim, c_prob=None)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "Abs"))

    # a = time.time()

    bl = MultiMutateLoop(pset=pset0, gen=20, pop=2000, hall=2, batch_size=60, re_hall=2,
                         n_jobs=1, mate_prob=1, max_value=3, initial_max=1, initial_min=1,
                         mutate_prob=0.8, tq=True, dim_type="coef",
                         re_Tree=None, store=False, random_state=2,
                         stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["max"]},
                         add_coef=True, cal_dim=False, inner_add=False, vector_add=True, personal_map=False)
    # b = time.time()
    bl.run()
    bl.run(warm_start=True)

    # population = [bl.PTree(bl.genFull()) for _ in range(30)]
    # pset = bl.cpset
    # for i in population:
    #     # i.ppprint(bl.cpset)
    #     # i = "exp(gx0/gx1)"
    #
    #     i = compile_context(i, pset.context, pset.gro_ter_con, simplify=False)
    #     # print(i)
    #     # print(i)
    #     # fun = Coef("V", np.array([1.4,1.3]))
    #     # i = fun(i)
    #     # f = Function("MAdd")
    #     # i = f(i)
    #     try:
    #         # group_str(i,pset)
    #         # i=general_expr(i, pset, simplifying=True)
    #         i = general_expr(i, pset, simplifying=False)
    #         # print(i)
    #         # print(i)
    #         # pprint(i)
    #     except NotImplementedError as e:
    #         print(e)
    # # c = time.time()
    # # print(c - b, b - a, a - z)
    # a, b, c = sympy.Symbol("a"), sympy.Symbol("b"), sympy.Symbol("c")
    # print(sympy.simplify((a + (b + 1)) * c) == sympy.simplify(a * c + b * c + c))
