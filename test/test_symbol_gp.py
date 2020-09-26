import operator
import unittest

from mgetool import newclass
from mgetool.packbox import Toolbox

from bgp.base import CalculatePrecisionSet
from bgp.base import SymbolSet
from bgp.base import SymbolTree
from bgp.functions.dimfunc import dless
from bgp.gp import cxOnePoint, varAnd, genGrow, staticLimit, mutShrink, selKbestDim


class MyTestgp(unittest.TestCase):

    def setUp(self):
        self.SymbolTree = SymbolTree
        self.pset = SymbolSet()

        from sklearn.datasets import load_boston

        data = load_boston()
        x = data["data"]
        y = data["target"]

        self.x = x
        self.y = y
        # self.pset.add_features(x, y, )
        self.pset.add_features(x, y, x_group=[[1, 2], [4, 5]])
        self.pset.add_constants([6, 3, 4], c_dim=[dless, dless, dless], c_prob=None)
        self.pset.add_operations(power_categories=(2, 3, 0.5),
                                 categories=("Add", "Mul", "Neg", "Abs"),
                                 self_categories=None)

        from sklearn.metrics import r2_score, mean_squared_error

        self.cp = CalculatePrecisionSet(self.pset, scoring=[r2_score, mean_squared_error],
                                        score_pen=[1, -1], dim_type=None,
                                        filter_warning=True)

    def test_gp_flow(self):
        from numpy import random
        random.seed(1)
        cpset = self.cp
        # def Tree
        from deap.base import Fitness

        Fitness_ = newclass.create("Fitness_", Fitness, weights=(1, -1))
        PTree_ = newclass.create("PTrees_", SymbolTree, fitness=Fitness_)

        # def selection
        toolbox = Toolbox()

        # toolbox.register("select", selTournament, tournsize=3)
        toolbox.register("select", selKbestDim, dim_type=dless)
        # selBest
        toolbox.register("mate", cxOnePoint)
        # def mutate
        toolbox.register("generate", genGrow, pset=cpset, min_=2, max_=3)
        # toolbox.register("mutate", mutUniform, expr=toolbox.generate, pset=cpset)
        # toolbox.register("mutate", mutNodeReplacement, pset=cpset)
        toolbox.register("mutate", mutShrink, pset=cpset)

        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=10))
        # def elaluate

        # toolbox.register("evaluate", cpset.parallelize_calculate, n_jobs=4, add_coef=True,
        # inter_add=False, inner_add=False)

        # toolbox.register("parallel", parallelize, n_jobs=1, func=toolbox.evaluate, respective=False, tq=False)

        population = [PTree_.genGrow(cpset, 3, 4) for _ in range(10)]
        # si = sys.getsizeof(cpset)
        for i in range(5):
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            invalid_ind_score = cpset.parallelize_score(inds=invalid_ind)

            for ind, score in zip(invalid_ind, invalid_ind_score):
                ind.fitness.values = score[0]
                ind.y_dim = score[1]
            # si2 = sys.getsizeof(invalid_ind[0])
            # invalid_ind=[i.compress() for i in invalid_ind]
            # si3 = sys.getsizeof(invalid_ind[0])
            # print(si3,si2,si)
            population = toolbox.select(population, len(population))
            offspring = varAnd(population, toolbox, 1, 1)
            population[:] = offspring
            # cpsl.compress()


# if __name__ == '__main__':
#
#     unittest.main()

if __name__ == "__main__":
    import time

    a = time.time()
    se = MyTestgp()
    se.setUp()
    b = time.time()
    se.test_gp_flow()
    c = time.time()
