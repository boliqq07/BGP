import unittest

import numpy

from bgp.base import CalculatePrecisionSet
from bgp.base import SymbolSet
from bgp.base import SymbolTree
from bgp.functions.dimfunc import dless


class MyTestbase(unittest.TestCase):

    def setUp(self):
        self.SymbolTree = SymbolTree
        self.pset = SymbolSet()

        from sklearn.datasets import load_boston

        data = load_boston()
        x = data["data"]
        y = data["target"]
        # No = Normalizer()
        # y=y/max(y)
        # x = No.fit_transform(x)
        self.x = x
        self.y = y
        # self.pset.add_features(x, y, )
        self.pset.add_features(x, y, x_group=[[1, 2], [4, 5]])
        self.pset.add_constants([6, 3, 4], c_dim=[dless, dless, dless], c_prob=None)
        self.pset.add_operations(power_categories=(2, 3, 0.5),
                                 categories=("Add", "Mul", "Self", "Abs"),
                                 self_categories=None)

        from sklearn.metrics import r2_score, mean_squared_error
        self.cp = CalculatePrecisionSet(self.pset, scoring=[r2_score, mean_squared_error],
                                        score_pen=[1, -1],
                                        filter_warning=True)

    def test_pset_passed_to_cpset_will_change(self):
        cp = CalculatePrecisionSet(self.pset)
        self.assertNotEqual(cp, self.cp)

    def test_tree_gengrow_repr_and_str_different(self):
        from numpy import random
        random.seed(1)
        sl = SymbolTree.genGrow(self.pset, 3, 4)
        print(sl)
        # self.assertNotEqual(repr(sl), str(sl))

    def test_add_tree_back(self):
        from numpy import random
        random.seed(1)
        sl = SymbolTree.genGrow(self.pset, 3, 4)
        self.pset.add_tree_to_features(sl)

    #
    def test_barch_tree(self):
        from numpy import random
        random.seed(1)
        for i in range(10):

            sl = SymbolTree.genGrow(self.pset, 3, 4)
            cpsl = self.cp.calculate_detail(sl)
            self.assertIsNotNone(cpsl.y_dim)
            self.assertIsNotNone(cpsl.expr)
            self.assertIsNone(cpsl.p_name)
            if cpsl.pre_y is not None:
                self.assertIsInstance(cpsl.pre_y, numpy.ndarray)
                self.assertEqual(cpsl.pre_y.shape, self.y.shape)
                print(cpsl.coef_pre_y[:3])
                print(cpsl.pre_y[:3])
                print(cpsl.coef_score)
                print(cpsl.coef_expr)
                print(cpsl.pure_expr)

    def test_depart_tree(self):
        from numpy import random
        random.seed(1)
        for i in range(10):

            sl = SymbolTree.genGrow(self.pset, 5, 6)
            sl_departs = sl.depart()
            for i in sl_departs:
                cpsl = self.cp.calculate_simple(i)
                self.assertIsNotNone(cpsl.y_dim)
                self.assertIsNotNone(cpsl.expr)
                self.assertIsNone(cpsl.p_name)


if __name__ == '__main__':
    unittest.main()
