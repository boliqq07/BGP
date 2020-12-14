import sympy
from sklearn import metrics

from bgp.base import CalculatePrecisionSet
from bgp.base import SymbolSet
from bgp.calculation.scores import calculate_score

#
# class MyTestbase(unittest.TestCase):
#
#     def setUp(self):
#         self.SymbolTree = SymbolTree
#         self.pset = SymbolSet()
#
#         from sklearn.datasets import load_boston
#
#         import numpy as np
#         x=np.array([[1,2,3,4,5,6,7,8,9,9,10,9,7,5,3,1],
#            [1,2,3,4,4,3,2,4,5,6,7,8,9,10,12,15],
#            [2,3,4,8,12,16,30,32,33,30,20,10,5,3]])
#         y=x[:,1]*x[:,2]
#         self.x = x
#         self.y = y
#
#         self.pset.add_features(x, y)
#         self.pset.add_operations(
#                                  categories=("Add", "Mul", "Self", "Abs"),
#                                  self_categories=None)
#
#         from sklearn.metrics import r2_score, mean_squared_error
#         self.cp = CalculatePrecisionSet(self.pset, scoring=[r2_score, mean_squared_error],
#                                         score_pen=[1, -1],
#                                         filter_warning=True)
#
#     def test_pset_passed_to_cpset_will_change(self):
#         x1,x2,x3 = sympy.symbols("x1, x2, x3")
#         expr01=x1*x2
#         y = calculate_derivative_y(expr01, self.cp.data_x, self.cp.terminals_and_constants_repr, np_maps=self.cp.np_map)


if __name__ == '__main__':
    # unittest.main()
    import numpy as np

    x = np.array([[10, 6, 3, 4, 5, 6, 7, 8, 9, 9, 10, 9, 7, 5, 3, 1],
                  [1, 2, 3, 4, 4, 3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15],
                  [2, 3, 4, 8, 12, 16, 30, 32, 33, 30, 20, 10, 5, 3, 2, 1]]).T
    x[:, 2] = x[:, 0] / x[:, 1]
    y = np.zeros(x.shape[0])
    x = x
    y = y

    pset = SymbolSet()

    pset.add_features(x, y)
    pset.add_operations(
        categories=("Add", "Mul", "Self", "Abs"),
        self_categories=None)

    from sklearn.metrics import r2_score, mean_squared_error

    cp = CalculatePrecisionSet(pset, scoring=[r2_score, mean_squared_error],
                               score_pen=[1, -1],
                               filter_warning=True)
    x0, x1, x2 = sympy.symbols("x0, x1, x2")

    # t=Function("t")
    # expr00 = (x2*x1+x0*x2*2).subs(x0, t(x1))
    # dv1 = sympy.diff(expr00, x1, evaluate=True)
    # dv1 = dv1.subs(t(x1), x0)
    #
    # t = Function("t")
    # expr00 = (x2*x1+x0*x2*2).subs(x1, t(x0))
    # dv2 = sympy.diff(expr00, x0, evaluate=True)
    # dv2 = dv2.subs(t(x0), x1)
    #
    # k = dv1/dv2
    #
    # func0 = sympy.utilities.lambdify(cp.terminals_and_constants_repr, k, modules=[cp.np_map, "numpy"])
    # pre_y = func0(*x.T)
    #
    # ff = sympy.diff(x0, x1, evaluate=False)
    # func0 = sympy.utilities.lambdify(cp.terminals_and_constants_repr, ff, modules=[cp.np_map, "numpy"])
    # dy = func0(*x.T)

    # pre_y, dy = calculate_derivative_y(x2*x1+x0*x2*2, x.T, cp.terminals_and_constants_repr, np_maps=cp.np_map)

    score = calculate_score((x0 / x1) - x2, x.T, y, cp.terminals_and_constants_repr,
                            scoring=[metrics.mean_absolute_error], score_pen=(-1,),
                            add_coef=False, filter_warning=True, inter_add=True,
                            inner_add=False, vector_add=False, out_add=False, flat_add=False, np_maps=cp.np_map,
                            classification=False, score_object="dy", details=False)

    #
    # import matplotlib.pyplot as plt
    # bp = BasePlot()
    # bp.lines(pre_y.reshape(-1,1))
    # plt.show()
    # bp = BasePlot()
    # bp.lines(dy.reshape(-1,1))
    # plt.show()
