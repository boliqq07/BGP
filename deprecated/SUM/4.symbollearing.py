# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 16:10
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
from BGP.combination.common import custom_loss_func, calculateExpr
from BGP.combination.dictbase import FixedSetFill
from BGP.combination.dim import Dim
from BGP.combination.symbolunit import mainPart
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.show import BasePlot
from mgetool.tool import name_to_name
from sklearn.metrics import r2_score
from sklearn.utils import shuffle

# 4
# if __name__ == '__main__':
#     import pandas as pd
#
#     store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\4.symbol')
#     datamnist = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
#     data_import = datamnist.csv().all_import
#     name_init, abbr_init = datamnist.name_and_abbr
#
#     select = ['electron density', 'energy cohesive brewer', 'distance core electron(schubert)']
#
#     X_frame_abbr = name_to_name(name_init, abbr_init, search=select, search_which=1, return_which=2,
#                                 two_layer=False)
#
#     select = ['electron density'] + [j + "_%i" % i for j in select[1:] for i in range(2)]
#
#     select_abbr = ['$\\rho_e$'] + [j + "_%i" % i for j in X_frame_abbr[1:] for i in range(2)]
#
#     data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
#
#     X_frame = data225_import[select]
#     y_frame = data225_import['exp_gap']
#
#     X = X_frame.values
#     y = y_frame.values
#
#     # scal = preprocessing.MinMaxScaler()
#     # X = scal.fit_transform(X)
#     X, y = shuffle(X, y, random_state=5)
#
#     dim1 = Dim([0, -3, 0, 0, 0, 0, 0])
#     dim2 = Dim([1, 2, -2, 0, 0, 0, 0])
#     dim3 = Dim([1, 2, -2, 0, 0, 0, 0])
#     dim4 = Dim([0, 1, 0, 0, 0, 0, 0])
#     dim5 = Dim([0, 1, 0, 0, 0, 0, 0])
#     target_dim = [Dim([1, 2, -2, 0, 0, 0, 0])]
#
#     dim_list = [dim1, dim2, dim3, dim4, dim5]
#
#     pset = FixedSetFill(x_name=select_abbr, power_categories=[1 / 3, 1 / 2, 2, 3],
#                         categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Self", "Abs", "Neg", "Rem"),
#                         partial_categories=None, self_categories=None, dim=dim_list, max_=5,
#                         definate_operate=[
#                             [-13, ['Mul', 'Div']],
#                             [-12, ['Mul', 'Div']],
#
#                             [-11, ["Self"]],
#                             [-10, [0, 1, 2, 3, "Self", "Abs"]],
#                             [-9, [0, 1, 2, 3, "Self", "Abs","log"]],
#
#                             [-8, ["Self"]],
#                             [-7, ["Sub"]],
#                             [-6, ['Sub',"Div"]],
#
#                             [-5, [0, 1, 2, 3, "Self"]],
#                             [-4, [0, 1, 2, 3, "Self"]],
#                             [-3, [0, 1, 2, 3, "Self"]],
#
#                             [-2, [0, 1, 2, 3, "Self"]],
#                             [-1, [0, 1, 2, 3, "Self"]],
#
#                         ],
#                         definate_variable=[[-5, [0]],
#                                            [-4, [1]],
#                                            [-3, [2]],
#                                            [-2, [3]],
#                                            [-1, [4]]],
#                         operate_linkage=[[-1, -2], [-3, -4]],
#                         variable_linkage=None)
#     result = mainPart(X, y, pset, pop_n=500, random_seed=1, cxpb=0.8, mutpb=0.6, ngen=20, tournsize=3, max_value=10,
#                       double=False, score=[r2_score, custom_loss_func], target_dim=target_dim)

# 5
if __name__ == '__main__':
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\4.symbol')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
    data_import = data.csv().all_import
    name_init, abbr_init = data.name_and_abbr

    select = ['latent heat of fusion', 'valence electron number']

    X_frame_abbr = name_to_name(name_init, abbr_init, search=select, search_which=1, return_which=2,
                                two_layer=False)

    select = [j + "_%i" % i for j in select[:] for i in range(2)]

    select_abbr = [j + "_%i" % i for j in X_frame_abbr[:] for i in range(2)]

    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    # scal = preprocessing.MinMaxScaler()
    # X = scal.fit_transform(X)
    X, y = shuffle(X, y, random_state=5)

    dim1 = Dim([1, 2, -2, 0, 0, 0, 0])
    dim2 = Dim([1, 2, -2, 0, 0, 0, 0])
    dim3 = Dim([0, 0, 0, 0, 0, 0, 0])
    dim4 = Dim([0, 0, 0, 0, 0, 0, 0])

    target_dim = [Dim([1, 2, -2, 0, 0, 0, 0])]
    dim_list = [dim1, dim2, dim3, dim4]
    # GP
    # pset = ExpressionSetFill(x_name=select, power_categories=[2, 3], categories=("Add", "Mul", "exp"),
    #                          partial_categories=None, self_categories=None, dim=dim_list)
    # GEP
    pset = FixedSetFill(x_name=select_abbr, power_categories=[1 / 3, 1 / 2, 2, 3],
                        categories=('Add', 'Sub', 'Mul', 'Div', "Rec", 'exp', "log", "Self", "Abs", "Neg", "Rem"),
                        partial_categories=None, self_categories=None, dim=dim_list, max_=4,
                        definate_operate=[
                            [-12, [0, 1, 2, 3, "Self"]],
                            [-11, ['Mul', 'Div']],

                            [-10, [0, 1, 2, 3, "Self"]],
                            [-9, [0, 1, 2, 3, "Self"]],

                            [-8, ["Self", "Rec", 'Rem', "log", "exp"]],
                            [-8, ["Self", "Rec", "Rem", "log", "exp"]],

                            [-6, ['Div', "Sub"]],
                            [-5, ['Div']],

                            [-4, [0, 1, 2, 3, "Self"]],
                            [-3, [0, 1, 2, 3, "Self"]],

                            [-2, [0, 1, 2, 3, "Self"]],
                            [-1, [0, 1, 2, 3, "Self"]],
                        ],
                        definate_variable=[
                            [-4, [0]],
                            [-3, [1]],
                            [-2, [2]],
                            [-1, [3]]],
                        operate_linkage=[[-1, -2], [-3, -4]],
                        variable_linkage=None)
    #
    result = mainPart(X, y, pset, pop_n=500, random_seed=1, cxpb=1, mutpb=0.6, ngen=30, tournsize=3, max_value=10,
                      max_=4,
                      double=False, score=[r2_score, custom_loss_func], inter_add=False, target_dim=target_dim)

    import sympy

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]

    X_frame = data216_import[select]
    y_frame = data216_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    x0 = sympy.Symbol("x0")
    x1 = sympy.Symbol("x1")
    x2 = sympy.Symbol("x2")
    x3 = sympy.Symbol("x3")

    #
    expr01 = (x0 ** 0.5 - x1 ** 0.5 + 1) ** 2 * sympy.log(x2 / x3) ** 2

    terminals = [x0, x1, x2, x3]
    score, expr01 = calculateExpr(expr01, X, y, terminals, scoring=None, add_coeff=True,
                                  del_no_important=False, filter_warning=True, inter_add=False, iner_add=True,
                                  random_add=None)
    x = X
    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]
    x3 = x[:, 3]

    t = expr01
    func0 = sympy.utilities.lambdify(terminals, t)
    re = func0(*x.T)
    p = BasePlot(font=None)
    p.scatter(y, re, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    import matplotlib.pyplot as plt

    plt.show()
