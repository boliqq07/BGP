# -*- coding: utf-8 -*-

# @Time    : 2020/1/14 16:28
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer
from sklearn.metrics import r2_score

dataC = pd.read_excel(r"C:\Users\Administrator\Desktop\wxx\HER-Feature.xlsx", sheet_name="C", index_col=0)
dataN = pd.read_excel(r"C:\Users\Administrator\Desktop\wxx\HER-Feature.xlsx", sheet_name="N", index_col=0)
data = dataC
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values


# X = normalize(X)
# pset = FixedSetFill(x_name=["n1","n2","n3","n4"], power_categories=[1 / 3, 1 / 2, 2, 3, 4,5,6],
#                     categories=('Add', 'Mul', 'Div', "Rec", "Rem"),
#                     partial_categories=None, self_categories=None, dim=None, max_=3,
#
#                     definate_variable=[
#                                        # [-4, [3]],
#                                        [-3, [2]],
#                                        [-2, [1]],
#                                        [-1, [0]]],
#                     variable_linkage=None)
# result = mainPart(X, y, pset, pop_n=500, random_seed=6, cxpb=0.5, mutpb=0.5, ngen=10, tournsize=3, max_value=10,max_=3,
#                   double=False, score=[r2_score,custom_loss_func], iner_add=True, target_dim=None,cal_dim=False,store=False)


def _mape(y, y_pred, w):
    """Calculate the mean absolute percentage error."""
    return r2_score(y, y_pred, w)


mape = make_fitness(_mape, greater_is_better=True)

# X = normalize(X)

# sr = SymbolicRegressor(population_size=1000, generations=50, tournament_size=100, stopping_criteria=0.1,
#                        const_range=(-1.0, 1.0), init_depth=(4, 6), init_method='half and half',
#                        function_set=('add', 'sub', 'mul', 'div',"log"), metric=mape,
#                        parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01,
#                        p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05,
#                        max_samples=1.0, feature_names=None, warm_start=False, low_memory=False,
#                        n_jobs=1, verbose=0, random_state=7)

sr = SymbolicTransformer(population_size=1000,
                         hall_of_fame=100,
                         n_components=10,
                         generations=20,
                         tournament_size=20,
                         stopping_criteria=1.0,
                         const_range=(-1., 1.),
                         init_depth=(2, 6),
                         init_method='half and half',
                         function_set=('add', 'sub', 'mul', 'div'),
                         metric=mape,
                         parsimony_coefficient=0.001,
                         p_crossover=0.9,
                         p_subtree_mutation=0.01,
                         p_hoist_mutation=0.01,
                         p_point_mutation=0.01,
                         p_point_replace=0.05,
                         max_samples=1.0,
                         feature_names=None,
                         warm_start=False,
                         low_memory=False,
                         n_jobs=1,
                         verbose=0,
                         random_state=None)

tran = sr.fit_transform(X, y)
print(sr._best_programs[0])
print(sr._best_programs[0].fitness_)
# print(sr._program)
# pre = sr.predict(X)
#
# #
# bp = BasePlot()
# bp.scatter(y, pre, strx='y_true', stry='y_predict')
# plt.show()
