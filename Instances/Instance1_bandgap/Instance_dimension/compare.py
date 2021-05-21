import numpy as np
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import tt
from sklearn.metrics import r2_score

from bgp.base import SymbolSet
from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    import os

    os.chdir(r'../band_gap')
    data = Call()
    name_and_abbr = data.csv().name_and_abbr

    SL_data = data.SL_data
    si_transformer = data.si_transformer

    store = Store()

    x, x_dim, y, y_dim, c, c_dim, X, Y = SL_data
    x_g = np.arange(x.shape[1])
    x_g = x_g.reshape(-1, 2)
    x_g = list(x_g[1:])

    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=x_g)
    pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5, 1 / 3, 4, 1 / 4),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    total_height = 3
    h_bgp = 2
    # This random_state is under Linux system. For others system ,the random_state maybe different,please
    # try with different random_state.
    for i in range(1, 10):

        stop = lambda ind: ind.fitness.values[0] >= 0.99
        sl = SymbolLearning(loop="MultiMutateLoop", pset=pset0, gen=30, pop=1000, hall=1, batch_size=40, re_hall=3,
                            n_jobs=12, mate_prob=0.9, max_value=h_bgp, initial_min=2, initial_max=h_bgp,
                            mutate_prob=0.8, tq=False, dim_type="ignore", stop_condition=stop,
                            re_Tree=0, store=False, random_state=4, verbose=True,
                            # stats=None,
                            stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
                            add_coef=True, inter_add=True, out_add=True, cal_dim=True, vector_add=False,
                            personal_map=False)
        tt.t
        sl.fit()
        tt.t
        tt.p
        score = sl.score(x, y, "r2")
        print(sl.expr)
        y_pre = sl.predict(x)
        break

    # def _mape(y, y_pred, w):
    #
    #     return r2_score(y_true=y, y_pred=y_pred,sample_weight=w)
    #
    # mape = make_fitness(_mape, greater_is_better=True)
    # est = SymbolicRegressor(population_size=1000,
    #              generations=200,
    #              tournament_size=5,
    #              stopping_criteria=1,
    #              const_range=(-1., 1.),
    #              init_depth=(2, 5),
    #              init_method='half and half',
    #              function_set=('add', 'sub', 'mul', 'div',"log"),
    #              metric=mape,
    #              parsimony_coefficient=0.001,
    #              p_crossover=0.9,
    #              p_subtree_mutation=0.01,
    #              p_hoist_mutation=0.01,
    #              p_point_mutation=0.01,
    #              p_point_replace=0.05,
    #              max_samples=1.0,
    #              feature_names=None,
    #              warm_start=False,
    #              low_memory=False,
    #              n_jobs=12,
    #              verbose=True,
    #              random_state=0)
    # est.fit(x, y)




    # just for shown
    # y_pre = si_transformer.scale_y * y_pre
    # ssc = Dim.inverse_convert(y_dim, target_units=eV)[0]
    # y_pre = y_pre * ssc
    #
    # p = BasePlot(font=None)
    # p.scatter(Y, y_pre, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    # import matplotlib.pyplot as plt
    #
    # plt.show()


