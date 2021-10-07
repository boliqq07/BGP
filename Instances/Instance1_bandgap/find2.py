import numpy as np
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import tt

from bgp.base import SymbolSet
from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    import os

    os.chdir(r'band_gap')
    data = Call()
    name_and_abbr = data.csv().name_and_abbr

    SL_data = data.SL_data
    si_transformer = data.si_transformer

    store = Store()

    x, x_dim, y, y_dim, c, c_dim, X, Y = SL_data
    x_g = np.arange(x.shape[1])
    x_g = list(x_g[1:])

    x_g = x_g.reshape(-1, 2)
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=x_g)
    pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5, 1 / 3, 4, 1 / 4),
                         # categories=("Mul",),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    total_height = 3
    h_bgp = 2
    # This random_state is under Linux system. For others system ,the random_state maybe different,please
    # try with different random_state.
    for i in range(1, 10):
        stop = lambda ind: ind.fitness.values[0] >= 0.95
        sl = SymbolLearning(loop="MultiMutateLoop", pset=pset0, gen=20, pop=1000, hall=1, batch_size=40, re_hall=3,
                            n_jobs=12, mate_prob=0.9, max_value=h_bgp, initial_min=2, initial_max=h_bgp,
                            mutate_prob=0.8, tq=False, dim_type="coef", stop_condition=stop,
                            re_Tree=0, store=False, random_state=4, verbose=True,
                            # stats=None,
                            stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
                            add_coef=True, inter_add=True, out_add=True, cal_dim=True, vector_add=True,
                            personal_map=False)
        tt.t
        sl.fit()
        tt.t
        tt.p
        score = sl.score(x, y, "r2")
        print(sl.expr)
        y_pre = sl.predict(x)
        break

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

    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()

    XX = np.vstack((X[:, 1] ** (0.333), X[:, 24], X[:, 25])).T
    lin.fit(XX, Y)
    coef = lin.coef_
    inter = lin.intercept_
    score = lin.score(XX, Y)
