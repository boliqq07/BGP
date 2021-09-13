import numpy as np
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.show import BasePlot
from sympy.physics.units import eV

from bgp.base import SymbolSet
from bgp.functions.dimfunc import Dim
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
    x_g = x_g.reshape(-1, 2)
    x_g = list(x_g[1:])

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=x_g)
    pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    height = 2
    h_bgp = 1

    # stop = None
    # This random_state is under Linux system. For others system ,the random_state maybe different。
    # try with different random_state.
    stop = lambda ind: ind.fitness.values[0] >= 0.897
    sl = SymbolLearning(loop="OnePointMutateLoop", pset=pset0, gen=10, pop=1000, hall=1, batch_size=40, re_hall=3,
                        n_jobs=12, mate_prob=0.9, max_value=h_bgp, initial_min=1, initial_max=h_bgp,
                        mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,
                        re_Tree=0, store=False, random_state=1, verbose=True,
                        stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
                        add_coef=True, inter_add=True, out_add=True, cal_dim=True, vector_add=True,
                        flat_add=False,
                        personal_map=False)

    sl.fit()
    score = sl.score(x, y, "r2")
    print(sl.expr)
    y_pre = sl.predict(x)

    y_pre = si_transformer.scale_y * y_pre
    ssc = Dim.inverse_convert(y_dim, target_units=eV)[0]
    y_pre = y_pre * ssc

    p = BasePlot(font=None)
    p.scatter(Y, y_pre, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    import matplotlib.pyplot as plt

    plt.show()

    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()

    XX = (X[:, 1] ** (-1) * X[:, 25] - X[:, 1] ** (-1) * X[:, 24]).reshape(-1, 1)
    lin.fit(XX, Y)
    coef = lin.coef_
    inter = lin.intercept_
    score2 = lin.score(XX, Y)

    """exhaustion show all under 2, get the best, some parameter are shrinking for calculated"""

    # def find_best():
    #     bl = sl
    #     pset = bl.loop.cpset
    #     prim = pset.primitives
    #     pset = pset
    #
    #     ter = pset.terminals_and_constants
    #     prim1 = [_ for _ in prim if _.arity == 1]
    #     prim2 = [_ for _ in prim if _.arity == 2]
    #     dispose = pset.dispose
    #
    #     top = [i for i in dispose if i.name not in ["Self", "Conv"]]
    #     pop_all2 = product(top, prim2, dispose, ter, dispose, ter)
    #
    #     pop_all1 = product(top, prim1, dispose, ter)
    #     pop_all = chain(pop_all1, pop_all2)
    #     pop_all = list(pop_all)
    #     tt.t1
    #     pop_all = [SymbolTree(i) for i in pop_all]
    #     tt.t2
    #     invalid_ind_score = pset.parallelize_score(pop_all)
    #     tt.t3
    #     score = [(i[0] * i[2]) for i in invalid_ind_score]
    #     tt.t4
    #     score = np.array(score)
    #     index = np.argmax(score)
    #     tt.t5
    #     tt.p
    #     score_best = score[index]
    #     pop_all_best = pop_all[int(index)]
    #
    #     i = compile_context(pop_all_best, pset.context, pset.gro_ter_con, simplify=True)
    #     expr = general_expr_dict(i, pset.expr_init_map, pset.free_symbol,
    #                              pset.gsym_map, simplifying=True)
    #
    #     return len(pop_all), expr
    #
    # a = find_best()
    """end"""
