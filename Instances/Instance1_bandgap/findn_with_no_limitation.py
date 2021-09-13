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

    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=x_g)
    pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
    pset0.add_operations(power_categories=(2, 3, 0.5, 1 / 3, 4, 1 / 4),
                         categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                         self_categories=None)

    stop = lambda ind: ind.fitness.values[0] >= 0.95
    sl = SymbolLearning(loop="MultiMutateLoop", pset=pset0, gen=30, pop=1000, hall=1, batch_size=40, re_hall=5,
                        n_jobs=12, mate_prob=0.9, max_value=7, initial_min=2, initial_max=4,
                        mutate_prob=0.8, tq=False, dim_type="coef", stop_condition=stop,
                        re_Tree=0, store=False, random_state=1, verbose=True,
                        stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
                        add_coef=True, inter_add=True, flat_add=True, cal_dim=True, vector_add=True,
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
