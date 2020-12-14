# -*- coding: utf-8 -*-

# @Time    : 2020/10/20 11:42
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import tt
from numpy import random
from sklearn.metrics import r2_score

from Instances.pendulum.ternary_pendulum import TernaryPendulum
from bgp.iteration.newpoints import new_points, search_space
from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    import numpy as np

    tpen = TernaryPendulum(1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0, 0, 2, 1, 2, 1)
    st = Store()
    cal = Call()

    ########
    # data = tpen.odeint(0, 20, 0.25)
    ###############
    # st.to_pkl_pd(data, "ter1")
    ################
    data = cal.pickle_pd().ter1
    x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array, _, _, _ = data
    # error = x3 - np.sin(th1_array) - 2 * np.sin(th2_array) - 3 * np.sin(th3_array)

    random.seed(2)
    th3_array = (0.0005 * random.random(th3_array.shape) + 1) * th3_array

    x = np.vstack((th1_array, th2_array, th3_array, y1, y2, y3)).T
    y = x3


    def func(ind):
        c = ind.fitness.values[0] >= 0.999999
        return c


    sl = SymbolLearning(loop='MultiMutateLoop', pop=1000, gen=10, mutate_prob=0.5, mate_prob=0.8, hall=1, re_hall=1,
                        re_Tree=None, initial_min=1, initial_max=2, max_value=3,
                        scoring=(r2_score,), score_pen=(1,), filter_warning=True, cv=1,
                        add_coef=True, inter_add=False, inner_add=False, vector_add=False, out_add=True,
                        flat_add=False,
                        cal_dim=False, dim_type=None, fuzzy=False, n_jobs=1, batch_size=40,
                        random_state=1, store=False,
                        stats={"h_bgp": ("max",), "fitness": ("max",)},
                        verbose=True, migrate_prob=0,
                        tq=True, personal_map="auto", stop_condition=func, details=False,
                        classification=False,
                        score_object="y", sub_mu_max=1)
    tt.t
    sl.fit(x, y, power_categories=(2, 3, 0.5, 0.333),
           categories=("Add", "Sub", "sin", "cos", "Self"), )
    for _ in range(2):
        xx = search_space(np.arange(0, 1, 0.1), np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), )
        x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array = tpen.odeint_x(*xx.T)
        xx = np.vstack((xx[:, 0], xx[:, 1], xx[:, 2], y1, y2, y3)).T

        newx, new_y = new_points(sl.loop, xx, method="get_max_std", resample_number=20, n=3)

        x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array = tpen.odeint_x(*newx[:, :3].T)
        n_x = np.vstack((th1_array, th2_array, th3_array, y1, y2, y3)).T
        n_y = x3

        x = np.concatenate((x, n_x), axis=0)
        y = np.concatenate((y, n_y), axis=0)

        sl.fit(x, y, power_categories=(2, 3, 0.5, 0.333),
               categories=("Add", "Sub", "sin", "cos", "Self"), warm_start=True)

        datas = sl.loop.top_n(10)
    tt.t
    tt.p
