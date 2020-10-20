# -*- coding: utf-8 -*-

# @Time    : 2020/10/20 11:42
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from mgetool.exports import Store
from mgetool.imports import Call
from sklearn.metrics import r2_score

from Instances.pendulum.ternary_pendulum import TernaryPendulum
from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    import numpy as np

    tpen = TernaryPendulum(1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0, 0, 2, 1, 2, 1)
    st = Store()
    cal = Call()

    ########
    # data = tpen.odeint(0, 20, 0.05)
    ###############
    # st.to_pkl_pd(data,"ter1")
    ################
    data = cal.pickle_pd().ter1
    x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array, v1_array, v2_array, v3_array = data
    error = x3 - np.sin(th1_array) - 2 * np.sin(th2_array) - 3 * np.sin(th3_array)


    def func(ind):
        c = ind.fitness.values[0] >= 0.999999
        return c


    sl = SymbolLearning(loop='MultiMutateLoop', pop=1000, gen=20, mutate_prob=0.5, mate_prob=0.8, hall=1, re_hall=1,
                        re_Tree=None, initial_min=None, initial_max=2, max_value=3,
                        scoring=(r2_score,), score_pen=(1,), filter_warning=True, cv=1,
                        add_coef=True, inter_add=False, inner_add=False, vector_add=False, out_add=True,
                        flat_add=False,
                        cal_dim=False, dim_type=None, fuzzy=False, n_jobs=12, batch_size=40,
                        random_state=3,
                        stats={"h_bgp": ("mean",), "fitness": ("max",)},
                        verbose=True, migrate_prob=0,
                        tq=True, store=False, personal_map="auto", stop_condition=func, details=False,
                        classification=False,
                        score_object="y", sub_mu_max=2)
    x = np.vstack((th1_array, th2_array, th3_array)).T
    y = x3
    sl.fit(x, y, power_categories=(2, 3, 0.5, 0.333),
           categories=("Add", "Sub", "sin", "Self"), )
