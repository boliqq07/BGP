# -*- coding: utf-8 -*-

# @Time    : 2021/6/22 11:04
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import numpy as np
import pandas as pd
from mgetool.exports import Store
from mgetool.show import BasePlot
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from bgp.base import SymbolSet
from bgp.functions.dimfunc import dless
from bgp.skflow import SymbolLearning


def search_space(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


csv = pd.read_csv("LCB-WCX2.csv",index_col=0)

y = csv.values.ravel(order="A")

ind = csv.index.values.astype(float)
col = np.array([i.replace(".1","") for i in csv.columns.values]).astype(float)

X = search_space(ind,col)
#
data = np.concatenate((X,y.reshape(-1,1)),axis=1).astype(float)




#
# if __name__ == "__main__":
#
#     half = data.shape[0]//2
#     q1 = data[:half]
#     q2 = data[half:]
#     #q1
#     X = q1[:, :-1]
#     y = q1[:, -1]
#     X = np.concatenate((X, (X[:, 1] / X[:, 0]).reshape(-1, 1)), axis=1)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)
#
#     store = Store()
#
#     # symbolset
#     pset0 = SymbolSet()
#     pset0.add_features(X_train,y_train)
#     # pset0.add_constants(c=[1, ])
#     #pset0.add_operations(power_categories=(2,0.5),
#     pset0.add_operations(
#         # power_categories=(2,),
#                          categories=("exp","Mul","Sub"),
#                          self_categories=None)
#
#     h_bgp = 3
#
#     # stop = None
#     # This random_state is under Linux system. For others system ,the random_state maybe different。
#     # try with different random_state.
#     stop = lambda ind: ind.fitness.values[0] >= 0.999
#     sl = SymbolLearning(loop ='MultiMutateLoop',pset=pset0, gen=10, pop=3000, hall=1, batch_size=40, re_hall=5,
#                         n_jobs=12, mate_prob=0.8, max_value=h_bgp, initial_min=1, initial_max=h_bgp,
#                         mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,
#                         re_Tree=0, random_state=2, verbose=True,
#                         add_coef=True, inter_add=True,  cal_dim=False, inner_add=True,
#                         personal_map=False)
#
#     sl.fit()
#     score = sl.score(X_test, y_test, "r2")
#     print(sl.expr)
#
#     y_pre = sl.predict(X_test)
#
#     p = BasePlot(font=None)
#     p.scatter(y_test, y_pre, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
#     import matplotlib.pyplot as plt
#
#     plt.show()
#
#
#     y_pre = sl.predict(X)
#     score_all = sl.score(X, y, "r2")
#     p = BasePlot(font=None)
#     p.scatter(y, y_pre, strx='Experimental $Frequency$', stry='Calculated $Frequency$')
#     import matplotlib.pyplot as plt
#
#     plt.show()

if __name__ == "__main__":
    data = data[np.where(data[:, 1] != 0.006)]

    half = data.shape[0] // 2
    q1 = data[:half]
    q2 = data[half:]
    #q1
    X = q2[:, :-1]
    y = q2[:, -1]
    X = np.concatenate((X, (X[:, 1] / X[:, 0]).reshape(-1, 1)), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)

    store = Store()

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(X_train,y_train)
    # pset0.add_constants(c=[1, ])
    #pset0.add_operations(power_categories=(2,0.5),
    pset0.add_operations(
        # power_categories=(2,),
                         categories=("exp","Mul","Sub"),
                         self_categories=None)

    h_bgp = 3

    # stop = None
    # This random_state is under Linux system. For others system ,the random_state maybe different。
    # try with different random_state.
    stop = lambda ind: ind.fitness.values[0] >= 0.999
    sl = SymbolLearning(loop ='MultiMutateLoop',pset=pset0, gen=10, pop=3000, hall=1, batch_size=40, re_hall=5,
                        n_jobs=12, mate_prob=0.8, max_value=h_bgp, initial_min=1, initial_max=h_bgp,
                        mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,
                        re_Tree=0, random_state=2, verbose=True,
                        add_coef=True, inter_add=True,  cal_dim=False, inner_add=True,
                        personal_map=False)

    sl.fit()
    score = sl.score(X_test, y_test, "r2")
    print(sl.expr)

    y_pre = sl.predict(X_test)

    p = BasePlot(font=None)
    p.scatter(y_test, y_pre, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    import matplotlib.pyplot as plt

    plt.show()


    y_pre = sl.predict(X)
    score_all = sl.score(X, y, "r2")
    p = BasePlot(font=None)
    p.scatter(y, y_pre, strx='Experimental $Frequency$', stry='Calculated $Frequency$')
    import matplotlib.pyplot as plt

    plt.show()
