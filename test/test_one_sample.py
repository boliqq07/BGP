# -*- coding: utf-8 -*-

# @Time    : 2020/12/16 18:53
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from sklearn import metrics
from sklearn.utils import shuffle

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from bgp.skflow import SymbolLearning

    data = load_boston()
    x = data["data"]
    y = data["target"]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50, gen=2, random_state=1)
    sl.fit(x, y)
    score = sl.score(x, y, "r2")
    print(sl.expr)

    from sklearn.datasets import load_iris
    from bgp.skflow import SymbolLearning

    data = load_iris()
    x = data["data"][:98, :]
    x[40:60] = shuffle(x[40:60], random_state=2)
    y = data["target"][:98]
    c = None

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50, gen=2, random_state=1,
                        classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ])
    sl.fit(x, y)

    print(sl.expr)

    from sklearn.datasets import load_boston
    from bgp.skflow import SymbolLearning
    from sklearn import metrics

    data = load_boston()
    x = data["data"]
    y = data["target"]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50, gen=2, random_state=1,
                        scoring=[metrics.mean_absolute_error, ],
                        score_pen=[-1, ],
                        # stats={"fitness_dim_min": ("min",), "dim_is_target": ("sum",)},
                        )
    sl.fit(x, y)
    print(sl.expr)

    from sklearn.datasets import load_boston
    from bgp.skflow import SymbolLearning

    data = load_boston()
    x = data["data"]
    y = data["target"]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50, gen=2, random_state=1)
    sl.fit(x, y, x_group=[[1, 2], [3, 4], [6, 7]])
    score = sl.score(x, y, "r2")
    print(sl.expr)

    from bgp.functions.dimfunc import dless
    from sklearn.datasets import load_boston
    from bgp.skflow import SymbolLearning

    data = load_boston()
    x = data["data"]
    y = data["target"]
    x_dim = [dless, dless, dless, dless, dless, dless, dless, dless, dless, dless, dless, dless, dless]
    y_dim = dless

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50, gen=2, random_state=1, cal_dim=True, dim_type="coef")
    sl.fit(x, y, x_dim=x_dim, y_dim=y_dim)
    score = sl.score(x, y, "r2")
    print(sl.expr)
