# -*- coding: utf-8 -*-
import sympy

from bgp.postprocess import pprint
# @Time    : 2020/12/17 17:29
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from mgetool.tool import tt

if __name__ == "__main__":
    from sklearn.datasets import load_iris, fetch_california_housing
    from bgp.skflow import SymbolLearning
    from sklearn import metrics
    from sklearn.utils import shuffle

    # data = load_iris()
    data = fetch_california_housing()
    x = data["data"][:100]
    y = data["target"][:100]
    c = [1, 2, 3]

    #    data = pd.read_csv(r"..data/a.csv")
    #    y = data[:,0]
    #    x = data[:,1:]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=500,
                        re_hall=3,
                        gen=5, random_state=1,
                        # classification=True,
                        classification=False,
                        # scoring=[metrics.accuracy_score, ], score_pen=[1, ],
                        # store=False,
                        store=True,
                        n_jobs=4,

                        batch_size=10,
                        batch_para=False,

                        # add_coef=False,
                        # out_add =True,
                        # vector_add=True,
                        )
    tt.t
    sl.fit(x, y, c=c,categories=("Mul", "Sub", "exp", "ln")
           # x_group=[[0, 1], [2, 3]]
           )
    tt.t
    tt.p
    # score = sl.score(x, y, "r2")
    top_n = sl.loop.top_n(10)
    pprint(top_n["expr"].values[4])

    # print(sl.expr)

    # top_n.to_csv(r"./re.csv")

# if __name__ == "__main__":
#     from sklearn.datasets import fetch_california_housing
#     from bgp.skflow import SymbolLearning
#
#     data = fetch_california_housing()
#     x = data["data"][:100]
#     y = data["target"][:100]
#     c = [1, 2, 3]
#
#     sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3, cal_dim=True, re_hall=2, add_coef=True, cv=1, random_state=1
#                 )
#     sl.fit(x, y, c=c,x_group=[[1, 3], [0, 2], [4, 7]])
#     score = sl.score(x, y, "r2")
#     print(sl.expr)
