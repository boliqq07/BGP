# -*- coding: utf-8 -*-

# @Time    : 2020/12/17 17:29
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from bgp.skflow import SymbolLearning
    from sklearn import metrics
    from sklearn.utils import shuffle

    data = load_iris()
    x = data["data"]
    y = data["target"]
    c = [1, 2, 3]

    #    data = pd.read_csv(r"..data/a.csv")
    #    y = data[:,0]
    #    x = data[:,1:]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=50,
                        re_hall=3,
                        gen=3, random_state=1,
                        classification=True,
                        scoring=[metrics.accuracy_score, ], score_pen=[1, ],
                        store=True,
                        n_jobs=4
                        )

    sl.fit(x, y, c=c,
           #           x_group=[[1, 3], [0, 2], [4, 7]]
           )
    score = sl.score(x, y, "r2")
    top_n = sl.loop.top_n(10)
    print(sl.expr)

    top_n.to_csv(r"./re.csv")