# -*- coding: utf-8 -*-

# @Time    : 2020/10/5 16:20
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from sklearn.utils import shuffle

from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    # data
    from sklearn.datasets import load_iris

    data = load_iris()
    x = data["data"][:98, (0, 1, 2, 3)]
    x[40:60] = shuffle(x[40:60], random_state=2)
    y = data["target"][:98]
    c = None

    sl = SymbolLearning(loop=None, pop=100, gen=10, cal_dim=False, re_hall=2, add_coef=True, cv=1, random_state=2,
                        classification=True,
                        re_Tree=1, details=False,
                        store=r"/data/home/wangchangxin"
                        )
    sl.fit(x, y, c=c)
    sl.fit(x, y, c=c, warm_start=True)

    print(sl.expr)
    y = sl.predict(x)
