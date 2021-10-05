# -*- coding: utf-8 -*-

# @Time    : 2021/7/26 10:36
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

# pip install BindingGP

import pandas as pd
import numpy as np
from mgetool.tool import tt
from sklearn.linear_model import LinearRegression

from bgp.calculation.translate import compile_context

from bgp.skflow import SymbolLearning
from sklearn.metrics import r2_score, mean_absolute_error

dataTi = pd.read_csv("Ti.csv", index_col=0)

datanp = dataTi.values

x = datanp[:, 1:]
y = datanp[:, 0]

ind = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

index = np.where(np.array(ind) == 1)[0]
x = x[:, index]

sl = SymbolLearning(loop='MultiMutateLoop',pop=1000, gen=10, random_state=1,n_jobs=1,
                    add_coef=True,initial_max=2,initial_min=1,
                    max_value=2, store=False,)
tt.t
sl.fit(x,y)
tt.t
tt.p
print(sl.expr)
print(sl.fitness)

# y_pre = sl.predict(x)

x0=x[:,0]
x1=x[:,1]
y_pre2= x1/x0**3
s1 = r2_score(y,y_pre2)
lr = LinearRegression()
lr.fit(y_pre2.reshape(-1,1),y)
s2 = lr.score(y_pre2.reshape(-1,1),y)
e1 = mean_absolute_error(y,y_pre2)

