# -*- coding: utf-8 -*-

# @Time    : 2021/7/19 11:52
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pandas as pd
from sklearn import metrics
from bgp.skflow import SymbolLearning
from bgp.base import SymbolSet
from bgp.functions.dimfunc import dless
from sklearn.datasets import load_boston
from bgp.skflow import SymbolLearning

if __name__ == "__main__":

    # 数据
    data = pd.read_csv("204-6.csv")
    data_np = data.values
    x = data_np[:,1:]
    y = data_np[:,0]


    # 量纲
    from sympy.physics.units import kg, m, pa, J, mol, K
    from bgp.functions.dimfunc import Dim, dless

    # 由单位获得缩放因子和量纲
    gpa_dim= Dim.convert_to_Dim(1e9*pa, unit_system="SI")
    j_d_mol_dim = Dim.convert_to_Dim(1000*J/mol, unit_system="SI")
    K_dim= Dim.convert_to_Dim(K, unit_system="SI")
    kg_d_m3_dim = Dim.convert_to_Dim(kg/m**3, unit_system="SI")

    # 忽视缩放因子
    y_dim = dless
    x_dim = [dless,gpa_dim[1],j_d_mol_dim[1],K_dim[1],dless,kg_d_m3_dim[1]]

    # 符号集合
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=( "Mul", "Div", "exp"),
                    )

    # 符号回归

    # 方式选择1，系数加在最外层
    # sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=2, random_state=1,pset=pset0,
    #                     classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
    #                     cal_dim=True, n_jobs = 10,
    #                     store =True
    #                     )

    # # 方式选择2，系数加在最外层,认定系数可以自动补全量纲
    # pset0.y_dim=None
    # sl = SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=3, random_state=1,pset=pset0,
    #                     classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
    #                     cal_dim=True, n_jobs = 10, store=True,
    #
    #                     )


    # # 方式选择3，系数加在公式内层,认定系数可以自动补全量纲
    pset0.y_dim = None
    sl = SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=20, random_state=1,pset=pset0,
                        classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
                        cal_dim=True, inner_add=True, out_add=False,
                        n_jobs = 10,)
    #
    # 方式选择4，系数加在公式内层,不考虑量纲计算,
    # sl = SymbolLearning(loop="MultiMutateLoop", pop=200, gen=10, random_state=1,pset=pset0,
    #                     classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
    #                     cal_dim=False,
    #                     inner_add=False, out_add=False,
    #                     n_jobs = 2,)
    # #
    # # 方式选择5，,公式项全部展开，每一项前面加系数（非常复杂),不考虑量纲计算,
    # pset0.y_dim = None
    # sl = SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=20, random_state=1,pset=pset0,
    #                     classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
    #                     cal_dim=False, flat_add=True, out_add=False,
    #                     n_jobs = 10,)
    #
    # # 方式选择6，,公式项全部展开，每一项前面加系数（非常复杂),不考虑量纲计算, 公式复杂度不限制
    # pset0.y_dim = None
    # sl = SymbolLearning(loop="MultiMutateLoop", pop=1000, gen=10, random_state=1, max_value=7,pset=pset0,
    #                     classification=True, scoring=[metrics.accuracy_score, ], score_pen=[1, ],
    #                     cal_dim=False, flat_add=True, out_add=False,
    #                     n_jobs = 10,)
    sl.fit(x, y)
    pre_y = sl.predict(x)
    print(sl.expr)





