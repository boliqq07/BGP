# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:59
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from mgetool.show import BasePlot

from Instances.double_pendulum.double_pendulum import DoublePendulum, double_pendulum_odeint
from bgp.base import SymbolSet

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from bgp.skflow import SymbolLearning
    from sklearn import metrics
    import numpy as np

    pendulum = DoublePendulum(1.0, 2.0, 1.0, 2.0)
    ########初始角度theta1=1 theta2=2 这里的单位为弧度
    th1, th2 = 1.0, 2.0
    #####定义初始条件，这里认为初始两个球的角速度为零，时间60秒，步长0.02
    dt = 0.02
    pendulum.init_status[:2] = th1, th2
    x1, y1, x2, y2, th1_array, th2_array, v1_array, v2_array = double_pendulum_odeint(pendulum, 0, 60, dt)

    x = np.vstack((th1_array, th2_array)).T
    y = x2

    # def func(ind):
    #     c = ind.fitness.values[0] <= 0.001
    #     return c

    # sl = SymbolLearning(loop="MutilMutateLoop", pop=500, gen=20, random_state=1, max_value=4,
    #                     scoring=[metrics.mean_absolute_error, ],
    #                     score_pen=[-1, ],
    #                     stats={"fitness_dim_min": ("min",), "dim_is_target": ("sum",)},
    #                     add_coef=True,
    #                     re_hall = 3,
    #                     stop_condition=func,
    #                     vector_add=False, inter_add=False,out_add=True, cal_dim=False, store=False
    #                     )
    # #
    # pset0 = SymbolSet()
    # pset0.add_features(x, y)
    # pset0.add_operations(power_categories=None,
    #                      categories=("Add", "Mul", "Sub", "cos", "sin"),
    #                      )
    #
    # sl.fit(pset=pset0)
    # print(sl.expr)
