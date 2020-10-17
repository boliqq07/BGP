# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:59
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from Instances.pendulum.double_pendulum import DoublePendulum
from Instances.pendulum.draw import plot_point
from Instances.pendulum.ternary_pendulum import TernaryPendulum

if __name__ == "__main__":
    import numpy as np

    tpen = TernaryPendulum(1.0, 2.0,1.0, 1.0, 2.0,3.0, 0, 0, 2, 1, 2, 1)
    ########初始角度theta1=1 theta2=2 这里的单位为弧度
    #####定义初始条件，这里认为初始两个球的角速度为零，时间60秒，步长0.02
    x1, y1,x2, y2,x3,y3, th1_array, th2_array,th3_array, v1_array, v2_array, v3_array = tpen.odeint(0, 20, 0.05)
    # plot_curve(x1, i=300, save=True)
    data = np.vstack((x1, y1, x2, y2, x3, y3)).T
    plot_point(data)
