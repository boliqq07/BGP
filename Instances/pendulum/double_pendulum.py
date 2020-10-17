# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:03
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-


from math import sin, cos

import numpy as np
from scipy.integrate import odeint

g = 9.8


class DoublePendulum(object):
    def __init__(self, m1, m2, l1, l2, th1=0.0, th2=0.0, dth1=0.0, dth2=0.0):
        #######把这个双摆模型定义成一个类，包括四个参数，两个微分方程，但是由于微分方程是
        #######二阶的无法直接求解，把原来二阶的微分方程转化为一阶的状态方程来求解
        """

        Parameters
        ----------
        m1:
            weight of ball 1
        m2:
            weight of ball 2
        l1:
            length of bar 1
        l2:
            length of bar 2
        th1:
            上球角度
        th2:
            下球角度
        dth1:
            上球角速度
        dth2:
            下球角速度
        """
        self.m1, self.m2, self.l1, self.l2 = m1, m2, l1, l2
        # self.th1, self.th2, self.dth1, self.dth2= th1, th2, dth1, dth2
        self.init_status = np.array([th1, th2, dth1, dth2])

    def equations(self, init_status, t):
        """

        Parameters
        ----------
        init_status:边界条件
        t：时间

        Returns
        -------

        """
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        th1, th2, dth1, dth2 = init_status

        # eq of th1
        a = l1 * (m1 + m2)  # dv1 parameter
        b = m2 * l2 * cos(th1 - th2)  # dv2 paramter
        c = (m2 * l2 * sin(th1 - th2) * dth2 * dth2 + (m1 + m2) * g * sin(th1))

        # eq of th2
        d = l1 * cos(th1 - th2)  # dv1 parameter
        e = l2  # dv2 parameter
        f = (-l1 * sin(th1 - th2) * dth1 * dth1 + g * sin(th2))

        #######这里注意一下，因为我们的方程里面含有两个二次导数项，这里是为了把
        #######每个方程变为只含有一个二次倒数项的形式
        dv1, dv2 = np.linalg.solve([[a, b], [d, e]], [-c, -f])

        return np.array([dth1, dth2, dv1, dv2])

    def odeint(self, ts, te, tstep):
        ######这里是调用odeint函数求解双摆的动力学方程
        """
        对双摆系统的微分方程组进行数值求解，返回两个小球的X-Y坐标,角度，角速度
        Parameters
        ----------
        pendulumP:函数
        ts:开始时间
        te:结束时间
        tstep:时间步长

        Returns
        -------
        坐标
        """

        t = np.arange(ts, te, tstep)
        track = odeint(self.equations, self.init_status, t)
        th1_array, th2_array = track[:, 0], track[:, 1]
        dth1_array, dth2_array = track[:, 2], track[:, 3]
        l1, l2 = self.l1, self.l2
        x1 = l1 * np.sin(th1_array)
        y1 = -l1 * np.cos(th1_array)
        x2 = x1 + l2 * np.sin(th2_array)
        y2 = y1 - l2 * np.cos(th2_array)
        self.init_status = track[-1, :].copy()  # 将最后的状态赋给pendulum
        return x1, y1, x2, y2, th1_array, th2_array, dth1_array, dth2_array


if __name__ == "__main__":
    pendulum = DoublePendulum(1.0, 2.0, 1.0, 2.0, 0, 0, 1, 2)
    ########初始角度theta1=1 theta2=2 这里的单位为弧度
    #####定义初始条件，这里认为初始两个球的角速度为零，时间60秒，步长0.02
    x1, y1, x2, y2, th1_array, th2_array, v1_array, v2_array = pendulum.odeint(0, 60, 0.2)
