# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:03
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import odeint

#########################################
# import sympy
# from sympy import Derivative as D, cos
# from sympy import diff, simplify, collect, Function, sin
#
# x1, x2, x3, y1, y2, y3, l1, l2, l3, m1, m2, m3, dth1, dth2, dth3, ddth1, ddth2, ddth3, t, g, tmp, vk = sympy.symbols(
#     "x1, x2, x3, y1, y2, y3, l1, l2, l3, m1, m2, m3, dth1, dth2, dth3, ddth1, ddth2, ddth3, t, g, tmp,vk")
# th1f, th2f, th3f = Function('th1f'), Function('th2f'), Function('th3f')
# th1, th2, th3 = sympy.symbols("th1, th2, th3")
# sublist = [
#     (D(th1f(t), t, t), ddth1),
#     (D(th1f(t), t), dth1),
#     (D(th2f(t), t, t), ddth2),
#     (D(th2f(t), t), dth2),
#     (D(th3f(t), t, t), ddth3),
#     (D(th3f(t), t), dth3),
#     (th1f(t), th1),
#     (th2f(t), th2),
#     (th3f(t), th3)
# ]
#
# x1 = l1 * sin(th1f(t))
# y1 = -l1 * cos(th1f(t))
# x2 = l1 * sin(th1f(t)) + l2 * sin(th2f(t))
# y2 = -l1 * cos(th1f(t)) - l2 * cos(th2f(t))
# x3 = l1 * sin(th1f(t)) + l2 * sin(th2f(t)) + l3 * sin(th3f(t))
# y3 = -l1 * cos(th1f(t)) - l2 * cos(th2f(t)) - l3 * cos(th3f(t))
#
# vx1 = diff(x1, t)
# vx2 = diff(x2, t)
# vx3 = diff(x3, t)
# vy1 = diff(y1, t)
# vy2 = diff(y2, t)
# vy3 = diff(y3, t)
#
# # 拉格朗日量
# L = m1 / 2 * (vx1 ** 2 + vy1 ** 2) + m2 / 2 * (vx2 ** 2 + vy2 ** 2) + m3 / 2 * (
#         vx3 ** 2 + vy3 ** 2) - m1 * g * y1 - m2 * g * y2 - m3 * g * y3
#
#
# # 拉格朗日方程
# def lagrange_equation(L, v):
#     a = L.subs(D(v(t), t), tmp).diff(tmp).subs(tmp, D(v(t), t))
#     b = L.subs(D(v(t), t), tmp)
#     b = b.subs(v(t), vk)
#     b = b.diff(vk)
#     b = b.subs(vk, v(t))
#     b = b.subs(tmp, D(v(t), t))
#     c = a.diff(t) - b
#     c = c.subs(sublist)
#     c = sympy.trigsimp(simplify(c))
#     c = collect(c, [th1, th2, th3, dth1, dth2, dth3, ddth1, ddth2, ddth3])
#     return c
#
#
# eq1 = lagrange_equation(L, th1f)
# eq2 = lagrange_equation(L, th2f)
# eq3 = lagrange_equation(L, th3f)

# l1*(ddth1*(l1*m1 + l1*m2 + l1*m3) + ddth2*(l2*m2*cos(th1 - th2) + l2*m3*cos(th1 - th2)) + ddth3*l3*m3*cos(th1 - th3) + dth2**2*(l2*m2*sin(th1 - th2) + l2*m3*sin(th1 - th2)) + dth3**2*l3*m3*sin(th1 - th3) + g*m1*sin(th1) + g*m2*sin(th1) + g*m3*sin(th1))
# l2*(ddth1*(l1*m2*cos(th1 - th2) + l1*m3*cos(th1 - th2)) + ddth2*(l2*m2 + l2*m3) + ddth3*l3*m3*cos(th2 - th3) + dth1**2*(-l1*m2*sin(th1 - th2) - l1*m3*sin(th1 - th2)) + dth3**2*l3*m3*sin(th2 - th3) + g*m2*sin(th2) + g*m3*sin(th2))
# l3*m3*(ddth1*l1*cos(th1 - th3) + ddth2*l2*cos(th2 - th3) + ddth3*l3 - dth1**2*l1*sin(th1 - th3) - dth2**2*l2*sin(th2 - th3) + g*sin(th3))
#
# l1*(l1*m1 + l1*m2 + l1*m3), l1*(l2*m2*cos(th1 - th2)+l2*m3*cos(th1 - th2)),l1*l3*m3*cos(th1 - th3),l1*(dth2**2*(l2*m2*sin(th1 - th2) + l2*m3*sin(th1 - th2)) + dth3**2*l3*m3*sin(th1 - th3) + g*m1*sin(th1) + g*m2*sin(th1) + g*m3*sin(th1))
# l2*(l1*m2*cos(th1 - th2) + l1*m3*cos(th1 - th2)),l2*(l2*m2 + l2*m3),l2*l3*m3*cos(th2 - th3), l2*(dth1**2*(-l1*m2*sin(th1 - th2) - l1*m3*sin(th1 - th2)) + dth3**2*l3*m3*sin(th2 - th3) + g*m2*sin(th2) + g*m3*sin(th2))
# l3*m3*l1*cos(th1 - th3),l3*m3*l2*cos(th2 - th3),l3*m3*l3,l3*m3*(- dth1**2*l1*sin(th1 - th3) - dth2**2*l2*sin(th2 - th3) + g*sin(th3))
###################################
from sympy import cos, sin

g = 9.8


class TernaryPendulum(object):
    def __init__(self, m1, m2, m3, l1, l2, l3, th1=0.0, th2=0.0, th3=0.0, dth1=0.0, dth2=0.0, dth3=0.0):
        #######把这个双摆模型定义成一个类，包括四个参数，两个微分方程，但是由于微分方程是
        #######二阶的无法直接求解，把原来二阶的微分方程转化为一阶的状态方程来求解
        """

        Parameters
        ----------
        m1:
            weight of ball 1
        m2:
            weight of ball 2
        m3:
            weight of ball 3
        l1:
            length of bar 1
        l2:
            length of bar 2
        l3:
            length of bar 3
        th1:
            上球角度
        th2:
            中球角度
        th3:
            下球角度
        dth1:
            上球角速度
        dth2:
            中球角速度
        dth2:
            下球角速度
        """
        self.m1, self.m2, self.m3, self.l1, self.l2, self.l3 = m1, m2, m3, l1, l2, l3
        self.init_status = np.array([th1, th2, th3, dth1, dth2, dth3])

    def equations(self, init_status, t):
        """

        Parameters
        ----------
        init_status:边界条件
        t：时间

        Returns
        -------

        """
        m1, m2, m3, l1, l2, l3 = self.m1, self.m2, self.m3, self.l1, self.l2, self.l3
        th1, th2, th3, dth1, dth2, dth3 = init_status

        a1, a2, a3, a4 = l1 * (l1 * m1 + l1 * m2 + l1 * m3), l1 * (
                l2 * m2 * cos(th1 - th2) + l2 * m3 * cos(th1 - th2)), l1 * l3 * m3 * cos(th1 - th3), l1 * (
                                 dth2 ** 2 * (l2 * m2 * sin(th1 - th2) + l2 * m3 * sin(
                             th1 - th2)) + dth3 ** 2 * l3 * m3 * sin(th1 - th3) + g * m1 * sin(th1) + g * m2 * sin(
                             th1) + g * m3 * sin(th1))
        a5, a6, a7, a8 = l2 * (l1 * m2 * cos(th1 - th2) + l1 * m3 * cos(th1 - th2)), l2 * (
                l2 * m2 + l2 * m3), l2 * l3 * m3 * cos(th2 - th3), l2 * (dth1 ** 2 * (
                -l1 * m2 * sin(th1 - th2) - l1 * m3 * sin(th1 - th2)) + dth3 ** 2 * l3 * m3 * sin(
            th2 - th3) + g * m2 * sin(th2) + g * m3 * sin(th2))
        a9, a10, a11, a12 = l3 * m3 * l1 * cos(th1 - th3), l3 * m3 * l2 * cos(th2 - th3), l3 * m3 * l3, l3 * m3 * (
                - dth1 ** 2 * l1 * sin(th1 - th3) - dth2 ** 2 * l2 * sin(th2 - th3) + g * sin(th3))

        #######这里注意一下，因为我们的方程里面含有两个二次导数项，这里是为了把每个方程变为只含有一个二次倒数项的形式
        ddth1, ddth2, ddth3 = np.linalg.solve(
            [[float(a1), float(a2), float(a3)], [float(a5), float(a6), float(a7)], [float(a9), float(a10), float(a11)]],
            [-float(a4), -float(a8), -float(a12)])

        return np.array([dth1, dth2, dth3, ddth1, ddth2, ddth3])

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
        th1_array, th2_array, th3_array, dth1_array, dth2_array, dth3_array = track.T
        l1, l2, l3, = self.l1, self.l2, self.l3
        x1 = l1 * np.sin(th1_array)
        y1 = -l1 * np.cos(th1_array)
        x2 = x1 + l2 * np.sin(th2_array)
        y2 = y1 - l2 * np.cos(th2_array)
        x3 = x2 + l3 * np.sin(th3_array)
        y3 = y2 - l3 * np.cos(th3_array)
        self.init_status = track[-1, :].copy()  # 将最后的状态赋给pendulum
        return x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array, dth1_array, dth2_array, dth3_array

    def odeint_x(self, th1_array, th2_array, th3_array):

        l1, l2, l3, = self.l1, self.l2, self.l3
        x1 = l1 * np.sin(th1_array)
        y1 = -l1 * np.cos(th1_array)
        x2 = x1 + l2 * np.sin(th2_array)
        y2 = y1 - l2 * np.cos(th2_array)
        x3 = x2 + l3 * np.sin(th3_array)
        y3 = y2 - l3 * np.cos(th3_array)
        return x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array


if __name__ == "__main__":
    pendulum = TernaryPendulum(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0)
    #######初始角度theta1=1 theta2=2 这里的单位为弧度
    ####定义初始条件，这里认为初始两个球的角速度为零，时间60秒，步长0.02
    x1, y1, x2, y2, x3, y3, th1_array, th2_array, th3_array, dth1_array, dth2_array, dth3_array = pendulum.odeint(
        0.0, 60.0, 0.2)
