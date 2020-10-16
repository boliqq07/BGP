# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:03
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
# -*- coding: utf-8 -*-


from math import sin, cos

import numpy as np
from scipy.integrate import odeint
import matplotlib.animation as animation
import matplotlib.pyplot as pl


g = 9.8


#######把这个双摆模型定义成一个类，包括四个参数，两个微分方程，但是由于微分方程是
#######二阶的无法直接求解，把原来二阶的微分方程转化为一阶的状态方程来求解
class DoublePendulum(object):
    def __init__(self, m1, m2, l1, l2):
        """

        Parameters
        ----------
        m1::
            weight of ball 1
        m2:
            weight of ball 2
        l1:
            length of bar 1
        l2:
            length of bar 2
        """
        self.m1, self.m2, self.l1, self.l2 = m1, m2, l1, l2

        # init_status
        # th1: 上球角度
        # th2: 下球角度
        # v1: 上球角速度
        # v2: 下球角速度
        self.init_status = np.array([0.0, 0.0, 0.0, 0.0])

    def equations(self, w, t):
        """
        微分方程公式
        """
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        th1, th2, v1, v2 = w
        dth1 = v1
        dth2 = v2

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


######这里是调用odeint函数求解双摆的动力学方程
def double_pendulum_odeint(pendulum, ts, te, tstep):
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
    track = odeint(pendulum.equations, pendulum.init_status, t)
    th1_array, th2_array = track[:, 0], track[:, 1]
    v1_array, v2_array = track[:, 2], track[:, 3]
    l1, l2 = pendulum.l1, pendulum.l2
    x1 = l1 * np.sin(th1_array)
    y1 = -l1 * np.cos(th1_array)
    x2 = x1 + l2 * np.sin(th2_array)
    y2 = y1 - l2 * np.cos(th2_array)
    pendulum.init_status = track[-1, :].copy()  # 将最后的状态赋给pendulum
    return x1, y1, x2, y2, th1_array, th2_array, v1_array, v2_array


if __name__ == "__main__":
    pendulum = DoublePendulum(1.0, 2.0, 1.0, 2.0)
    ########初始角度theta1=1 theta2=2 这里的单位为弧度
    th1, th2 = 1.0, 2.0
    #####定义初始条件，这里认为初始两个球的角速度为零，时间60秒，步长0.02
    dt = 0.02
    pendulum.init_status[:2] = th1, th2
    x1, y1, x2, y2, th1_array, th2_array, v1_array, v2_array = double_pendulum_odeint(pendulum, 0, 60, dt)

    fig = pl.figure(figsize=(5, 20), dpi=20)

    ##########下面将我们的求解结果进行可视化
    x2=x2[::10]
    fig = pl.figure()
    ax = fig.add_subplot(111, xlim=(0, 300), ylim=(-5, 5))
    # ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], '-', lw=1)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    ##########下面是函数animation.FuncAnimation使用模板
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    # def animate(i):
    #     thisx = [0, x1[i], x2[i]]
    #     thisy = [0, y1[i], y2[i]]
    #
    #     line.set_data(thisx, thisy)
    #     time_text.set_text(time_template % (i * dt))
    #     return line, time_text

    this=[[0], [x2[0]]]

    def animate(i):

        this[0].append(i)
        this[1].append(x2[i])

        line.set_data(*this)
        time_text.set_text(time_template % (i * dt))
        return line, time_text


    ani = animation.FuncAnimation(fig, animate, range(1, 300),
                                  interval=dt * 1000, blit=True, init_func=init)
    ani.save('shuangbai2.gif', writer='pillow')
    pl.show()
