import matplotlib.pyplot as pl
from matplotlib import animation
import numpy as np

def plot_curve(x, i, save=True, step=1):
    dt = 0.2
    fig = pl.figure(figsize=(20, 5), dpi=20)

    ##########下面将我们的求解结果进行可视化
    ax = fig.add_subplot(111, xlim=(0, i), ylim=(min(x), max(x)))
    # ax.set_aspect('equal')
    ax.grid()
    ax.tick_params(labelsize=40)

    line, = ax.plot([], [], '-', lw=1)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.0, 0.9, '', size=40)

    ##########下面是函数animation.FuncAnimation使用模板
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # init
    this = [[0], [x[0]]]

    def animate(i):
        this[0].append(i)
        this[1].append(x[i])

        line.set_data(*this)
        time_text.set_text(time_template % (i * dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(0, i, step),
                                  interval=dt * 1000, blit=True, init_func=init)
    if save:
        ani.save('{}.gif'.format(save), writer='pillow')
    pl.tick_params(labelsize=100)
    pl.show()


def plot_point(x, i=None, save=True, step=1):

    fig = pl.figure(figsize=(5, 5), dpi=20)
    dt = 0.2

    ##########下面将我们的求解结果进行可视化
    aa,bb = np.min((np.min(x),-5)), np.max((np.max(x),5))
    ax = fig.add_subplot(111, xlim=(aa, bb), ylim=(aa, bb))

    ax.grid()
    ax.tick_params(labelsize=40)

    line, = ax.plot([], [], 'b-o', ms=5)
    line2, = ax.plot([], [], 'r-', lw=1, color="g")
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', size=20)

    ##########下面是函数animation.FuncAnimation使用模板
    def init():
        line.set_data([], [])
        time_text.set_text('')

        line2.set_data([], [])

        return line, line2, time_text
        # return line2, time_text

    # init
    this = [x[0, ::2], x[0, 1::2]]
    his = [x[0, -2], x[0, -1]]
    ze = np.zeros_like(x[:, :2])
    x = np.concatenate((ze, x), axis=1)

    def animate(i):
        this[0] = x[i, ::2]
        this[1] = x[i, 1::2]
        ss = 30
        if i > ss:
            j = i - ss
        else:
            j = 0

        his[0] = x[j:i, -2]
        his[1] = x[j:i, -1]

        line.set_data(*this)
        line2.set_data(*his)
        time_text.set_text(time_template % (i * dt))
        return line, line2, time_text
        # return line2, time_text

    if i is None:
        i = x.shape[0]
    ani = animation.FuncAnimation(fig, animate, range(1, i, step),
                                  interval=dt * 1000, blit=True, init_func=init)
    if save:
        ani.save('{}.gif'.format(save), writer='pillow')
    pl.show()
