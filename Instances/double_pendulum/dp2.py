# -*- coding: utf-8 -*-

# @Time    : 2020/10/13 20:59
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from mgetool.show import BasePlot

from Instances.double_pendulum.double_pendulum import DoublePendulum, double_pendulum_odeint
from bgp.base import SymbolSet

if __name__ == "__main__":
    import sympy
    import numpy as np
    t = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    a = np.array([0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0])
    b = np.array([0,1,2,3,3,3,6,7,7,8,8,7,7,6,5,4,4,6,7,8])
    A, B,T = sympy.symbols("A B T")

    def Derivative(a, b):
        delta_a = np.gradient(a)
        delta_b = np.gradient(b)
        return delta_a/delta_b


    f = A*B**2+A

    dfdxi = sympy.diff(f, A, evaluate=False)

    dxjdxt = sympy.diff(A, T, evaluate=False)
    ddxjdxt = sympy.diff(dxjdxt, T, evaluate=False)