# -*- coding: utf-8 -*-

# @Time    : 2020/8/22 13:38
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
import sympy
from scipy.optimize import least_squares

k1, k_1, k2, k3 = sympy.symbols(["k1", "k_1", "k2", "k3"])
k1p, k_1p, k2p = sympy.symbols(["k1p", "k_1p", "k2p"])
Thetah, taup = sympy.symbols(["thetah", "taup"])
E, Rct, Rp, Cp, R0 = sympy.symbols(['E', "Rct", "Rp", "Cp", "R0"])
F, T, R, q, beta = sympy.symbols(['F', "T", "R", "q", "beta"])

"""calculate"""
result = sympy.solve([
    # Rct ** (-1) - beta * F ** 2 / (R * T)*(k1p * (1 - Thetah) - k_1p * Thetah + k2p * Thetah),
    # taup ** (-1) - F / q * (4 * k3 * Thetah + k1p + k_1p + k2p),
    Thetah - ((k1p + k_1p + k2p) + sympy.sqrt((k1p + k_1p + k2p) ** 2) + 8 * k1p * k3),
    k1p - k1 * sympy.exp(-beta * F * E / (R * T)),
    k_1p - k_1 * sympy.exp((1 - beta) * F * E / (R * T)),
    k2p - k2 * sympy.exp(-beta * F * E / (R * T)),
],
    [Thetah, k1p, k_1p, k2p])

print(result)

from mgetool.exports import Store

store = Store(r'C:\Users\Administrator\Desktop\cl')
store.to_pkl_pd(result, "result")

"""fitting"""
exps1 = (beta * F ** 2 / (R * T) * (k1p * (1 - Thetah) - k_1p * Thetah + k2p * Thetah)) ** (-1)
exps2 = (F / q * (4 * k3 * Thetah + k1p + k_1p + k2p)) ** (-1)
exps3 = (beta * F ** 2 / (R * T) * (k2p - k1p - k_1p) * (k1p * (1 - Thetah) - k_1p * Thetah + k2p * Thetah) / (
        4 * k3 * Thetah + k2p + k1p + k_1p)) ** (-1)

subbb1 = {
    Thetah: result[0][0],
}
subbb2 = {
    k1p: result[0][1],
    k_1p: result[0][2],
    k2p: result[0][3],
}

exps1 = exps1.subs(subbb1)
exps1 = exps1.subs(subbb2)
# exps1 = sympy.simplify(exps1)

exps2 = exps2.subs(subbb1)
exps2 = exps2.subs(subbb2)
# exps2 = sympy.simplify(exps2)

exps3 = exps3.subs(subbb1)
exps3 = exps3.subs(subbb2)
# exps3 = sympy.simplify(exps3)

from mgetool.imports import Call

data = Call(r'C:\Users\Administrator\Desktop\cl', index_col=None)
values_data = data.xlsx().values_data
E_values = values_data["E"].values
Rct_values = values_data["Rct"].values
Rp_values = values_data["Rp"]
taup_values = (values_data["Rp"] * values_data["Cp"]).values
R0_values = -(Rct_values ** 2 + Rct_values * Rp_values) / Rp_values
F_values = 96485
T_values = 298
R_values = 8.314
q_values = 8 * 10e-5

beta_values = 0.5

std_Rct = np.std(Rct_values)
std_taup = np.std(taup_values)
std_R0 = np.std(R0_values)

func_list = []
for resi in (exps1, exps2, exps3):
    func0 = sympy.utilities.lambdify([E, R0, Rct, taup, F, T, R, beta, q,
                                      k1, k_1, k2, k3], resi)
    func_list.append(func0)


def func(funci, x, p):
    """"""
    num_list = []
    num_list.extend(x)
    num_list.extend(p)
    return funci(*num_list)


def minin(p, x_):
    """"""
    vv = (func(func_list[0], x_, p) - Rct_values) ** 2 / (std_Rct ** 2) + \
         (func(func_list[1], x_, p) - taup_values) ** 2 / (std_taup ** 2) + \
         (func(func_list[2], x_, p) - R0_values) ** 2 / (std_R0 ** 2)
    return vv


result = least_squares(minin, x0=[1] * 4,

                       method="trf",
                       args=(
                           ((E_values, R0_values, Rct_values, taup_values, F_values, T_values, R_values, beta_values,
                             q_values),)
                       ), verbose=True,
                       max_nfev=10000)

cof = result.x
print(result.cost)
