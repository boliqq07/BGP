# -*- coding: utf-8 -*-

# @Time    : 2020/8/22 13:38
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import sympy

#
from scipy.optimize import least_squares

k1, k_1, k2, k3 = sympy.symbols(["k1", "k_1", "k2", "k3"])
k1p, k_1p, k2p = sympy.symbols(["k1p", "k_1p", "k2p"])
Thetah, taup = sympy.symbols(["thetah", "taup"])
eta, Rct, Rp, Cp = sympy.symbols(['eta', "Rct", "Rp", "Cp"])
F, T, R, q, beta = sympy.symbols(['F', "T", "R", "q", "beta"])
#
# """calculate"""
# result = sympy.solve([
#     Rct ** (-1) - beta * F ** 2 / (R * T)*(k1p * (1 - Thetah) - k_1p * Thetah + k2p * Thetah),
#     taup ** (-1) - F / q * (4 * k3 * Thetah + k1p + k_1p + k2p),
#     Thetah - ((k1p + k_1p + k2p) + sympy.sqrt((k1p + k_1p + k2p) ** 2) + 8 * k1p * k3),
#     k1p - k1 * sympy.exp(-beta * F * eta / (R * T)),
#     k_1p - k_1 * sympy.exp((1 - beta) * F * eta / (R * T)),
#     k2p - k2 * sympy.exp(-beta * F * eta / (R * T)),
# ],
#     [Thetah, k1p, k_1p, k2p, k3])
#
# print(result)

# from mgetool.exports import Store
# store = Store(r'C:\Users\Administrator\Desktop\cl')
# store.to_pkl_pd(result,"result")

"""fitting"""
exps1 = Rct ** (-1) - beta * F ** 2 / (R * T) * (k1p * (1 - Thetah) - k_1p * Thetah + k2p * Thetah)
exps2 = taup ** (-1) - F / q * (4 * k3 * Thetah + k1p + k_1p + k2p)

subbb1 = {
    Thetah: ((k1p + k_1p + k2p) + sympy.sqrt((k1p + k_1p + k2p) ** 2) + 8 * k1p * k3)}
subbb2 = {
    k1p: k1 * sympy.exp(-beta * F * eta / (R * T)),
    k_1p: k_1 * sympy.exp((1 - beta) * F * eta / (R * T)),
    k2p: k2 * sympy.exp(-beta * F * eta / (R * T)),
}

exps1 = exps1.subs(subbb1)
exps1 = exps1.subs(subbb2)
exps1 = sympy.simplify(exps1)

exps2 = exps2.subs(subbb1)
exps2 = exps2.subs(subbb2)
exps2 = sympy.simplify(exps2)

from mgetool.imports import Call

data = Call(r'C:\Users\Administrator\Desktop\cl', index_col=None)
values_data = data.xlsx().values_data
eta_values = values_data["eta"].values
Rct_values = values_data["Rct"].values
taup_values = (values_data["Rp"] * values_data["Cp"]).values
F_values = 96485
T_values = 298
R_values = 8.314
q_values = 8 * 10 ** (-5)
beta_values = 0.5

func_list = []
for resi in (exps1, exps2):
    func0 = sympy.utilities.lambdify([eta, Rct, taup, F, T, R, q, beta, k1, k_1, k2, k3], resi)
    func_list.append(func0)


def func(funci, x, p):
    """"""
    num_list = []
    num_list.extend(x)
    num_list.extend(p)
    return funci(*num_list)


def minin(p, x_):
    """"""
    return func(func_list[0], x_, p) + func(func_list[1], x_, p)


result = least_squares(minin, x0=[1] * 4,
                       args=(
                           ((eta_values, Rct_values, taup_values, F_values, T_values, R_values, q_values,
                             beta_values),)
                       ),
                       loss='linear', ftol=1e-8, jac='2-point', method='dogbox', max_nfev=10000)

cof = result.x
print(result.cost)
