import time

from numpy import random
from sklearn.datasets import load_boston

from bgp.base import SymbolSet, SymbolTree
from bgp.calculation.translate import compile_context
from bgp.functions.dimfunc import dless, Dim

if __name__ == "__main__":
    # data
    data = load_boston()
    x = data["data"]
    y = data["target"]
    c = [6, 3, 4]
    # unit
    from sympy.physics.units import kg

    x_u = [kg] * 13
    y_u = kg
    c_u = [dless, dless, dless]

    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    t = time.time()

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[[1, 2], [3, 4, 5]])
    pset0.add_constants(c, c_dim=c_dim, c_prob=None)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "exp"),
                         self_categories=None)

    random.seed(0)
    z = time.time()
    sl = [SymbolTree.genGrow(pset0, 3, 4) for _ in range(100)]
    a = time.time()
    sl = [compile_context(sli, pset0.context, pset0.gro_ter_con) for sli in sl]
    b = time.time()

    print(b - a, a - z)
