import time

from numpy import random
from sklearn.datasets import load_boston

from bgp.base import SymbolSet, SymbolTree, CalculatePrecisionSet
from bgp.calculation.scores import calculate_y
from bgp.calculation.translate import compile_context
from bgp.functions.dimfunc import dless, Dim
from bgp.functions.npfunc import np_map

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
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[[1, 2], [3, 4, 5], [6, 7]],
                       feature_name=["Ss%i" % i for i in range(13)])
    pset0.add_constants(c, c_dim=c_dim, c_prob=None)
    pset0.add_operations(power_categories=(2, 3, 0.5),
                         categories=("Add", "Mul", "Sub", "Div", "ln"),
                         self_categories=None)

    random.seed(2)
    z = time.time()
    sl = [SymbolTree.genGrow(pset0, 3, 4) for _ in range(500)]
    a = time.time()
    # sli =" MAdd(Sub(Add(Mul(x0, gx1), exp(x10)), Mul(Conv(Add(x0, gx0)), Mul(x6, MAdd(x10)))))"
    # sl =["MAdd(gx1 * x11 * (-x0 + x11) * MAdd(gx1))"]

    # sl = [compile_context(sli, pset0.context, pset0.gro_ter_con) for sli in sl]
    # sl = [simple(sli.args[0], pset0.gro_ter_con) for sli in sl if len(sli.args)>0]
    c = 1
    a = time.time()
    pset0 = CalculatePrecisionSet(pset0, scoring=None, score_pen=(1,), filter_warning=True, cal_dim=False,
                                  dim_type=None,
                                  fuzzy=False, add_coef=False, inter_add=True,
                                  inner_add=False, n_jobs=1, batch_size=20,
                                  tq=True)
    from sys import getsizeof

    T100 = getsizeof(sl)
    psize = getsizeof(pset0)
    # print(T100,psize)
    for i in sl:
        b = time.time()
        i0 = compile_context(i, pset0.context, pset0.gro_ter_con)
        r2 = time.time()
        # pprint(i0,pset0)
        calculate_y(i0, pset0.data_x, pset0.y, pset0.terminals_and_constants_repr, add_coef=True,
                    filter_warning=True, inter_add=True, inner_add=True, np_maps=np_map())
        # i = pset0.calculate_detail(i)
        # j = simple(i.coef_expr, pset0.gro_ter_con)
        # print(i)
        # print(repr(i))
        # print(i0)
        # ppprint(i, pset0, feature_name=True)
        if y[0] is None:
            c = c + 1

        r3 = time.time()
    #     print(r2 - b, "com")
    #     print(r3-r2,"cal")
    # e = time.time()
    # print(e-a)
