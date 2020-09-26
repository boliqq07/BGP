import time

from sklearn import preprocessing
from sklearn.datasets import load_boston

from bgp.base import SymbolSet, SymbolTree, CalculatePrecisionSet
from bgp.calculation.coefficient import try_add_coef
from bgp.calculation.translate import general_expr, compile_context
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

    sc = preprocessing.MinMaxScaler((1, 2))
    x = sc.fit_transform(x)
    t = time.time()

    # symbolset
    pset0 = SymbolSet()
    pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=2,
                       feature_name=["Ss%i" % i for i in range(13)])
    pset0.add_constants(c, c_dim=c_dim, c_prob=None)
    pset0.add_operations(power_categories=(2, 3, 4),
                         categories=("Add", "Mul", "Sub", "Div", "ln", "exp"),
                         self_categories=None)

    pset0 = CalculatePrecisionSet(pset0, scoring=None, score_pen=(1,), filter_warning=True, cal_dim=False,
                                  dim_type=None,
                                  fuzzy=False, add_coef=False, inter_add=True,
                                  inner_add=False, n_jobs=1, batch_size=20,
                                  tq=True)
    sl = [SymbolTree.genGrow(pset0, 3, 4) for _ in range(50)]
    sll = [compile_context(o, pset0.context, pset0.gro_ter_con, simplify=True) for o in sl]

    slll = [try_add_coef(i, pset0.data_x, pset0.y, pset0.terminals_and_constants_repr,
                         filter_warning=True, inter_add=True, inner_add=False, vector_add=True, np_maps=np_map())[1]
            for i in sll]
    sllll = [general_expr(o, pset0, simplifying=False) for o in slll]
    # tree="MSub(exp(gx5))"
    # sll = compile_context(tree, pset0.context, pset0.gro_ter_con, simplify=True)
    # slll = try_add_coef(sll, pset0.data_x, pset0.y, pset0.terminals_and_constants_repr,
    #               filter_warning=True, inter_add=False, inner_add=False, vector_add=True, np_maps=np_map())[1]
    # sllll=general_expr(slll, pset0, simplifying=False)
