from mgetool.exports import Store
from mgetool.imports import Call
from sklearn.utils import shuffle

from bgp.base import SymbolSet
from bgp.calculation.translate import group_str
from bgp.flow import MultiMutateLoop
from bgp.functions.dimfunc import Dim, dless
from bgp.preprocess import MagnitudeTransformer

if __name__ == "__main__":
    import os

    os.chdir(r'../../Instances/Instance1_bandgap/band_gap')

    data = Call()
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr

    store = Store()

    data_import = all_import
    data225_import = data_import

    cal = []

    from sympy.physics.units import eV

    select = ['electronegativity(martynov&batsanov)', 'fusion enthalpy', 'valence electron number']
    select_unit = [dless, eV, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['covalent radii', 'electronegativity(martynov&batsanov)', 'valence electron number']
    select_unit = [pm, dless, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['electronegativity(martynov&batsanov)', 'ionic radii(shannon)', 'valence electron number']
    select_unit = [dless, pm, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['covalent radii', 'valence electron number']
    select_unit = [pm, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['covalent radii', 'fusion enthalpy', 'valence electron number']
    select_unit = [pm, eV, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['electronegativity(martynov&batsanov)', 'valence electron number']
    select_unit = [dless, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['covalent radii', 'ionic radii(shannon)', 'valence electron number']
    select_unit = [pm, pm, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['core electron distance(schubert)', 'covalent radii', 'valence electron number']
    select_unit = [pm, pm, dless]
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['core electron distance(schubert)', 'electronegativity(martynov&batsanov)', 'valence electron number']
    select_unit = [pm, dless, dless]  ###
    cal.append((select, select_unit))

    from sympy.physics.units import eV, pm

    select = ['cohesive energy(Brewer)', 'covalent radii', 'valence electron number']
    select_unit = [eV, pm, dless]
    cal.append((select, select_unit))

    for select, select_unit in cal[:]:
        fea_name = [name_and_abbr[j][1] + "_%i" % i for j in select for i in range(2)]
        select = [j + "_%i" % i for j in select for i in range(2)]
        x_u = [j for j in select_unit for i in range(2)]

        X_frame = data225_import[select]
        y_frame = data225_import['exp_gap']
        X = X_frame.values
        y = y_frame.values
        x, y = shuffle(X, y, random_state=5)

        # y_unit
        from sympy.physics.units import eV, elementary_charge, m

        y_u = eV
        # c_unit
        c = [1, 5.290 * 10 ** -11, 1.74, 2, 3, 4, 0.5, 1 / 3, 1 / 4]
        c_u = [elementary_charge, m, dless, dless, dless, dless, dless, dless, dless]

        """preprocessing"""
        x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
        y, y_dim = Dim.convert_xi(y, y_u)
        c, c_dim = Dim.convert_x(c, c_u)

        scal = MagnitudeTransformer(tolerate=1)
        x, y = scal.fit_transform_all(x, y, group=2)
        c = scal.fit_transform_constant(c)

        # symbolset
        pset0 = SymbolSet()
        pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=2, feature_name=fea_name)
        pset0.add_constants(c, c_dim=c_dim, c_prob=0.05)
        pset0.add_operations(power_categories=(2, 3, 0.5),
                             categories=("Add", "Mul", "Sub", "Div", "exp", "ln"),
                             self_categories=None)

        # a = time.time()
        dicts = {}
        for i in range(12):
            bl = MultiMutateLoop(pset=pset0, gen=20, pop=1000, hall=1, batch_size=40, re_hall=3,
                                 n_jobs=12, mate_prob=0.9, max_value=5,
                                 mutate_prob=0.8, tq=False, dim_type="coef",
                                 re_Tree=0, store=False, random_state=12, verbose=True,
                                 stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},
                                 add_coef=True, inner_add=False, cal_dim=True, vector_add=True, personal_map=False)
            # b = time.time()
            exps = bl.run()
            print([i.coef_expr for i in exps])
            score = exps.keys[0].values[0]
            name = group_str(exps[0], pset0, feature_name=True)
            dicts["s%s" % i] = [score, name]
            print(i)

        store.to_csv(dicts, model="a+")
