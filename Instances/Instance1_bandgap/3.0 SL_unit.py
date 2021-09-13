from mgetool.exports import Store
from mgetool.imports import Call
from sklearn.utils import shuffle

from bgp.functions.dimfunc import Dim, dless
from bgp.preprocess import MagnitudeTransformer

if __name__ == "__main__":
    import os

    os.chdir(r'band_gap')
    data = Call()
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr

    store = Store()

    data_import = all_import
    data225_import = data_import

    select = ['cell volume', 'electron density',
              'lattice constants a', 'lattice constants c', 'covalent radii', 'ionic radii(shannon)',
              'core electron distance(schubert)',
              'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', "electron number", 'valence electron number',
              'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']

    from sympy.physics.units import eV, pm, nm

    select_unit = [100 ** 3 * pm ** 3, 100 ** -3 * pm ** -3, 100 * pm, 100 * pm, 100 * pm, 100 * pm, 100 * pm, eV, eV,
                   eV, dless, dless, dless, 100 ** -1 * pm ** -1, 10 ** -2 * nm ** 3]

    fea_name = ['V_c', 'rho_e'] + [name_and_abbr[j][1] + "_%i" % i for j in select[2:] for i in range(2)]
    select = ['cell volume', 'electron density'] + [j + "_%i" % i for j in select[2:] for i in range(2)]
    x_u = [100 ** 3 * pm ** 3, 100 ** -3 * pm ** -3] + [j for j in select_unit[2:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']
    X = X_frame.values
    Y = y_frame.values
    X, Y = shuffle(X, Y, random_state=5)
    x, y = X, Y

    # y_unit
    from sympy.physics.units import eV, elementary_charge, m, pm

    y_u = eV
    # c_unit
    c = [1, 5.290 * 10 ** -11, 1.74, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4]
    c_u = [elementary_charge, m, dless, dless, dless, dless, dless, dless, dless]

    """preprocessing"""
    dims = [Dim.convert_to_Dim(i, target_units=None, unit_system="SI") for i in x_u]

    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)

    scal = MagnitudeTransformer(tolerate=1)

    group = 2
    n = X.shape[1]
    indexes = [_ for _ in range(n)]
    group = [indexes[i:i + group] for i in range(2, len(indexes), group)]
    x, y = scal.fit_transform_all(x, y, group=group)
    c = scal.fit_transform_constant(c)
    store.to_pkl_pd(scal, "si_transformer")
    store.to_pkl_pd((x, x_dim, y, y_dim, c, c_dim, X, Y), "SL_data")
