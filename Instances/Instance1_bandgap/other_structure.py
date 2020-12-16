import numpy as np
import pandas as pd
import sklearn
from featurebox.featurizers.compositionfeaturizer import DepartElementFeaturizer
from mgetool.exports import Store
from pymatgen import Composition

if __name__ == "__main__":

    import os

    store = Store()

    os.chdir(r'band_gap')

    com_data = pd.read_excel(r'initial_band_gap_data.xlsx')
    #
    # """for element site"""
    from bgp.data.impot_element_table import element_table

    name_and_abbr = element_table.iloc[[0, 1], :]
    element_table = element_table.iloc[2:, :]

    feature_select = [
        'lattice constants a',
        'lattice constants b',
        'lattice constants c',
        'atomic radii(empirical)',
        'atomic radii(clementi)',
        'ionic radii(pauling)',
        'ionic radii(shannon)',
        'covalent radii',
        'metal radii(waber)',
        'valence electron distance(schubert)',
        'core electron distance(schubert)',
        'pseudo-potential radii(zunger)',

        'first ionization energy',
        'second ionization energy',
        'third ionization energy',
        'atomization enthalpy',
        'vaporization enthalpy',
        'fusion enthalpy',

        'cohesive energy(Brewer)',
        'total energy',

        'electron number',
        'valence electron number',
        'effective nuclear charge(slater)',
        'effective nuclear charge(clementi)',
        'electronegativity(martynov&batsanov)',
        'electronegativity(pauling)',
        'electronegativity(alfred-rochow)',
        'atomic volume(villars,daams)',

    ]
    #
    select_element_table = element_table[feature_select]
    #
    # """transform composition to pymatgen Composition"""
    composition = pd.Series(map(eval, com_data['composition']))
    composition_mp = pd.Series(map(Composition, composition))


    # # """get ele_ratio"""

    def comdict_to_df(composition_mp):
        composition_mp = pd.Series([i.to_reduced_dict for i in composition_mp])
        com = [[j[i] for i in j] for j in composition_mp]
        com = pd.DataFrame(com)
        colu_name = {}
        for i in range(com.shape[1]):
            colu_name[i] = "com_%s" % i
        com.rename(columns=colu_name, inplace=True)
        return com


    ele_ratio = comdict_to_df(composition_mp)
    # #
    # #
    """get departed element feature"""
    departElementProPFeature = DepartElementFeaturizer(elem_data=select_element_table, n_composition=2, n_jobs=1, )
    departElement = departElementProPFeature.fit_transform(composition_mp)
    # """join"""
    depart_elements_table = departElement.set_axis(com_data.index.values, axis='index', inplace=False)
    ele_ratio = ele_ratio.set_axis(com_data.index.values, axis='index', inplace=False)
    # #
    all_import_title = com_data.join(ele_ratio)
    all_import_title = all_import_title.join(depart_elements_table)

    """add ele density"""
    select2 = ['electron number_0', 'electron number_1', 'cell volume']
    x_rame = (all_import_title['electron number_0'] + all_import_title['electron number_1']) / all_import_title[
        'cell volume']

    all_import_title.insert(10, "electron density", x_rame, )

    all_import = all_import_title.drop(
        ['name_number', 'name_number', "name", "structure", "structure_type", "space_group", "reference", 'material_id',
         'composition', "com_0", "com_1"], axis=1)

    all_import = all_import.iloc[np.where(all_import['group_number'] == 216)[0]]
    # store.to_csv(all_import, "all_import_216", reverse=False)

    data_import = all_import
    data216_import = data_import
    X_frame = data216_import.drop(['exp_gap'], axis=1)
    y_frame = data216_import['exp_gap']
    Y = y_frame.values
    X = X_frame.values

    ##################
    X_rho = X_frame['electron density']
    X_chi_mA = X_frame['electronegativity(martynov&batsanov)_0']
    X_chi_mB = X_frame['electronegativity(martynov&batsanov)_1']

    X = np.vstack((X_rho, X_chi_mA, X_chi_mB)).T
    #
    # x, y = shuffle(X, Y, random_state=5)
    #
    # nor = MaxAbsScaler()
    # x = nor.fit_transform(x)
    #
    #
    #
    # x_dim = [Dim([-3,0,0,0,0,0,0]),Dim([-1,0,0,0,0,0,0]),Dim([-1,0,0,0,0,0,0])]
    # y_dim = Dim([2,-1,2,0,0,0,0])
    #
    #
    # pset0 = SymbolSet()
    # pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[(1,2),])
    # pset0.add_operations(power_categories=(2, 3, 0.5, 1 / 3),
    #                      categories=("Add", "Mul", "Sub", "Div", "exp", "ln", "Self"),
    #                      self_categories=None)
    #
    # total_height = 4
    # h_bgp = 2
    # This random_state is under Linux system. For others system ,the random_state maybe different,please
    # try with different random_state.
    # for i in range(1, 10):
    #     stop = lambda ind: ind.fitness.values[0] >= 0.95
    #     sl = SymbolLearning(loop="MultiMutateLoop", pset=pset0, gen=20, pop=1000, hall=1, batch_size=40, re_hall=3,
    #                         n_jobs=4, mate_prob=0.9, max_value=h_bgp, initial_min=2, initial_max=h_bgp,
    #                         mutate_prob=0.8, dim_type="coef", stop_condition=stop,
    #                         re_Tree=0, store=False, random_state=4, verbose=True,tq=True,
    #                         # scoring=(sklearn.metrics.mean_absolute_error,),score_pen=(-1,),
    #                         # stats={"fitness_dim_min": ["min"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
    #                         scoring=(sklearn.metrics.r2_score,), score_pen=(1,),
    #                         stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
    #                         add_coef=True, inter_add=True, out_add=True, cal_dim=True, vector_add=True,
    #                         personal_map=False)
    #     sl.fit()
    #     score = sl.score(x, y, "r2")
    #     print(i,score,sl.expr)
    #     y_pre = sl.predict(x)
    #
    #     p = BasePlot(font=None)
    #     p.scatter(y, y_pre, strx='Experimental $E_{gap}$', stry='Calculated $E_{gap}$')
    #     import matplotlib.pyplot as plt
    #
    #     plt.show()
    #     break

    from sklearn.linear_model import LinearRegression

    lin = LinearRegression()

    # XX = np.vstack((X[:, 1] - X[:, 2]) / X[:, 0]).reshape(-1, 1)
    #
    # XX = np.vstack((X[:, 1]*X[:, 0]**(-0.5), X[:, 2]*X[:, 0]**(-0.5),)).T
    #
    XX = np.vstack((X[:, 1], X[:, 2], X[:, 0] ** (0.33))).T

    # XX = np.vstack((X[:, 1] ** 0.333,
    #                 X[:, 24] / (X[:, 22] ** 0.333 + X[:, 23] ** 0.333),
    #                 X[:, 25] / (X[:, 22] ** 0.333 + X[:, 23] ** 0.333),
    #                 )).T
    lin.fit(XX, Y)
    coef = lin.coef_
    inter = lin.intercept_
    score_mae = sklearn.metrics.median_absolute_error(lin.predict(XX), Y)
    score_r2 = sklearn.metrics.r2_score(lin.predict(XX), Y)
    z = lin.predict(XX)
