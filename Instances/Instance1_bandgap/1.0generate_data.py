# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 18:47
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 3.0select_method.py
# @Software: PyCharm
import numpy as np
import pandas as pd

from featurebox.featurizers.atom.mapper import AtomTableMap
from featurebox.featurizers.state.statistics import DepartElementFeature

from mgetool.exports import Store
from pymatgen.core import Composition

"""
this is a description
"""
if __name__ == "__main__":

    import os

    os.chdir(r'band_gap')

    store = Store()

    com_data = pd.read_csv('initial_band_gap_data.csv')
    #
    # """for element site"""
    element_table = pd.read_csv("ele_table.csv", index_col=0)

    name_and_abbr = element_table.iloc[[0, 1], :]
    element_table = element_table.iloc[2:, :]

    feature_select = [
        'lattice constants a',
        'lattice constants b',
        'lattice constants c',
        'atomic radii (empirical)',
        'atomic radii (clementi)',
        'ionic radii (pauling)',
        'ionic radii (shannon)',
        'covalent radii',
        'metal radii (waber)',
        'valence electron distance (schubert)',
        'core electron distance (schubert)',
        'pseudo-potential radii (zunger)',

        'first ionization energy',
        'second ionization energy',
        'third ionization energy',
        'atomization enthalpy',
        'vaporization enthalpy',
        'fusion enthalpy',

        'cohesive energy (Brewer)',
        'total energy',

        'electron number',
        'valence electron number',
        'effective nuclear charge (slater)',
        'effective nuclear charge (clementi)',
        "periodic number",
        "group number",
        'electronegativity (martynov&batsanov)',
        'electronegativity (pauling)',
        'electronegativity (alfred-rochow)',

        'atomic volume (villars,daams)',

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


    #
    ele_ratio = comdict_to_df(composition_mp)
    # #
    # #
    """get departed element feature"""
    data_map = AtomTableMap(search_tp="name", tablename=select_element_table, n_jobs=1)
    departElementProPFeature = DepartElementFeature(data_map=data_map, n_composition=2, n_jobs=1, )
    departElementProPFeature.set_feature_labels(data_map.feature_labels)
    departElement = departElementProPFeature.fit_transform(composition_mp)

    # """join"""
    depart_elements_table = departElement.set_axis(com_data.index.values, axis='index', inplace=False)
    ele_ratio = ele_ratio.set_axis(com_data.index.values, axis='index', inplace=False)
    # #
    all_import_title = com_data._join(ele_ratio)
    all_import_title = all_import_title._join(depart_elements_table)

    """add ele density"""
    select2 = ['electron number_0', 'electron number_1', 'cell volume']
    x_rame = (all_import_title['electron number_0'] + all_import_title['electron number_1']) / all_import_title[
        'cell volume']

    all_import_title.insert(10, "electron density", x_rame, )

    # store.to_csv(all_import_title, "all_import_title", reverse=False)

    all_import = all_import_title.drop(
        ['name_number', "cell density", 'name_number', "name", "structure", "structure_type", "space_group",
         "reference", 'material_id',
         'composition', "com_0", "com_1"], axis=1)

    all_import = all_import.iloc[np.where(all_import['group_number'] == 225)[0]]
    all_import = all_import.drop(['group_number'], axis=1)

    store.to_csv(all_import, "all_import", transposition=False)


    def get_abbr():
        name = ["electron density", "cell density", 'cell volume', "component"]
        abbrTex = [r"$\rho_e$", r"$\rho_c$", "$V_c$", "$com$"]
        abbr = [r"rho_e", r"rho_c", "V_c", "com"]

        for i, j, k in zip(name, abbrTex, abbr):
            name_and_abbr.insert(0, i, [j, k])


    get_abbr()

    store.to_csv(name_and_abbr, "name_and_abbr", transposition=False)
