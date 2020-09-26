# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 18:47
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 3.0select_method.py
# @Software: PyCharm
import numpy as np
import pandas as pd

from BGP.featurizers.compositionfeaturizer import DepartElementFeaturizer
from mgetool.exports import Store

# from pymatgen import Composition

"""
this is a description
"""
if __name__ == "__main__":

    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp')

    com_data = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp\init_band_data.xlsx',
                             sheet_name='binary_4_structure')

    """for element site"""
    element_table = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp\element_table.xlsx',
                                  header=4, skiprows=0, index_col=0)
    """get x_name and abbr"""


    def get_abbr():
        abbr = list(element_table.loc["abbrTex"])
        name = list(element_table.columns)
        name.extend(['face_dist1', 'vor_area1', 'face_dist2', 'vor_area2', "cell density", 'cell volume', "ele_ratio"])
        abbr.extend(['$d_{vf1}$', '$S_{vf1}$', '$d_{vf2}$', '$S_{vf2}$', r"$\rho_c$", "$V_c$", "$ele_ratio$"])

        return name, abbr


    def get_dim():
        dims = [np.array([0, 0, 0, 0, 0, 0, 0])] * 7 + [np.array([0, 1, 0, 0, 0, 0, 0])] * 14 + [
            np.array([1, 2, -2, 0, 0, 0, 0])] * 16 \
               + [np.array([0, 0, 0, 0, 0, 0, 0])] * 2 + [np.array([0, -1, 0, 0, 0, 0, 0])] * 2 + [
                   np.array([0, 0, 0, 0, 0, 0, 0])] * 11 \
               + [np.array([1, -1, -2, 0, 0, 0, 0])] * 8 + [np.array([1, 1, 3, 0, -1, 0, 0])] + [
                   np.array([1, 3, -3, -2, 0, 0, 0])] \
               + [np.array([0, 2, -2, 0, -1, 0, 0])] + [np.array([-1, -3, 3, 2, 0, 0, 0])] + [
                   np.array([0, 0, 0, 0, 0, 0, 0])] \
               + [np.array([0, 0, 0, 1, 0, 0, 0])] * 2 + [np.array([0, 0, 0, 0, 0, 0, 0])] + [
                   np.array([0, 3, 0, 0, 0, 0, 0])] \
               + [np.array([0, 0, 0, 0, 0, 0, 0])] + [np.array([0, 3, 0, 0, 0, -1, 0])] + [
                   np.array([1, -3, 0, 0, 0, 0, 0])] \
               + [np.array([1, -1, -2, 0, 0, 0, 0])]

        dims.extend([np.array([0, 1, 0, 0, 0, 0, 0]), np.array([0, 2, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0]),
                     np.array([0, 2, 0, 0, 0, 0, 0]),
                     np.array([1, -3, 0, 0, 0, 0, 0]), np.array([0, 3, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])
                     ])
        return dims


    name_and_abbr = get_abbr()
    dims = get_dim()

    element_table = element_table.iloc[5:, 7:]
    feature_select = [
        'lattice constants a',
        'lattice constants b',
        'lattice constants c',
        'radii atomic(empirical)',
        'radii atomic(clementi)',
        'radii ionic(pauling)',
        'radii ionic(shannon)',
        'radii covalent',
        'radii metal(waber)',
        'distance valence electron(schubert)',
        'distance core electron(schubert)',
        'radii pseudo-potential(zunger)',

        'energy ionization first',
        'energy ionization second',
        'energy ionization third',
        'enthalpy atomization',
        'enthalpy vaporization',
        'latent heat of fusion',

        'energy cohesive brewer',
        'total energy',

        'electron number',
        'valence electron number',
        'charge nuclear effective(slater)',
        'charge nuclear effective(clementi)',
        'periodic number',
        'electronegativity(martynov&batsanov)',
        'electronegativity(pauling)',
        'electronegativity(alfred-rochow)',

        'volume atomic(villars,daams)',

    ]

    select_element_table = element_table[feature_select]

    """transform composition to pymatgen Composition"""
    composition_mp = pd.Series(map(eval, com_data['composition']))
    # composition_mp = pd.Series(map(Composition, composition))

    """get ele_ratio"""


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

    """get structure"""
    # with MPRester('Di2IZMunaeR8vr9w') as m:
    #     ids = [i for i in com_data['material_id']]
    #     structures = [m.get_structure_by_material_id(i) for i in ids]
    # store.to_pkl_pd(structures, "id_structures")
    # id_structures = pd.read_pickle(
    #     r'C:\Users\Administrator\Desktop\band_gap_exp\1.generate_data\id_structures.pkl.pd')

    """get departed element feature"""
    departElementProPFeature = DepartElementFeaturizer(elem_data=select_element_table, n_composition=2, n_jobs=1, )
    departElement = departElementProPFeature.fit_transform(composition_mp)
    """join"""
    depart_elements_table = departElement.set_axis(com_data.index.values, axis='index', inplace=False)
    ele_ratio = ele_ratio.set_axis(com_data.index.values, axis='index', inplace=False)

    all_import_title = com_data.join(ele_ratio)
    all_import_title = all_import_title.join(depart_elements_table)

    """sub density to e density"""
    select2 = ['electron number_0', 'electron number_1', 'cell volume']
    x_rame = (all_import_title['electron number_0'] + all_import_title['electron number_1']) / all_import_title[
        'cell volume']
    all_import_title['cell density'] = x_rame
    all_import_title.rename(columns={'cell density': "electron density"}, inplace=True)

    name = ["electron density" if i == "cell density" else i for i in name_and_abbr[0]]
    abbr = [r"$\rho_e$" if i == r"$\rho_c$" else i for i in name_and_abbr[1]]
    name_and_abbr = [name, abbr]
    dims[-3] = np.array([0, -3, 0, 0, 0, 0, 0])

    store.to_csv(all_import_title, "all_import_title")
    all_import = all_import_title.drop(
        ['name_number', 'name_number', "name", "structure", "structure_type", "space_group", "reference", 'material_id',
         'composition', "com_0", "com_1"], axis=1)

    store.to_pkl_pd(dims, "dims")
    store.to_pkl_pd(name_and_abbr, "name_and_abbr")
    store.to_csv(all_import, "all_import")
