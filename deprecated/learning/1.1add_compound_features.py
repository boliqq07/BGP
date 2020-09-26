# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 21:04
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 1.1add_compound_features.py
# @Software: PyCharm

import pandas as pd
import pymatgen as mg

from bgp.featurizers.voronoifeature import count_voronoinn
from mgetool.exports import Store
from mgetool.imports import Call

"""
this is a description
"""
store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data')
data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data')

com_data = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp_last\init_band_data.xlsx',
                         sheet_name='binary_4_structure', header=0, skiprows=None, index_col=0, names=None)
composition = pd.Series(map(eval, com_data['composition']))
composition_mp = pd.Series(map(mg.Composition, composition))
"""for element site"""
com_mp = pd.Series([i.to_reduced_dict for i in composition_mp])
# com_mp = composition_mp
all_import = data.csv.all_import
id_structures = data.id_structures
structures = id_structures
vor_area = count_voronoinn(structures, mess="area")
vor_dis = count_voronoinn(structures, mess="face_dist")
vor = pd.DataFrame()
vor.insert(0, 'vor_area0', vor_area[:, 0])
vor.insert(0, 'face_dist0', vor_dis[:, 0])
vor.insert(0, 'vor_area1', vor_area[:, 1])
vor.insert(0, 'face_dist1', vor_dis[:, 1])

data_title = all_import[
    ['name_number', "x_name", "structure", "structure_type", "space_group", "reference", 'material_id', 'composition',
     'exp_gap', 'group_number']]

data_tail = all_import.drop(
    ['name_number', "x_name", "structure", "structure_type", "space_group", "reference", 'material_id', 'composition',
     'exp_gap', 'group_number'], axis=1)

data_import = data_title.join(vor[["face_dist0", "vor_area0", "face_dist1", "vor_area1"]])
data_import = data_import.join(data_tail)

store.to_csv(data_import, "all_import")
