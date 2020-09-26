#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/12 17:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""
import numpy as np
import pandas as pd

from BGP.selection.corr import Corr
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import name_to_name

# import seaborn as sns

if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\2.corr')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp')
    all_import = data.csv().all_import

    name_init, abbr_init = data.pickle_pd().name_and_abbr

    data_import = all_import
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    X_frame = data225_import.drop(['exp_gap', 'group_number'], axis=1)
    y_frame = data225_import['exp_gap']
    X = X_frame.values
    y = y_frame.values

    """calculate corr"""
    corr = Corr(threshold=0.90, muti_grade=2, muti_index=[2, len(X)])
    corr.fit(X_frame)
    cof_list = corr.count_cof()

    """get x_name and abbr"""

    X_frame_name = corr.transform(X_frame.columns.values)
    X_frame_name = [i.replace("_0", "") for i in X_frame_name]

    X_frame_abbr = name_to_name(name_init, abbr_init, search=X_frame_name, search_which=1, return_which=2,
                                two_layer=False)

    """rename"""
    # cov = pd.DataFrame(corr.cov_shrink)
    # cov = cov.set_axis(X_frame_abbr, axis='index', inplace=False)
    # cov = cov.set_axis(X_frame_abbr, axis='columns', inplace=False)
    #
    # # fig = plt.figure()
    # # fig.add_subplot(111)
    # # sns.heatmap(cov, vmin=-1, vmax=1, cmap="bwr", linewidths=0.3, xticklabels=True, yticklabels=True, square=True,
    # #             annot=True, annot_kws={'size': 3})
    # # plt.show()
    #
    # corr_plot(corr.cov_shrink, X_frame_abbr,title="", left_down="fill", right_top="pie", threshold_right=0, front_raito=0.4)
    #
    # list_name, list_abbr = name_to_name(X_frame_name, X_frame_abbr, search=corr.list_count, search_which=0,
    #                                     return_which=(1, 2),
    #                                     two_layer=True)
    # #
    # # store.to_csv(cov, "cov")
    # # store.to_txt(list_name, "list_name")
    # # store.to_txt(list_abbr, "list_abbr")
    #
    # # 2
    # select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'radii covalent',
    #           'radii ionic(shannon)',
    #           'distance core electron(schubert)', 'latent heat of fusion', 'energy cohesive brewer', 'total energy',
    #           'charge nuclear effective(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
    #           'volume atomic(villars,daams)']  # human select
    #
    # select_index, select_abbr = name_to_name(X_frame_name, X_frame_abbr, search=select, search_which=1,
    #                                          return_which=(0, 2),
    #                                          two_layer=False)
    #
    # cov_select = corr.cov_shrink[select_index, :][:, select_index]
    #
    # # store.to_csv(cov_select, "cov_select")
    # # store.to_txt(select, "list_name_select")
    # # store.to_txt(select_abbr, "list_abbr_select")
    #
    # corr_plot(cov_select, select_abbr,title="", left_down="fill", right_top="pie", threshold_right=0, front_raito=0.6)

    select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'radii covalent',
              'radii ionic(shannon)',
              'distance core electron(schubert)', 'latent heat of fusion', 'energy cohesive brewer', 'total energy',
              'charge nuclear effective(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'volume atomic(villars,daams)']

    X_frame_abbr = name_to_name(name_init, abbr_init, search=select, search_which=1, return_which=2,
                                two_layer=False)

    select = ['cell volume', 'electron density', ] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    corr = Corr(threshold=0.90, muti_grade=2, muti_index=[2, len(X)])
    corr.fit(X_frame[select].values, method="mean", pre_cal="mic", alpha=0.55, c=15)
    # cof_mic = corr.count_cof()
    cof_mic = corr.cov_shrink
    cov = pd.DataFrame(corr.cov_shrink)
    cov = cov.set_axis(X_frame_abbr, axis='index', inplace=False)
    cov = cov.set_axis(X_frame_abbr, axis='columns', inplace=False)
