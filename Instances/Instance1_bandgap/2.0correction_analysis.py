#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# @Time   : 2019/8/12 17:44
# @Author : Administrator
# @Software: PyCharm
# @License: BSD 3-Clause

"""
this is a description
"""

from featurebox.selection.corr import Corr
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.show import corr_plot
from mgetool.tool import name_to_name

# import seaborn as sns

if __name__ == "__main__":
    import os

    os.chdir(r'band_gap')

    store = Store()
    data = Call()
    all_import = data.csv().all_import
    name_and_abbr = data.csv().name_and_abbr

    data_import = all_import
    data225_import = data_import
    X_frame = data225_import.drop(['exp_gap'], axis=1)
    y_frame = data225_import['exp_gap']
    X = X_frame.values
    y = y_frame.values
    #
    # """calculate corr"""
    corr = Corr(threshold=0.90, muti_grade=2, muti_index=[2, len(X)])
    corr.fit(X_frame)
    cof_list = corr.count_cof()
    #
    """get x_name and abbr"""

    X_frame_name = corr.transform(X_frame.columns.values)
    X_frame_name = [i.replace("_0", "") for i in X_frame_name]

    X_frame_abbr = name_to_name(name_and_abbr.columns.values, list(name_and_abbr.iloc[0, :]),
                                search=X_frame_name, search_which=1,
                                return_which=[2, ],
                                two_layer=False)

    """rename"""
    # cov = pd.DataFrame(corr.cov_shrink)
    # cov = cov.set_axis(X_frame_abbr, axis='index', inplace=False)
    # cov = cov.set_axis(X_frame_abbr, axis='columns', inplace=False)
    #
    # fig = plt.figure()
    # fig.add_subplot(111)
    # sns.heatmap(cov, vmin=-1, vmax=1, cmap="bwr", linewidths=0.3, xticklabels=True, yticklabels=True, square=True,
    #             annot=True, annot_kws={'size': 3})
    # plt.show()
    #
    plt0 = corr_plot(corr.cov_shrink, X_frame_abbr, title="", left_down="fill", right_top="pie", threshold_right=0.8,
                     front_raito=0.6)

    plt0.savefig("corr.pdf")

    # list_name, list_abbr = name_to_name(X_frame_name, X_frame_abbr, search=corr.list_count, search_which=0,
    #                                     return_which=(1, 2),
    #                                     two_layer=True)
    #
    # store.to_csv(cov, "cov")
    # store.to_txt(list_name, "list_name")
    # store.to_txt(list_abbr, "list_abbr")

    # 2
    select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'covalent radii',
              'ionic radii(shannon)',
              'core electron distance(schubert)', 'fusion enthalpy', 'cohesive energy(Brewer)', 'total energy',
              'effective nuclear charge(slater)', "electron number", 'valence electron number',
              'electronegativity(martynov&batsanov)',
              'atomic volume(villars,daams)']  # human select

    select_index, select_abbr = name_to_name(X_frame_name, X_frame_abbr, search=select, search_which=1,
                                             return_which=(0, 2),
                                             two_layer=False)

    cov_select = corr.cov_shrink[select_index, :][:, select_index]

    # store.to_csv(cov_select, "cov_select")
    # store.to_txt(select, "list_name_select")
    # store.to_txt(select_abbr, "list_abbr_select")
    #
    # corr_plot(cov_select, select_abbr, title="", left_down="circle", right_top="pie", threshold_right=0, front_raito=0.8)
