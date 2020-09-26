# -*- coding: utf-8 -*-

# @Time   : 2019/6/13 21:04
# @Author : Administrator
# @Project : feature_toolbox
# @FileName: 1.1add_compound_features.py
# @Software: PyCharm
import re
import warnings

import numpy as np
import pandas as pd
from BGP.combination.dimanalysis import dimension_check
from BGP.selection.quickmethod import dict_method_reg
from BGP.selection.sum import SUM
from mgetool.exports import Store
from mgetool.imports import Call
from mgetool.tool import name_to_name
from sklearn import utils
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

"""
this is a description
"""
if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum\sub')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp',
                r'C:\Users\Administrator\Desktop\band_gap_exp\3.sum\method', )
    data_import = data.csv().all_import
    name_init, abbr_init = data.pickle_pd().name_and_abbr

    select = ['cell volume', 'electron density', 'lattice constants a', 'lattice constants c', 'radii covalent',
              'radii ionic(shannon)',
              'distance core electron(schubert)', 'latent heat of fusion', 'energy cohesive brewer', 'total energy',
              'charge nuclear effective(slater)', 'valence electron number', 'electronegativity(martynov&batsanov)',
              'volume atomic(villars,daams)']

    select = ['cell volume', 'electron density'] + [j + "_%i" % i for j in select[2:] for i in range(2)]

    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    """base_method"""

    method_name = ['GPR-set', 'SVR-set', 'KRR-set', 'KNR-set', 'GBR-em', 'AdaBR-em', 'RFR-em', "DTR-em"]

    index_all = [data.pickle_pd().GPR_set23, data.pickle_pd().SVR_set23, data.pickle_pd().KRR_set23]

    estimator_all = []
    for i in method_name:
        me1, cv1, scoring1, param_grid1 = dict_method_reg()[i]
        estimator_all.append(GridSearchCV(me1, cv=cv1, scoring=scoring1, param_grid=param_grid1, n_jobs=1))

    """union"""
    # [print(_[0]) for _ in index_all]
    index_slice = [tuple(index[0]) for _ in index_all for index in _[:round(len(_) / 3)]]
    index_slice = list(set(index_slice))
    index_slice = [index_slice[i] for i in [100, 8, 6, 152, 81, 106, 129, 19, 73, 170]]

    """get x_name and abbr"""
    index_all_name = name_to_name(X_frame.columns.values, search=[i for i in index_slice],
                                  search_which=0, return_which=(1,), two_layer=True)

    index_all_name = [list(set([re.sub(r"_\d", "", j) for j in i])) for i in index_all_name]
    [i.sort() for i in index_all_name]
    index_all_abbr = name_to_name(name_init, abbr_init, search=index_all_name, search_which=1, return_which=2,
                                  two_layer=True)

    parto = []
    table = []

    for i in range(100):
        print(i)
        X = X_frame.values
        y = y_frame.values

        # scal = preprocessing.MinMaxScaler()
        # X = scal.fit_transform(X)
        X, y = utils.shuffle(X, y, random_state=i)

        # X,y = utils.resample(X,y,replace=True,random_state=i)
        # from sklearn.model_selection import train_test_split
        # sample = 0
        # test_size = 1-sample
        # X_train, _X_reserved_test, y_train, _y_reserved_test = train_test_split(X, y, test_size=test_size,
        # random_state=i)
        # X, y = X_train, y_train

        """run"""
        self = SUM(estimator_all, index_slice, estimator_n=[0, 1, 2, 3, 4, 5, 6, 7], n_jobs=1)
        self.fit(X, y)
        mp = self.pareto_method()
        partotimei = list(list(zip(*mp))[0])

        tabletimei = np.vstack([self.resultcv_score_all_0, self.resultcv_score_all_1, self.resultcv_score_all_2,
                                self.resultcv_score_all_3, self.resultcv_score_all_4, self.resultcv_score_all_5,
                                self.resultcv_score_all_6, self.resultcv_score_all_7])

        parto.extend(partotimei)
        table.append(tabletimei)

    table = np.array(table)
    means_y = np.mean(table, axis=0).T
    result = pd.DataFrame(means_y)
    all_mean = np.mean(means_y, axis=1).T
    all_std = np.std(means_y, axis=1).T

    select_support = np.zeros(len(index_slice))
    mean_parto_index = self._pareto(means_y)
    select_support[mean_parto_index] = 1

    result["all_mean"] = all_mean
    result["all_std"] = all_std
    result["parto_support"] = select_support
    result['index_all_abbr'] = index_all_abbr
    result['index_all_name'] = index_all_name
    result['index_all'] = index_slice

    dim_init = data.dims

    index_all_name = name_to_name(X_frame.columns.values, search=[i for i in index_slice],
                                  search_which=0, return_which=(1,), two_layer=True)
    index_all_name = [list([re.sub(r"_\d", "", j) for j in i]) for i in index_all_name]

    index_all_dim = name_to_name(name_init, dim_init, search=index_all_name, search_which=1, return_which=2,
                                 two_layer=True)

    dim_target = [dimension_check(list(dim), np.array([1, 2, -2, 0, 0, 0, 0])) for dim in index_all_dim]
    dim_1 = [dimension_check(dim) for dim in index_all_dim]

    result['dim1'] = dim_1
    result['dim_target'] = dim_target

    result = result.sort_values(by="all_mean", ascending=False)
    store.to_csv(result, "result")
