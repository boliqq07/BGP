# -*- coding: utf-8 -*-

# @Time    : 2021/7/21 21:34
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import warnings
from functools import partial

from sklearn import kernel_ridge, gaussian_process, neighbors
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import BayesianRidge, SGDRegressor, Lasso, ElasticNet, PassiveAggressiveRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


kernel = 1.0 * RBF(1.0)
kernel2 = Matern(nu=0.5)
kernel3 = Matern(nu=1.5)
kernel4 = Matern(nu=2.5)
kernel5 = Matern(nu=0.5, length_scale=0.8)
kernel6 = Matern(nu=1.5, length_scale=0.8)
kernel7 = Matern(nu=2.5, length_scale=0.8)

ker = [kernel, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7]



def dict_method_reg():
    dict_method = {}
    # 1st part

    """4KNR"""
    me4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski')
    cv4 = 5
    scoring4 = 'r2'
    param_grid4 = [{'n_neighbors': [4, 5, 6, 7, 8], "leaf_size": [10, 20, 30]
                    }]
    dict_method.update({"KNR-set": [me4, cv4, scoring4, param_grid4]})

    """1SVR"""
    me1 = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=True, max_iter=2000)
    cv1 = 5
    scoring1 = 'r2'
    param_grid1 = [{'C': [10, 1, 0.1, 0.01, 0.001], 'kernel': ker}]
    dict_method.update({"SVR-set": [me1, cv1, scoring1, param_grid1]})

    """5kernelridge"""
    me5 = kernel_ridge.KernelRidge(alpha=1, gamma="scale", degree=3, coef0=-1, kernel_params=None)
    cv5 = 5
    scoring5 = 'r2'
    param_grid5 = [{'alpha': [10, 1, 0.1, 0.001], 'kernel': ker}]
    dict_method.update({'KRR-set': [me5, cv5, scoring5, param_grid5]})

    """6GPR"""
    me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                                    n_restarts_optimizer=0,
                                                    normalize_y=False, copy_X_train=True, random_state=0)
    cv6 = 5
    scoring6 = 'r2'
    param_grid6 = [{'kernel': ker}]
    dict_method.update({"GPR-set": [me6, cv6, scoring6, param_grid6]})

    # 2nd part

    """6RFR"""
    me7 = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, bootstrap=True, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    cv7 = 5
    scoring7 = 'r2'
    param_grid7 = [{'max_depth': [3, 4, 5], }]
    dict_method.update({"RFR-em": [me7, cv7, scoring7, param_grid7]})

    """7GBR"""
    me8 = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                    subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                    max_depth=3, min_impurity_decrease=0.,
                                    min_impurity_split=None, init=None, random_state=None,
                                    max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                    warm_start=False, )
    cv8 = 5
    scoring8 = 'r2'
    param_grid8 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'GBR-em': [me8, cv8, scoring8, param_grid8]})

    "AdaBR"
    dt = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=5, min_samples_split=4)
    me9 = AdaBoostRegressor(dt, n_estimators=200, learning_rate=0.05, loss='linear', random_state=0)
    cv9 = 5
    scoring9 = 'explained_variance'
    param_grid9 = [{'n_estimators': [100, 200]}]
    dict_method.update({"AdaBR-em": [me9, cv9, scoring9, param_grid9]})

    '''DTR'''
    me10 = DecisionTreeRegressor(
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        )
    cv10 = 5
    scoring10 = 'r2'
    param_grid10 = [
        {'max_depth': [2, 3, 4, 5, 6, 7], "min_samples_split": [2, 3, 4], "min_samples_leaf": [1, 2]}]
    dict_method.update({'DTR-em': [me10, cv10, scoring10, param_grid10]})

    'ElasticNet'
    me11 = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
                      copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)

    cv11 = 5
    scoring11 = 'r2'
    param_grid11 = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'l1_ratio': [0.3, 0.5, 0.8]}]
    dict_method.update({"EN-L1": [me11, cv11, scoring11, param_grid11]})

    'Lasso'
    me12 = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=0.001,
                 warm_start=False, positive=False, random_state=None, )

    cv12 = 5
    scoring12 = 'r2'
    param_grid12 = [{'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}, ]
    dict_method.update({"LASSO-L1": [me12, cv12, scoring12, param_grid12]})

    """2BayesianRidge"""
    me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                        n_iter=300, normalize=False, tol=0.01, verbose=False)
    cv2 = 5
    scoring2 = 'r2'
    param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05], 'alpha_2': [1e-07, 1e-06, 1e-05]}]
    dict_method.update({'BRR-L1': [me2, cv2, scoring2, param_grid2]})

    """3SGDRL2"""
    me3 = SGDRegressor(alpha=0.0001, average=False,
                       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                       penalty='l2', power_t=0.25,
                       random_state=0, shuffle=True, tol=0.01,
                       verbose=0, warm_start=False)
    cv3 = 5
    scoring3 = 'r2'
    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05], 'loss': ['squared_loss', "huber"],
                    "penalty": ["l1", "l2"]}]
    dict_method.update({'SGDR-L1': [me3, cv3, scoring3, param_grid3]})

    """PassiveAggressiveRegressor"""
    me14 = PassiveAggressiveRegressor(C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False,
                                      validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0,
                                      loss='epsilon_insensitive', epsilon=0.1, random_state=None,
                                      warm_start=False, average=False)
    cv14 = 5
    scoring14 = 'r2'
    param_grid14 = [{'C': [1.0e8, 1.0e6, 10000, 100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01]}]
    dict_method.update({'PAR-L1': [me14, cv14, scoring14, param_grid14]})

    return dict_method


def method_pack(method_all, me="reg", scoreing=None, gd=True, cv=10):
    if not method_all:
        method_all = ['KNR-set', 'SVR-set', "KRR-set", "GPR-set",
                      "RFR-em", "AdaBR-em", "DTR-em",
                      "LASSO-L1", "BRR-L1", "SGDR-L1", "PAR-L1"]
    dict_method = dict_method_reg()

    # print(dict_method.keys())
    if gd:
        estimator = []
        for method_i in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method_i]
            if me == "clf":
                scoring2 = scoreing if scoreing else 'balanced_accuracy'
            if me == "reg":
                scoring2 = scoreing if scoreing else 'r2'
            cv2 = cv if cv else cv2
            gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
            estimator.append(gd2)
        return estimator
    else:
        estimator = []
        for method_i in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method_i]
            if me == "clf":
                scoring2 = scoreing if scoreing else 'balanced_accuracy'
            if me == "reg":
                scoring2 = scoreing if scoreing else 'r2'
            cv2 = cv if cv else cv2
            gd2 = partial(cross_val_score, estimator=me2, cv=cv2, scoring=scoring2)
            # gd2 = cross_val_score(me2, cv=cv2, scoring=scoring2)
            estimator.append(gd2)
        return estimator

