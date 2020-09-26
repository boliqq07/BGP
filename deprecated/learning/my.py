import warnings

import numpy as np
import pandas as pd
from sklearn import kernel_ridge, gaussian_process, ensemble, linear_model, neighbors
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor, Lasso, ElasticNet, Perceptron
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate, LeaveOneOut, KFold
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def dict_method_clf():
    dict_method = {}
    # 1st part
    """1SVC"""
    me1 = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
              coef0=0.0, shrinking=True, probability=False,
              tol=1e-3, cache_size=200, class_weight='balanced',
              verbose=False, max_iter=-1, decision_function_shape='ovr',
              random_state=None)
    cv1 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring1 = 'accuracy'

    param_grid1 = [{'C': [10, 5, 2.5, 1, 0.5], 'gamma': [0.001, 0.01, 0.0001]}]

    dict_method.update({'SVC-set': [me1, cv1, scoring1, param_grid1]})

    """2LogRL2"""
    me2 = LogisticRegression(penalty='l2', solver='liblinear', dual=False, tol=1e-3, C=1.0, fit_intercept=True,
                             intercept_scaling=1, class_weight='balanced', random_state=0)
    cv2 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring2 = 'accuracy'

    param_grid2 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]}, ]

    dict_method.update({"LogRL2-set": [me2, cv2, scoring2, param_grid2]})

    """3SGDCL2"""
    me3 = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                                     fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                                     verbose=0, epsilon=0.1, random_state=0,
                                     learning_rate='optimal', eta0=0.0, power_t=0.5,
                                     class_weight="balanced", warm_start=False, average=False, n_iter=None)
    cv3 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring3 = 'accuracy'

    param_grid3 = [{'alpha': [0.0001, 0.001, 0.01]}, ]

    dict_method.update({"SGDCL2-set": [me3, cv3, scoring3, param_grid3]})

    """4KNC"""
    me4 = neighbors.KNeighborsClassifier(n_neighbors=5)
    cv4 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring4 = 'balanced_accuracy'

    param_grid4 = [{'n_neighbors': [3, 4, 5]}, ]

    dict_method.update({"KNC-set": [me4, cv4, scoring4, param_grid4]})

    """5GPC"""
    kernel = 1.0 * RBF(1.0)
    me5 = gaussian_process.GaussianProcessClassifier(kernel=kernel)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring5 = 'balanced_accuracy'
    param_grid5 = [{'max_iter_predict': [100, ]}, ]

    dict_method.update({'GPC-set': [me5, cv5, scoring5, param_grid5]})

    # 2nd part
    '''TreeC'''
    me6 = DecisionTreeClassifier(
        criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced", presort=False)
    cv6 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring6 = 'accuracy'
    param_grid6 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'TreeC-em': [me6, cv6, scoring6, param_grid6]})

    '''GBC'''
    me7 = ensemble.GradientBoostingClassifier(
        loss='deviance', learning_rate=0.1, n_estimators=100,
        subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.,
        max_depth=3, min_impurity_decrease=0.,
        min_impurity_split=None, init=None,
        random_state=None, max_features=None, verbose=0,
        max_leaf_nodes=None, warm_start=False,
        presort='auto')
    cv7 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring7 = 'balanced_accuracy'
    param_grid7 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'GBC-em': [me7, cv7, scoring7, param_grid7]})

    '''RFC'''
    me8 = ensemble.RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                          max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                                          min_impurity_split=None, bootstrap=True, oob_score=False,
                                          random_state=None, verbose=0, warm_start=False,
                                          class_weight="balanced")
    cv8 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring8 = 'accuracy'
    param_grid8 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({"RFC-em": [me8, cv8, scoring8, param_grid8]})

    "AdaBC"
    me9 = AdaBoostClassifier(n_estimators=100, learning_rate=1., algorithm='SAMME.R',
                             random_state=0)
    cv9 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring9 = 'accuracy'
    param_grid9 = [{'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2),
                                       DecisionTreeClassifier(max_depth=3)]}]
    dict_method.update({"AdaBC-em": [me9, cv9, scoring9, param_grid9]})

    # 3nd

    'SGDCL1'
    me12 = linear_model.SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, l1_ratio=0.15,
                                      fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                                      verbose=0, epsilon=0.1, random_state=0,
                                      learning_rate='optimal', eta0=0.0, power_t=0.5,
                                      class_weight="balanced", warm_start=False, average=False, n_iter=None)
    cv12 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring12 = 'accuracy'
    param_grid12 = [{'alpha': [0.0001, 0.001, 0.01]}, ]
    dict_method.update({"SGDC-L1": [me12, cv12, scoring12, param_grid12]})

    "Per"
    me14 = Perceptron(penalty="l1", alpha=0.0001, fit_intercept=True, max_iter=None, tol=None,
                      shuffle=True, verbose=0, eta0=1.0, random_state=0,
                      class_weight=None, warm_start=False, n_iter=None)
    cv14 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring14 = 'accuracy'
    param_grid14 = [{'alpha': [0.0001, 0.001, 0.01]}, ]
    dict_method.update({"Per-L1": [me14, cv14, scoring14, param_grid14]})

    """LogRL1"""
    me15 = LogisticRegression(penalty='l1', solver='liblinear', dual=False, tol=1e-3, C=1.0, fit_intercept=True,
                              intercept_scaling=1, class_weight='balanced', random_state=0)
    cv15 = StratifiedKFold(5, shuffle=True, random_state=0)
    scoring15 = 'accuracy'
    param_grid15 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]}, ]
    dict_method.update({"LogR-L1": [me15, cv15, scoring15, param_grid15]})

    return dict_method


def dict_method_reg():
    dict_method = {}
    # 1st part
    """1SVR"""
    me1 = SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=False, max_iter=2000)
    cv1 = 5
    scoring1 = 'r2'
    param_grid1 = [{'C': [1, 0.75, 0.5, 0.25, 0.1], 'epsilon': [0.01, 0.001, 0.0001]}]
    dict_method.update({"SVR-set": [me1, cv1, scoring1, param_grid1]})

    """2BayesianRidge"""
    me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                        copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                        n_iter=300, normalize=False, tol=0.01, verbose=False)
    cv2 = 5
    scoring2 = 'r2'
    param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05], 'alpha_2': [1e-07, 1e-05, 1e-03]}]
    dict_method.update({'BayR-set': [me2, cv2, scoring2, param_grid2]})

    """3SGDRL2"""
    me3 = SGDRegressor(alpha=0.0001, average=False,
                       epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                       learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                       penalty='l2', power_t=0.25,
                       random_state=0, shuffle=True, tol=0.01,
                       verbose=0, warm_start=False)
    cv3 = 5
    scoring3 = 'r2'
    param_grid3 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05]}]
    dict_method.update({'SGDRL2-set': [me3, cv3, scoring3, param_grid3]})

    """4KNR"""
    me4 = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                        metric='minkowski')
    cv4 = 5
    scoring4 = 'r2'
    param_grid4 = [{'n_neighbors': [3, 4, 5, 6]}]
    dict_method.update({"KNR-set": [me4, cv4, scoring4, param_grid4]})

    """5kernelridge"""
    kernel = 1.0 * RBF(1.0)
    me5 = kernel_ridge.KernelRidge(alpha=1, kernel=kernel, gamma="scale", degree=3, coef0=1, kernel_params=None)
    cv5 = 5
    scoring5 = 'r2'
    param_grid5 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001]}]
    dict_method.update({'KRR-set': [me5, cv5, scoring5, param_grid5]})

    """6GPR"""
    # kernel = 1.0 * RBF(1.0)
    kernel = Matern(length_scale=0.1, nu=0.5)
    me6 = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                                    n_restarts_optimizer=10,
                                                    normalize_y=False, copy_X_train=True, random_state=0)
    cv6 = 5
    scoring6 = 'r2'
    param_grid6 = [{'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}]
    dict_method.update({"GPR-set": [me6, cv6, scoring6, param_grid6]})

    # 2nd part

    """6RFR"""
    me7 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                         min_impurity_split=None, bootstrap=True, oob_score=False,
                                         random_state=None, verbose=0, warm_start=False)
    cv7 = 5
    scoring7 = 'r2'
    param_grid7 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({"RFR-em": [me7, cv7, scoring7, param_grid7]})

    """7GBR"""
    me8 = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100,
                                             subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                                             min_samples_leaf=1, min_weight_fraction_leaf=0.,
                                             max_depth=3, min_impurity_decrease=0.,
                                             min_impurity_split=None, init=None, random_state=None,
                                             max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                             warm_start=False, presort='auto')
    cv8 = 5
    scoring8 = 'r2'
    param_grid8 = [{'max_depth': [3, 4, 5, 6]}]
    dict_method.update({'GBR-em': [me8, cv8, scoring8, param_grid8]})

    "AdaBR"
    dt = DecisionTreeRegressor(criterion="mae", splitter="best", max_features=None, max_depth=3, min_samples_split=2)
    me9 = AdaBoostRegressor(dt, n_estimators=100, learning_rate=1, loss='square', random_state=0)
    cv9 = 5
    scoring9 = 'r2'
    param_grid9 = [{'n_estimators': [50, 120, 100, 200]}]
    dict_method.update({"AdaBR-em": [me9, cv9, scoring9, param_grid9]})

    '''TreeR'''
    me10 = DecisionTreeRegressor(
        criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=0, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
    cv10 = 5
    scoring10 = 'r2'
    param_grid10 = [{'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 3, 4]}]
    dict_method.update({'TreeC-em': [me10, cv10, scoring10, param_grid10]})

    'ElasticNet'
    me11 = ElasticNet(alpha=1.0, l1_ratio=0.7, fit_intercept=True, normalize=False, precompute=False, max_iter=1000,
                      copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None)

    cv11 = 5
    scoring11 = 'r2'
    param_grid11 = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'l1_ratio': [0.3, 0.5, 0.8]}]
    dict_method.update({"ElasticNet-L1": [me11, cv11, scoring11, param_grid11]})

    'Lasso'
    me12 = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000,
                 tol=0.001,
                 warm_start=False, positive=False, random_state=None, )

    cv12 = 5
    scoring12 = 'r2'
    param_grid12 = [{'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}, ]
    dict_method.update({"Lasso-L1": [me12, cv12, scoring12, param_grid12]})

    """SGDRL1"""
    me13 = SGDRegressor(alpha=0.0001, average=False,
                        epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
                        learning_rate='invscaling', loss='squared_loss', max_iter=1000,
                        penalty='l1', power_t=0.25,
                        random_state=0, shuffle=True, tol=0.01,
                        verbose=0, warm_start=False)
    cv13 = 5
    scoring13 = 'r2'
    param_grid13 = [{'alpha': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7], "epsilon": [0.1, 0.2, 1]}]
    dict_method.update({'SGDR-L1': [me13, cv13, scoring13, param_grid13]})

    """LinearSVR"""
    me14 = LinearSVR(epsilon=0.0, tol=1e-4, C=1.0,
                     loss='epsilon_insensitive', fit_intercept=True,
                     intercept_scaling=1., dual=True, verbose=0,
                     random_state=3, max_iter=1000)
    cv14 = 5
    scoring14 = 'r2'
    param_grid14 = [{'C': [10, 6, 5, 3, 2.5, 1, 0.75, 0.5, 0.25, 0.1], 'epsilon': [0.0, 0.1]}]
    dict_method.update({"LinearSVR-set": [me14, cv14, scoring14, param_grid14]})

    return dict_method


def dict_me(me="clf"):
    if me == "clf":
        dict_method_ = dict_method_clf()
    else:
        dict_method_ = dict_method_reg()
    return dict_method_


def method_pack(method_all, me="reg", gd=True):
    if not method_all:
        method_all = ['KNR-set', 'SVR-set', "KRR-set"]
    dict_method = dict_me(me=me)

    print(dict_method.keys())
    if gd:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method]
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'neg_mean_absolute_error'
            gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
            estimator.append(gd2)
        return estimator
    else:
        estimator = []
        for method in method_all:
            me2, cv2, scoring2, param_grid2 = dict_method[method]
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'neg_mean_absolute_error'
            gd2 = cross_val_score(me2, cv=cv2, scoring=scoring2)
            estimator.append(gd2)
        return estimator


def score_muti(x_select, y, me="reg", paras=True, method_name=None, shrink=2, str_name=False, param_grid=None):
    """score with different method
    :param param_grid: user's param_grid
    :param str_name:
    :param x_select: X
    :param y: y
    :param me: clf or reg
    :param paras: Gridsearch or not
    :param method_name: list or one x_name of method
    :param shrink: scale or not
    :return:
    """
    dict_method = dict_me(me=me)

    if method_name is not None:
        if isinstance(method_name, str):
            method_name = [method_name]
        dict_method = {_: dict_method[_] for _ in method_name}
    #         print(dict_method)

    # x_select, y = utils.shuffle(x_select, y, random_state=1)
    st = preprocessing.StandardScaler()
    x_select2 = st.fit_transform(x_select)

    if len(dict_method) > 1 and param_grid is not None:
        raise IndexError("only single method can accept param_grid, please set one method_name or param_grid=None ")

    score_all = []
    estimator = []

    for method in list(dict_method.keys()):
        me2, cv2, scoring2, param_grid2 = dict_method[method]
        if paras is None:
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'neg_mean_absolute_error'
            cv2 = 10
            if shrink == 1:
                score1 = cross_val_score(me2, x_select, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score_all.append(score1)
            elif shrink == 2:
                score2 = cross_val_score(me2, x_select2, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score_all.append(score2)
            else:
                score1 = cross_val_score(me2, x_select, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score2 = cross_val_score(me2, x_select2, y, scoring=scoring2, cv=cv2, n_jobs=1, verbose=0,
                                         fit_params=None).mean()
                score3 = max(score1, score2)
                score_all.append(score3)

            scores = cross_validate(me2, x_select2, y, scoring=scoring2, cv=cv2, return_train_score=False,
                                    return_estimator=True)
            me2 = scores['estimator'][0]
            estimator.append(me2)
        else:
            if isinstance(param_grid, (dict, list)):
                param_grid2 = param_grid
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'neg_mean_absolute_error'
                # scoring2 = 'r2'
            cv2 = LeaveOneOut()
            cv2 = 5
            if shrink == 1:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
                gd2.fit(x_select, y)
                score1 = gd2.best_score_
                score_all.append(score1)
                estimator.append(gd2.best_estimator_)
            elif shrink == 2:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)
                gd2.fit(x_select2, y)
                score2 = gd2.best_score_
                score_all.append(score2)
                estimator.append(gd2.best_estimator_)
            else:
                gd2 = GridSearchCV(me2, cv=cv2, param_grid=param_grid2, scoring=scoring2, n_jobs=1)

                gd2.fit(x_select, y)
                score1 = gd2.best_score_
                gd2.fit(x_select2, y)
                score2 = gd2.best_score_
                score_all.append(max(score1, score2))
                estimator.append(gd2.best_estimator_)

    if str_name is True:
        estimator = [str(estimator[i]).split("(")[0] for i in range(len(estimator))]

    #     if len(score_all) == 1:
    #         score_all = score_all[0]
    #         estimator = estimator[0]

    [print(i) for i in estimator]
    return score_all, estimator


def cv_predict(x, y, s_estimator, kf):
    y_test_predict_all = []
    for train, test in kf:
        X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
        s_estimator.fit(X_train, y_train)
        y_test_predict = s_estimator.predict(X_test)
        y_test_predict_all.append(y_test_predict)

    test_index = [i[1] for i in kf]
    y_test_true_all = [y[_] for _ in test_index]

    return y_test_true_all, y_test_predict_all


def pack_score(y_test_true_all, y_test_predict_all, scoring):
    if isinstance(y_test_true_all, np.ndarray) and isinstance(y_test_predict_all, np.ndarray):
        y_test_true_all = [y_test_true_all, ]
        y_test_predict_all = [y_test_predict_all, ]
    if scoring == "rmse":
        scoring2 = 'neg_mean_squared_error'
    else:
        scoring2 = scoring

    scorer = get_scorer(scoring2)

    scorer_func = scorer._score_func

    score = [scorer_func(i, j) for i, j in zip(y_test_true_all, y_test_predict_all)]

    if scoring == "rmse":
        score = np.sqrt(score)
    score_mean = np.mean(score)
    return score_mean


def my_score(gd_method, train_X, test_X, train_Y, test_Y):
    grid = gd_method
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)
    grid.cv = kf
    grid.fit(train_X, train_Y)
    metrics_method1 = "rmse"
    metrics_method2 = "r2"

    kf = KFold(n_splits=n_splits, shuffle=False)
    kf = list(kf.split(train_X))
    y_train_true_all, y_train_predict_all = cv_predict(train_X, train_Y, grid.best_estimator_, kf)

    cv_pre_train_y = np.concatenate([i.ravel() for i in y_train_predict_all]).T
    score1 = pack_score(y_train_true_all, y_train_predict_all, metrics_method1)
    score2 = pack_score(y_train_true_all, y_train_predict_all, metrics_method2)
    print("train_X's cv score %s" % score1, "train_X's cv score %s" % score2)


com_data_Pt = pd.read_excel('pt_re.xlsx', sheet_name="Pt")
com_data_Re = pd.read_excel('pt_re.xlsx', sheet_name="Re")

select_X = ['1dr2', '1dr6', '1dr12']
target_y = 'energy_eV'

calculate = "pt"

if calculate == "pt":
    com_data = com_data_Pt
    x = com_data[select_X].values
    y = com_data[target_y].values
    cut = 2637
    y = y + cut
else:
    com_data = com_data_Re
    x = com_data[select_X].values
    y = com_data[target_y].values
    cut = 3096
    y = y + cut

# x,y = utils.shuffle(x,y,random_state=2)

st = preprocessing.MinMaxScaler()
x = st.fit_transform(x)

method = ["SVR-set", "AdaBR-em", 'GBR-em', "LinearSVR-set"]

result = score_muti(x, y, me="reg", paras=True, method_name=method, shrink=1, str_name=False, param_grid=None)

from sklearn.model_selection import cross_val_predict

pre_y = cross_val_predict(
    SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.001, gamma='auto',
        kernel='rbf', max_iter=2000, shrinking=False, tol=0.001, verbose=False)
    , x, y, ) - cut
lin = LinearSVR(C=10, dual=True, epsilon=0.0, fit_intercept=True,
                intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
                random_state=3, tol=0.0001, verbose=0)

lin.fit(x, y)
pre_y2 = lin.predict(x) - cut

print(result[0])
print(result[1][-1].coef_)
print(result[1][-1].intercept_)

coef = lin.coef_
inter = lin.intercept_
data_max_ = st.data_max_
data_min_ = st.data_min_
data_range = st.data_range_

aa = coef / data_range
b = -cut + inter - sum(data_min_ * coef / data_range)

x_true = st.inverse_transform(x)
y2 = np.sum([i * j for i, j in zip(aa.T, x_true.T)], axis=0) + b
