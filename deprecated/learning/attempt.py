# -*- coding: utf-8 -*-

# @Time    : 2019/11/20 19:12
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause


def attempt_score(x_select, y, me="reg", paras=True, method_name=None, shrink=2, param_grid=None):
    """score with different method
    :param param_grid: user's param_grid

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
        # print(dict_method)

    x_select, y = utils.shuffle(x_select, y, random_state=1)
    x_select2 = preprocessing.scale(x_select)

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
                scoring2 = 'r2'
            cv2 = 3
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
        else:
            if isinstance(param_grid, (dict, list)):
                param_grid2 = param_grid
            if me == "clf":
                scoring2 = 'balanced_accuracy'
            if me == "reg":
                scoring2 = 'r2'
            cv2 = 3
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

    return score_all, estimator
