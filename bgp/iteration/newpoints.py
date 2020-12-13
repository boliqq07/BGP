# -*- coding: utf-8 -*-
"""find new point from searchspace to add to loop."""
import numpy as np
# @Time    : 2020/10/4 19:30
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from scipy import stats


def search_space(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


def get_max_std(grid_x, curves, n=1):
    new_xs = []
    new_ys = []
    stdd_list = []
    for curve in curves:
        stdd = np.std(curve, axis=0)
        meann = np.mean(curve, axis=0)
        top = np.argmax(stdd)

        new_x, new_y = grid_x[top], meann[top]
        new_xs.append(new_x)
        new_ys.append(new_y)
        stdd_list.append(stdd[top])

    stdd_list = np.array(stdd_list)
    stdd_list = stdd_list.argsort()
    top_n = list(stdd_list[:n])
    new_xs = np.array([new_xs[top_i] for top_i in top_n])
    new_ys = np.array([new_ys[top_i] for top_i in top_n])

    return new_xs, new_ys


def get_max_diff(grid_x, curves, n=1):
    meann = [np.mean(curve, axis=0) for curve in curves]
    meann = np.array(meann)
    stdd = np.std(meann, axis=0)
    temp = np.argpartition(-stdd, n)
    top_n = temp[:n]

    return grid_x[top_n, :], meann[:, top_n]


def new_points(loop, grid_x, method="get_max_std", resample_number=500, n=1):
    methods = {"get_max_std": get_max_std, "get_max_diff": get_max_diff}
    method = methods[method]
    self = loop.cpset
    data = loop.top_n(10)

    exprs = [self.compile_context(i) for i in data.iloc[:, 1]]
    pre_y_all_list = self.parallelize_try_add_coef_times(exprs, grid_x=grid_x, resample_number=resample_number)

    new_xs, new_ys = method(grid_x, pre_y_all_list, n=n)

    return new_xs, new_ys


def meanandstd(predict_dataj):
    mean = np.mean(predict_dataj, axis=1)
    std = np.std(predict_dataj, axis=1)
    data_predict = np.column_stack((mean, std))
    print(data_predict.shape)
    return data_predict


def CalculateEi(y, meanstd0):
    ego = (meanstd0[:, 0] - max(y)) / (meanstd0[:, 1])
    ei_ego = meanstd0[:, 1] * ego * stats.norm.cdf(ego) + meanstd0[:, 1] * stats.norm.pdf(ego)
    kg = (meanstd0[:, 0] - max(max(meanstd0[:, 0]), max(y))) / (meanstd0[:, 1])
    ei_kg = meanstd0[:, 1] * kg * stats.norm.cdf(kg) + meanstd0[:, 1] * stats.norm.pdf(kg)
    max_P = stats.norm.cdf(ego, loc=meanstd0[:, 0], scale=meanstd0[:, 1])
    ei = np.column_stack((meanstd0, ei_ego, ei_kg, max_P))
    print('ego is done')
    return ei
