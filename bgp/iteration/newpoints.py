# -*- coding: utf-8 -*-

# @Time    : 2020/10/4 19:30
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import featurebox
from mgetool.exports import Store
from mgetool.imports import Call
from numpy import random
from scipy import stats
from sklearn.metrics import r2_score
import numpy as np
from bgp.skflow import SymbolLearning


def get_max_sd(grid_x, curves, n=1):
    stdd = np.std(curves, axis=0)
    meann = np.mean(curves, axis=0)
    top = np.argmax(stdd)
    return grid_x[top], meann[top]


def get_max_std(grid_x, curves, n=1):
    new_xs = []
    new_ys = []
    for i in curves:
        new_x, new_y = _get_max_stdi(grid_x, i, n=n)
        new_xs.append(new_x)
        new_ys.append(new_y)

    new_xs = np.array(new_xs)
    new_ys = np.array(new_xs)
    return new_xs, new_ys


def _get_max_stdi(grid_x, curve, n=1):
    stdd = np.std(curve, axis=0)
    meann = np.mean(curve, axis=0)
    top = np.argmax(stdd)
    return grid_x[top], meann[top]


def new_points(loop, grid_x, method="get_max_std"):

    methods = {"get_max_std": get_max_std,}
    method = methods[method]
    self = loop.cpset
    data = loop.top_n(10)

    exprs = [self.compile_context(i) for i in data.iloc[:, 1]]
    pre_y_all_list = self.parallelize_try_add_coef_times(exprs, grid_x=grid_x)

    new_xs, new_ys = method(grid_x, pre_y_all_list)

    return new_xs, new_ys


def search_space(*arg):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes


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

