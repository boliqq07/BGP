# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:15:22 2020

@author: WXQ
"""

import os
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm


def change_cwd(file):
    driver, name = os.path.split(file)
    os.chdir(driver)


def parallelize(n_jobs, func, iterable, respective=False, tq=True, batch_size='auto', **kwargs):
    """
    parallelize the function for iterable.

    make sure in if __name__ == "__main__":

    Parameters
    ----------
    batch_size:int,str
    respective:bool
        Import the parameters respectively or as a whole
    tq:bool
         View Progress or not
    n_jobs:int
        cpu numbers. n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of CPU cores on the worker host(s).
    func:
        function to calculate
    iterable:
        interable object
    kwargs:
        kwargs for function

    Returns
    -------
    results
        function results

    """

    func = partial(func, **kwargs)
    if effective_n_jobs(n_jobs) == 1:
        parallel, func = list, func
    else:
        parallel = Parallel(n_jobs=n_jobs, batch_size=batch_size)
        func = delayed(func)
    if tq:
        if respective:
            return parallel(func(*iter_i) for iter_i in tqdm(iterable))
        else:
            return parallel(func(iter_i) for iter_i in tqdm(iterable))
    else:
        if respective:
            return parallel(func(*iter_i) for iter_i in iterable)
        else:
            return parallel(func(iter_i) for iter_i in iterable)


def distance(req, sha1, penalty=None):
    """match top n"""
    sha = np.copy(sha1)

    terms = []
    if penalty is None:
        penalty = [1.0] * req.shape[0]
    assert len(penalty) == req.shape[0]

    def distance_out_points(term, sha, points_set=None):
        if points_set is None:
            pass
        else:
            points_set = list(points_set)
            sha[points_set] = np.inf

        sh = sha - term[-1]
        dis = np.sqrt(np.sum(sh ** 2, axis=1)) * penalty[term[1]]
        index = np.argmin(dis)
        dis_min = np.min(dis)
        return index, term[1], dis_min, term[-1]

    for l, k in enumerate(req):
        term = distance_out_points((0, l, 0, k), sha, points_set=None)
        terms.append(term)

    terms = np.array(terms)

    # points = list(terms[:, 0])
    # num = [points.count(i) for i in points]

    # while max(num) > 1:
    #
    #     changes = [(np.where(terms[:, 0] == k))[0] for i, k in zip(num, points) if i > 1]
    #     changes = set([tuple(i) for i in changes if len(i) > 0])
    #
    #     new_change = []
    #     for i in changes:
    #         data = terms[i, 2]
    #         reserve = int(np.argmin(data))
    #         i = list(i)
    #         i.pop(reserve)
    #         new_change.append(i)
    #
    #     points_set = set(points)
    #     for i in chain(*new_change):
    #         term = terms[i]
    #         term = distance_out_points(term, sha, points_set=points_set)
    #         terms[i] = term
    #
    #     points = list(terms[:, 0])
    #     num = [points.count(i) for i in points]

    points = list(terms[:, 0])
    dis_average = np.mean(terms[:, 2])

    return points, dis_average, req


def move(datas, x_, x, y_, y, mv=0.01):
    for m, j in tqdm(list(product(np.arange(x_, x, mv), np.arange(y_, y, mv))), desc="平移完成度", mininterval=1,
                     unit="个"):
        yield np.column_stack((datas[:, 0] + j, datas[:, 1] + m))


def move2(datas, x_, x, y_, y, mv=0.01):
    return [np.column_stack((datas[:, 0] + j, datas[:, 1] + m)) for m, j in
            list(product(np.arange(x_, x, mv), np.arange(y_, y, mv)))]


def range_scale(req, sha):
    """推荐范围"""
    a = np.round(np.min(req, axis=0) - np.max(sha, axis=0), 1)
    b = np.round(np.max(req, axis=0) - np.min(sha, axis=0), 1)
    print("最大范围x: [{},{}], y: [{},{}]".format(a[0], b[0], a[1], b[1]))

    a = np.round(np.median(req, axis=0) - np.max(sha, axis=0) - 1.5, 1)
    b = np.round(np.median(req, axis=0) - np.min(sha, axis=0) + 1.5, 1)
    print("推荐范围x: [{},{}], y: [{},{}](可调整)".format(a[0], b[0], a[1], b[1]))
    return a[0], b[0], a[1], b[1]


def calculate(required, shape, x_=None, x=None, y_=None, y=None, mv=0.01, top_n=100, penalty=None, n_jobs=4):
    """
    Parameters
    ----------
    top_n:前100个
    mv: 移动步长
    shape：匹配数据
    required : 移动数据
    x_,x,y_,y: 移动范围
    penalty: None or list, 惩罚值

    Returns
    ----------
    距离，选出的shape点，移动后的req点
    """
    rang = range_scale(shape, required)
    r = []
    for i, j in zip(rang, (x_, x, y_, y)):
        if j is None:
            r.append(i)
        else:
            r.append(j)
    print("使用范围x: [{},{}], y: [{},{}]".format(*r))

    res = parallelize(n_jobs, distance, move2(required, *r, mv=mv), respective=False, tq=True, batch_size="auto",
                      sha1=shape,
                      penalty=penalty)

    res = np.array(res)

    ind = res[:, 1].argsort()

    res = res[ind]

    res = res[:top_n]

    return res

    ##################以上的都不要动####################


if __name__ == "__main__":
    file = r"C:\Users\Administrator\Desktop\wxq\ky_data.xlsx"  # 路径

    change_cwd(file)  # 调整工作路径
    data = pd.read_excel(file)

    shape = data[["shape_X", "shape_y"]].values
    required = data[["required_X", "required_y"]]
    name = data["name"].values

    required = required.dropna()  # 删除nan
    required = required.values  # 取出数值

    res = calculate(required, shape, x_=-4, x=3, y_=0.5, y=4, mv=0.1, top_n=100, penalty=None, n_jobs=4)

    index = res[:, 0]

    sort_names = np.array([str(name[np.array(i)]) for i in index])

    axiss = np.array([str(shape[np.array(i)]) for i in index])
    res = np.concatenate(((sort_names.reshape(-1, 1)), res, axiss.reshape(-1, 1),), axis=1)
    a = ["name_in_shape", "index_in_shape", "distance", "new_required", "site_in_shape", ]

    ##############################################################
    print("保存至 %s" % os.getcwd())
    pd.DataFrame(res).to_csv("res.csv")  # 改文件名

    ##画图###################
    import matplotlib.pyplot as plt

    data = res[0, :]
    plt.scatter(shape[:, 0], shape[:, 1], marker="^")
    # plt.scatter(required[:, 0], required[:, 1], c="b", marker="o")
    plt.scatter(data[-2][:, 0], data[-2][:, 1], c="g", marker="o")
    plt.scatter(shape[data[1]][:, 0], shape[data[1]][:, 1], c="r", marker="^")
    plt.show()
