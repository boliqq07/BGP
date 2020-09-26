# -*- coding: utf-8 -*-

# @Time    : 2020/1/3 18:52
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def count_cof(cof, threshold):
    """check cof and count the number"""

    if cof is None:
        raise NotImplemented("imported cof is None")
    list_count = []
    list_coef = []
    cof = np.nan_to_num(cof)
    g = np.where(abs(cof) >= threshold)
    for i in range(cof.shape[0]):
        e = np.where(g[0] == i)
        com = list(g[1][e])
        # ele_ratio.remove(i)
        list_count.append(com)
        list_coef.append([cof[i, j] for j in com])
    return list_coef, list_count


def name_to_name(*iters, search=None, search_which=1, return_which=(1,), two_layer=False):
    from collections import Iterable
    if isinstance(return_which, int):
        return_which = tuple([return_which, ])
    if two_layer:

        results_all = []
        if isinstance(search, Iterable):
            for index_i in search:
                results_all.append(
                    name_to_name(*iters, search=index_i, search_which=search_which,
                                 return_which=return_which, two_layer=False))

            if len(return_which) >= 2:
                return list(zip(*results_all))
            else:
                return results_all
        else:
            raise IndexError("search_name or search should be iterable")

    else:
        zeros = [list(range(len(iters[0])))]
        zeros.extend([list(_) for _ in iters])
        iters = zeros
        zips = list(zip(*iters))

        if isinstance(search, Iterable):
            search_index = [iters[search_which].index(i) for i in search]
            results = [zips[i] for i in search_index]
        else:
            raise IndexError("search_name or search should be iterable")

        res = list(zip(*results))
        if not res:
            return_res = [[] for _ in return_which]
        else:
            return_res = [res[_] for _ in return_which]
        if len(return_which) == 1:
            return_res = return_res[0]
        return return_res


os.chdir(r'C:\Users\Administrator\Desktop')
data1 = pd.read_excel('元素周期表性质11111.xlsx')  # 数据名称
data1 = data1.astype("float")
re = data1.corr()

# fig, corrolasdx = plt.subplots(figsize = (15,15))  #填入不同特征个数
size = list(range(len(re.columns.values)))

sns.heatmap(re, cmap="seismic", square=True, annot=False, xticklabels=size, yticklabels=size)
# corrolasdx.set_ylabel('features', fontsize = 15)
# corrolasdx.set_xlabel('features', fontsize = 15)
# plt.savefig('./out.png')
plt.show()

list_coef, list_count = count_cof(re, 0.85)  # 分数组，序号组
list_name = name_to_name(size, re.columns.values, search=list_count, search_which=0, return_which=(2,),
                         two_layer=True)  # 名字组
group = list(zip(list_count, list_coef, list_name))
