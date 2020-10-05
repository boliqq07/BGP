# -*- coding: utf-8 -*-

# @Time    : 2020/10/3 15:00
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
from sklearn.utils._random import check_random_state


def init(m_sample=100, n_feature=10):
    n = n_feature
    m = m_sample
    mean = [1] * n
    cov = np.zeros((n, n))
    for i in range(cov.shape[1]):
        cov[i, i] = 1
    rdd = check_random_state(1)
    X = rdd.multivariate_normal(mean, cov, m)
    return X


def add_noise(s, ratio):
    print(s.shape)
    rdd = check_random_state(1)
    return s + rdd.random_sample(s.shape) * np.max(s) * ratio
