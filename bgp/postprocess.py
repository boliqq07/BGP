# -*- coding: utf-8 -*-

# @Time    : 2020/10/19 13:32
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import pandas as pd


def top_n(loop, n=10, gen=-1, key="value"):
    data = loop.data_all
    data = pd.DataFrame(data)
    if gen == -1:
        gen = max(data["gen"])

    data = data[data["gen"] == gen]

    data = data.drop_duplicates(['expr'], keep="first")

    if key is not None:
        data[key] = data[key].str.replace("(", "")
        data[key] = data[key].str.replace(")", "")
        data[key] = data[key].str.replace(",", "")
        try:
            data[key] = data[key].astype(float)
        except ValueError:
            raise TypeError("check this key column can be translated into float")

        data = data.sort_values(by='value', ascending=False).iloc[:n, :]

    return data
