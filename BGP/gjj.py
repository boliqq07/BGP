# -*- coding: utf-8 -*-

# @Time    : 2020/8/20 14:37
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause

import numpy as np
import pandas as pd

from BGP.base import SymbolSet
from BGP.functions.dimfunc import Dim, dless
from BGP.skflow import SymbolLearning


def check_input_and_get_dim(in_table):
    """判断除去表头第二行是否可以转为列表,若可以，取出量纲，否则量纲为None"""
    table = in_table.iloc[0, :].values
    try:
        if all([isinstance(i, (int, float, np.int64, np.float64)) for i in table]):
            y_x_dim = None
        elif all([isinstance(i, str) for i in table]):
            y_x_dim = [eval(i) for i in table]
            in_table = in_table.iloc[1:, :]
        else:
            raise TypeError
    except BaseException as e:
        print(e)
        raise TypeError("If the member of second row must be float, int, or list like: [1,0,0,0,0,0,0]. "
                        "Please check the type of each member carefully.")
    return in_table, y_x_dim


if __name__ == "__main__":

    """表格"""

    # input_path = r"C:\Users\Administrator\Desktop\data_input.csv"  # (输入) 测试输入文件1，绝对路径,只接受csv
    input_path = r"C:\Users\Administrator\Desktop\data_input_2.csv"  # (输入) 测试输入文件2（含有量纲），绝对路径,只接受csv

    input_table = pd.read_csv(input_path, header=0, skiprows=None, index_col=0)
    yx_table, yx_dim = check_input_and_get_dim(input_table)
    y = yx_table.values[:, 0]
    x = yx_table.values[:, 1:]
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    if yx_dim is None:
        y_dim = 1
        x_dim = 1
    else:
        y_dim = yx_dim[0]
        x_dim = yx_dim[1:]

    # Features
    c = [2, ]  # （输入）
    y_dim = [1, 2, 3, 4, 5, 6, 7]  # (输入，窗口按键方式，或者手动写入，若使用，覆盖掉从表格中输入的的量纲)
    x_dim = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7],
             [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7],
             [1, 2, 3, 4, 5, 6, 7]]  # (输入，窗口按键方式,或者手动写入，若使用，覆盖掉表格中的量纲)
    c_dim = [[1, 2, 3, 4, 5, 6, 7]]  # （输入，窗口按键方式，或者手动写入）
    x_prob = None  # （输入）
    c_prob = None  # （输入）
    x_group = None  # （输入）

    if c_dim is None:
        c_dim = 1

    if isinstance(y_dim, (list, tuple)):
        y_dim = Dim(y_dim) if isinstance(y_dim, (list, tuple)) else dless
    if isinstance(x_dim, (list, tuple)):
        x_dim = [Dim(i) if isinstance(i, (list, tuple)) else dless for i in x_dim]
    if isinstance(c_dim, (list, tuple)):
        c_dim = [Dim(i) if isinstance(i, (list, tuple)) else dless for i in c_dim]

    # Operators
    power_categories = (0.5, 1, 2)  # （输入，必须全部为数字）
    categories = ("Add", 'Sub', 'Mul', 'Div', 'exp', 'ln', 'Abs', "Neg", "Rec", "Self")  # （输入，多选列表）
    power_categories_prob = "balance"  # （输入）
    categories_prob = "balance"  # （输入）
    special_prob = None  # （输入）

    feature_name = yx_table.columns.values[1:]
    sample_name = yx_table.index.values

    pset = SymbolSet()
    pset.add_features_and_constants(X=x,
                                    y=y,
                                    c=c,
                                    x_dim=x_dim,
                                    y_dim=y_dim,
                                    c_dim=c_dim,
                                    x_prob=x_prob,
                                    c_prob=c_prob,
                                    x_group=x_group
                                    )
    pset.add_operations(power_categories=power_categories,
                        categories=categories,
                        power_categories_prob=power_categories_prob,
                        categories_prob=categories_prob,
                        special_prob=special_prob)

    # 预处理部分参数
    stats_keys = ("fitness_dim_max",)  # （输入,控制具体每代打印内容）
    stats_values = ("max",)  # （输入）
    stats = {i: j for j in stats_values for i in stats_keys}

    score_pen = (1,)  # （输入）
    stop = None  # （输入）
    if isinstance(stop, float):
        if score_pen[0] >= 0:
            stop_condition = lambda ind: ind.fitness.values[0] >= stop
        else:
            stop_condition = lambda ind: ind.fitness.values[0] <= stop
    else:
        stop_condition = None

    sl = SymbolLearning(
        # Loop
        loop='BaseLoop',  # （输入）
        pop=500,  # （输入）
        gen=20,  # （输入）
        mutate_prob=0.5,  # （输入）
        mate_prob=0.8,  # （输入）
        migrate_prob=0,  # （输入）
        hall=1,  # （输入）
        re_hall=None,  # （输入）
        re_Tree=None,  # （输入）
        scoring=("r2",),  # （输入）
        score_pen=(1,),  # （输入,上面出现过一次，同名称为同一内容）
        stop_condition=stop_condition,
        cv=1,  # （输入）

        # Limitation
        initial_max=3,  # （输入）
        initial_min=None,  # （输入）
        max_value=5,  # （输入）

        add_coef=True,  # （输入）
        inter_add=True,  # （输入）
        out_add=False,  # （输入）
        vector_add=False,  # （输入）
        inner_add=False,  # （输入）
        flat_add=False,  # （输入）

        dim_type=None,  # （输入）
        cal_dim=False,  # （输入）

        # Running Control
        filter_warning=True,  # （输入）
        n_jobs=1,  # （输入）
        batch_size=40,  # （输入）
        random_state=None,  # （输入）

        stats=stats,
        verbose=True,  # （输出控制，打印每代结果）
        tq=True,  # （输出控制，打印进度条）
        store=False,  # （输出控制，保存文件，默认当前路径，应当输入绝对路径，更改保存位置）
    )
    sl.fit(pset=pset)
    score = sl.score(x, y, scoring="r2")  # (页面展示)
    print(sl.expr)  # (页面展示)
