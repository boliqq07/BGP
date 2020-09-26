# -*- coding: utf-8 -*-

# @Time   : 2019/6/8 21:35
# @Author : Administrator
# @Project : feature_preparation
# @FileName: 4.symbollearing.py
# @Software: PyCharm

"""

"""

import numpy as np
import pandas as pd
import sympy
from bgp.combination.symbolbase import calculateExpr, getName

from mgetool.exports import Store
from mgetool.imports import Call

if __name__ == "__main__":
    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\4.symbollearning')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\2.correction_analysis')

    all_import_structure = data.csv.all_import_structure
    data_import = all_import_structure
    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data221_import = data_import.iloc[np.where(data_import['group_number'] == 221)[0]]
    data216_225_221import = pd.concat((data216_import, data225_import, data221_import))

    list_name = data.csv.list_name
    list_name = list_name.values.tolist()
    list_name = [[i for i in _ if isinstance(i, str)] for _ in list_name]
    # grid = itertools.product(list_name[2],list_name[12],list_name[32])

    select = ['volume', 'radii covalent', 'electronegativity(martynov&batsanov)', 'electron number']

    select = ['volume'] + [j + "_%i" % i for j in select[1:] for i in range(2)]

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    name, rep_name = getName(X_frame)
    x0, x1, x2, x3, x4, x5, x6 = rep_name
    expr01 = sympy.log(1 / (x1 + x2) * x0 / (x5 + x6) * x4 / x3)

    results = calculateExpr(expr01, pset=None, x=X, y=y, score_method=r2_score, add_coeff=True,
                            del_no_important=False, filter_warning=True, terminals=rep_name,
                            inter_add=True, iner_add=False, random_add=False)
    print(select)
    print(results)

    store.to_csv(data216_225_221import, "plot221225216")
