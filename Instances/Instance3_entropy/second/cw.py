import numpy as np
import pandas as pd
from mgetool.exports import Store
from sklearn.utils import shuffle
from sympy.physics.units import pa

from bgp.functions.dimfunc import Dim, dless
from bgp.skflow import SymbolLearning


def error(y, pre_y):
    return np.mean(np.abs((y - pre_y) / y))


# if __name__ == "__main__":
#     pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
#     ###########第一个###########
#     """数据"""
#     com_data = pd.read_csv(r'BCC.csv')
#     x = com_data.iloc[:, :-1].values
#     y = com_data.iloc[:, -1].values
#     x, y = shuffle(x, y, random_state=0)
#
#     st = Store("BCC_result_error_no_intercept")
#     st.start()
#     sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=True, dim_type=pa_dim, pop=5000,
#                         gen=50, add_coef=True, re_hall=2,
#                         inter_add=False,
#                         random_state=1, n_jobs=12,
#                         initial_max=2, max_value=4, store=True,
#                         stats={"fitness_dim_max": ("max",)}
#                         )
#     sl.fit(x, y,x_dim=[dless,dless,pa_dim,dless,dless,dless,dless], y_dim=pa_dim,  power_categories=(2, 3, 0.5, 0.33),
#            categories=("Add", "Mul", "Sub", "Div"), )
#
#     print(sl.expr)
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     pre_y = sl.predict(x)
#
#     r = np.corrcoef(np.vstack((pre_y, y)))[1, 0]
#
#     error = np.mean(np.abs((y - pre_y) / y))
#
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     sl.loop.cpset.cv = 5
#     r2_cv = sl.cv_result(refit=False)
#     print("r:{},error:{},r2:{},MAE:{},r2_cv:{}".format(r, error, r2, mae, r2_cv[0]))
#
#     data = sl.loop.top_n(20, ascending=False)
#     st.end()
#     st.to_csv(data, file_new_name="top_n")

if __name__ == "__main__":
    pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
    ###########第一个###########
    """数据"""
    com_data = pd.read_csv(r'FCC.csv')
    x = com_data.iloc[:, :-1].values
    y = com_data.iloc[:, -1].values
    x, y = shuffle(x, y, random_state=0)

    st = Store("FCC_result_error_no_intercept")
    st.start()
    sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=False, dim_type=pa_dim, pop=5000,
                        gen=50, add_coef=True, re_hall=2,
                        inter_add=False,
                        random_state=2, n_jobs=16,
                        initial_max=2, max_value=4, store=True,
                        stats={"fitness_dim_max": ("max",)}
                        )
    sl.fit(x, y, x_dim=[pa_dim, dless, dless, dless, dless, dless, dless], y_dim=pa_dim,
           power_categories=(2, 3, 0.5, 0.33),
           categories=("Add", "Mul", "Sub", "Div"), )

    print(sl.expr)
    r2 = sl.score(x, y, "r2")
    mae = sl.score(x, y, "neg_mean_absolute_error")
    pre_y = sl.predict(x)

    r = np.corrcoef(np.vstack((pre_y, y)))[1, 0]

    error = np.mean(np.abs((y - pre_y) / y))

    r2 = sl.score(x, y, "r2")
    mae = sl.score(x, y, "neg_mean_absolute_error")
    sl.loop.cpset.cv = 5
    r2_cv = sl.cv_result(refit=False)
    print("r:{},error:{},r2:{},MAE:{},r2_cv:{}".format(r, error, r2, mae, r2_cv[0]))

    data = sl.loop.top_n(20, ascending=False)
    st.end()
    st.to_csv(data, file_new_name="top_n")

# if __name__ == "__main__":
#     pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
#     ###########第一个###########
#     """数据"""
#     com_data = pd.read_csv(r'FCC-BCC.csv')
#     x = com_data.iloc[:, :-1].values
#     y = com_data.iloc[:, -1].values
#     x, y = shuffle(x, y, random_state=0)
#
#     st = Store("FCC-BCC_result_error_no_intercept")
#     st.start()
#     sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=True,  pop=5000, dim_type = pa_dim,
#                         gen=50, add_coef=True, re_hall=2,
#                         inter_add=False,batch_size=50,
#                         random_state=1, n_jobs=16,
#                         initial_max=2, max_value=4, store=True,
#                         stats={"fitness_dim_max": ("max",)}
#                         )
#     sl.fit(x, y,x_dim=[dless,pa_dim,dless,dless,dless,dless,dless,dless], y_dim=pa_dim, power_categories=(2, 3, 0.5, 0.33),
#            categories=("Add", "Mul", "Sub", "Div"), )
#
#     print(sl.expr)
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     pre_y = sl.predict(x)
#
#     r = np.corrcoef(np.vstack((pre_y, y)))[1, 0]
#
#     error = np.mean(np.abs((y - pre_y) / y))
#
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     sl.loop.cpset.cv = 5
#     r2_cv = sl.cv_result(refit=False)
#     print("r:{},error:{},r2:{},MAE:{},r2_cv:{}".format(r, error, r2, mae, r2_cv[0]))
#
#     data = sl.loop.top_n(20, ascending=False)
#     st.end()
#     st.to_csv(data, file_new_name="top_n")
