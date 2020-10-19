import pandas as pd
from sklearn.utils import shuffle
from sympy.physics.units import pa

from bgp.functions.dimfunc import Dim, dless
from bgp.skflow import SymbolLearning

if __name__ == "__main__":
    pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
    ###########第一个###########
    """数据"""
    com_data = pd.read_csv(r'reg1.csv')
    x = com_data.iloc[:, :-1].values
    y = com_data.iloc[:, -1].values
    x, y = shuffle(x, y, random_state=0)

    sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=True, dim_type=None, pop=1000,
                        gen=2, add_coef=True, re_hall=2,
                        random_state=1, n_jobs=10,
                        initial_max=2, max_value=3, store="reg1_result"
                        )
    sl.fit(x, y, x_dim=[dless, pa_dim, dless], y_dim=pa_dim, power_categories=(2, 3, 0.5, 0.33),
           categories=("Add", "Mul", "Sub", "Div"), )

    data = sl.loop.top_n(10)
    


# if __name__ == "__main__":
#     pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
#     ###########第二个###########
#     """数据"""
#     com_data = pd.read_csv(r'reg2.csv')
#     x = com_data.iloc[:, :-1].values
#     y = com_data.iloc[:, -1].values
#     """回归"""
#
#     sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=True, dim_type=None,
#                         pop=1000, gen=20, add_coef=True, re_hall=2,store="reg2_result",
#                         initial_max=3, max_value=5,
#                         random_state=0, n_jobs=20)
#     sl.fit(x, y, x_dim=[dless, pa_dim,dless,dless,dless,dless,], y_dim=pa_dim)
#     print(sl.expr)
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     pre_y = sl.predict(x)
#
#     r = np.corrcoef(np.vstack((pre_y, y)))[1,0]
#     error = np.mean(np.abs((y - pre_y) / y))
#
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     sl.loop.cpset.cv = 5
#     r2_cv = sl.cv_result(refit=False)
#     print("r:{},error:{},r2:{},MAE:{},r2_cv:{}".format(r,error,r2,mae,r2_cv[0]))

# if __name__ == "__main__":
#     pa_factor, pa_dim = Dim.convert_to(10 * 6 * pa)
#     ###########第二个###########
#     """数据"""
#     com_data = pd.read_csv(r'reg3.csv')
#
#     x = com_data.iloc[:, :-1].values
#     y = com_data.iloc[:, -1].values
#     x,y = shuffle(x, y)
#     """回归"""
#
#     sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=True, dim_type=None,
#                         pop=1000, gen=20, add_coef=True, re_hall=2, store="reg3_result",
#                         initial_max=3, max_value=6,
#                         random_state=1, n_jobs=20)
#     sl.fit(x, y, x_dim=[dless, pa_dim, dless, dless, dless, dless, dless, dless], y_dim=pa_dim,
#            power_categories=(2, 3, 0.5, 0.33),
#            categories=("Add", "Mul", "Sub", "Div"),
#            )
#     print(sl.expr)
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     pre_y = sl.predict(x)
#
#     r = np.corrcoef(np.vstack((pre_y, y)))[1, 0]
#     error = np.mean(np.abs((y - pre_y) / y))
#
#     r2 = sl.score(x, y, "r2")
#     mae = sl.score(x, y, "neg_mean_absolute_error")
#     sl.loop.cpset.cv = 5
#     r2_cv = sl.cv_result(refit=False)
#     print("r:{},error:{},r2:{},MAE:{},r2_cv:{}".format(r, error, r2, mae, r2_cv[0]))
