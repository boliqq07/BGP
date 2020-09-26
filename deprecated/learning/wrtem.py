import numpy as np
import pandas as pd
from bgp.selection.quickmethod import method_pack
from mgetool.exports import Store
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# 数据导入

store = Store(r'/data/home/wangchangxin/data/wr/tem')

data = pd.read_excel(r'/data/home/wangchangxin/data/wr/tem/wrtem2.xlsx',
                     header=0, skiprows=None, index_col=0)

y = data["S"].values
x_p_name = ["t", 'v', 'hat']
x = data[x_p_name].values

# # # 预处理
# minmax = MinMaxScaler()
# x = minmax.fit_transform(x)
x_, y_ = shuffle(x, y, random_state=2)

# # # 建模
method_all = ['SVR-set', "GPR-set", "RFR-em", "AdaBR-em", "DTR-em", "LASSO-L1", "BRR-L1"]
methods = method_pack(method_all=method_all,
                      me="reg", gd=True)
pre_y = []
ests = []
for name, methodi in zip(method_all, methods):
    methodi.cv = 5
    methodi.scoring = "neg_root_mean_squared_error"
    gd = methodi.fit(X=x_, y=y_)
    score = gd.best_score_
    est = gd.best_estimator_
    print(name, "neg_root_mean_squared_error", score)
    score = cross_val_score(est, X=x_, y=y_, scoring="r2", ).mean()
    print(name, "r2", score)
    pre_yi = est.predict(x)
    pre_y.append(pre_yi)
    ests.append(est)
    store.to_pkl_pd(est, name)

pre_y.append(y)
pre_y = np.array(pre_y).T
pre_y = pd.DataFrame(pre_y)
pre_y.columns = method_all + ["realy_y"]
store.to_csv(pre_y, "wrtem_result")
