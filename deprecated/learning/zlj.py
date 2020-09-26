import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from BGP.selection.backforward import BackForward
from mgetool.exports import Store
from mgetool.imports import Call
from sklearn import svm
from sklearn.model_selection import GridSearchCV, LeaveOneOut

warnings.filterwarnings("ignore")

# 数据导入
store = Store(r'/data/home/wangchangxin/data/zlj/')
data = Call(r'/data/home/wangchangxin/data/zlj/', index_col=None)
all_import = data.xlsx().data

x_name = all_import.index.values
y = all_import["y"].values
x_frame = all_import.drop("y", axis=1)
x = x_frame.values
# # 预处理
# minmax = MinMaxScaler()
# x = minmax.fit_transform(x)
# 数据划分
xtrain, xtest = x[3:], x[:3]
ytrain, ytest = y[3:], y[:3]

xtrain, ytrain = sklearn.utils.shuffle(xtrain, ytrain, random_state=3)

# x = minmax.inverse_transform(x_new)

# # 网格搜索*前进后退


me4 = svm.SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=True, max_iter=2000)
# 网格
param_grid4 = [{'C': [10000000, 100000000], "epsilon": [1000, 0.1, 0.0001]}]
gd = GridSearchCV(me4, cv=LeaveOneOut(), param_grid=param_grid4, scoring='neg_root_mean_squared_error', n_jobs=1)
# 前进后退
ba = BackForward(gd, n_type_feature_to_select=3, primary_feature=None, muti_grade=2, muti_index=None,
                 must_index=None, tolerant=0.01, verbose=0, random_state=2)
# x_add = np.concatenate((x, xtest), axis=0)
# y_add = np.concatenate((y, ytest), axis=0)
x_add = xtrain
y_add = ytrain
# running!
ba.fit(x_add, y_add)
xtest = xtest[:, ba.support_]
xtrain = xtrain[:, ba.support_]

# 预测
scoretest = ba.estimator_.score(xtest, ytest)
scoretrain = ba.estimator_.score(xtrain, ytrain)
y_pre_test = ba.estimator_.predict(xtest)
y_pre_train = ba.estimator_.predict(xtrain)
#
# # 训练#
cor_ = abs(y_pre_train - ytrain) / ytrain
cors_ = cor_.mean()
# 测试
#
cor = abs(y_pre_test - ytest) / ytest
cors = cor.mean()
# # 合并
y_ = ba.estimator_.predict(x[:, ba.support_])


#
# # 画图
#
def scatter(y_true, y_predict, strx='y_true', stry='y_predict'):
    x, y = y_true, y_predict
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, marker='o', s=50, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
    ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
    plt.xlabel(strx)
    plt.ylabel(stry)
    plt.show()


scatter(ytest, y_pre_test, strx='y_true($10^4$T)', stry='y_predict($10^4$T)')
scatter(ytrain, y_pre_train, strx='y_true($10^4$T)', stry='y_predict($10^4$T)')


def scatter2(x, y_true, y_predict, strx='y_true', stry1='y_true(GWh)', stry2='y_predict', stry="y"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1 = ax.scatter(x, y_true, marker='o', s=50, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
    ax.plot(x, y_true, '-', ms=5, lw=2, alpha=0.7, color='black')
    l2 = ax.scatter(x, y_predict, marker='^', s=50, alpha=0.7, c='green', linewidths=None, edgecolors='blue')
    ax.plot(x, y_predict, '-', ms=5, lw=2, alpha=0.7, color='green')
    # ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
    plt.xlabel(strx)
    plt.legend((l1, l2),
               (stry1, stry2),
               loc='upper left')
    plt.ylabel(stry)
    plt.show()


a = np.arange(2000, 2020)

scatter2(a, y[::-1], y_[::-1], strx='year', stry="y($10^4$T)", stry1='y_true($10^4$T)', stry2='y_predict($10^4$T)')

# #导出
print(x_frame.iloc[:, :].columns.values[ba.support_])
store.to_pkl_sk(ba.estimator_, "model")
all_import["y_predict"] = y_
store.to_csv(all_import, "predict")
