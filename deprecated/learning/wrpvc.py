import pandas as pd
from BGP.selection.quickmethod import method_pack
from mgetool.exports import Store
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# 数据导入

store = Store(r'C:\Users\Administrator\Desktop/wr')

# """for element site"""
data = pd.read_excel(r'C:\Users\Administrator\Desktop/wr/wrpvc.xlsx',
                     header=0, skiprows=None, index_col=0)

y = data["t"].values
x_p_name = ['t_t', 'v', 'b', 'hat', 'd', 't1']
x = data[x_p_name].values

x_name = ["温度刻度", "速度刻度", "风量", "加盖", "焊口距地距离", "焊接前出风口温度"]

# # # 预处理
minmax = MinMaxScaler()
x = minmax.fit_transform(x)

x, y = shuffle(x, y)
# m_corr = Corr(threshold=0.85, muti_grade=None, muti_index=None, must_index=None)
# m_corr.fit(x)
# corr = m_corr.cov

#
# corr_plot(corr, x_name=x_p_name, left_down="pie", right_top="text", threshold_left=0, threshold_right=0,
#           title="pearsonr coefficient", label_axis="off", front_raito=1)
# # 机器挑选
# select = m_corr.remove_coef(result_list[1])
# x = x[:,select]
# # 人工挑选？
# # select=[]

# 预筛选模型
method_all = ['KNR-set', 'SVR-set', "KRR-set", "GPR-set",
              "RFR-em", "AdaBR-em", "DTR-em",
              "LASSO-L1", "BRR-L1", "SGDR-L1", "PAR-L1"]
methods = method_pack(method_all=method_all,
                      me="reg", gd=False)

for name, methodi in zip(method_all, methods):
    methodi.keywords["cv"] = 3
    # methodi.keywords["scoring"] = 'neg_mean_absolute_error'
    methodi.keywords["scoring"] = 'r2'
    score = methodi(X=x, y=y).mean()
    print(name, score)

# method_select=["RFR-em", "AdaBR-em", "BRR-L1"]
#
# me1 = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                             min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
#                             min_impurity_split=None, bootstrap=True, oob_score=False,
#                             random_state=None, verbose=0, warm_start=False)
# param_grid1 = [{'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 3]}]
#
# me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
#                     copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
#                     n_iter=300, normalize=False, tol=0.01, verbose=False)
# param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05], 'alpha_2': [1e-07, 1e-06, 1e-05]}]
#
#
# dt = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=5, min_samples_split=4)
# me3 = AdaBoostRegressor(dt, n_estimators=200, learning_rate=0.05, loss='linear', random_state=0)
# param_grid3 = [{'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}]
#
# #2 model
# ref = RFECV(me2,cv=3)
# x_ = ref.fit_transform(x,y)
# gd = GridSearchCV(me2, cv=3, param_grid=param_grid2, scoring="r2", n_jobs=1)
# gd.fit(x_,y)
# score = gd.best_score_
#
# #1,3 model
# # gd = GridSearchCV(me1, cv=3, param_grid=param_grid1, scoring="r2", n_jobs=1)
# # gd.fit(x,y)
# # es = gd.best_estimator_
# # sf = SelectFromModel(es, threshold=None, prefit=False,
# #                  norm_order=1, max_features=None)
# # sf.fit(x,y)
# # feature = sf.get_support()
# #
# # gd.fit(x[:,feature],y)
# # score = gd.best_score_
#
# # 其他模型
# # 穷举等...
#
# #导出
# # pd.to_pickle(gd,r'C:\Users\Administrator\Desktop\skk\gd_model')
# # pd.read_pickle(r'C:\Users\Administrator\Desktop\skk\gd_model')
# store.to_pkl_sk(gd)
# store.to_csv(x)
# store.to_txt(score)
