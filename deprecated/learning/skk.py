import pandas as pd
from bgp.featurizers.compositionfeaturizer import WeightedAverage
from bgp.selection.corr import Corr
from mgetool.exports import Store
from mgetool.imports import Call
from pymatgen import Composition
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

# 数据导入
store = Store(r'C:\Users\Administrator\Desktop\skk')
data = Call(r'C:\Users\Administrator\Desktop\skk')
all_import = data.csv().skk

# """for element site"""
element_table = pd.read_excel(r'C:\Users\Administrator\Desktop\band_gap_exp\element_table.xlsx',
                              header=4, skiprows=0, index_col=0)
element_table = element_table.iloc[5:, 7:]

# 其他数据获取
feature_select = [
    'lattice constants a',
    'lattice constants b',
    'lattice constants c',
    'radii atomic(empirical)',
    'radii ionic(pauling)',
    'radii covalent',
    'radii metal(waber)',
    'distance valence electron(schubert)',
    'distance core electron(schubert)',
    'radii pseudo-potential(zunger)',

    'energy ionization first',
    'energy ionization second',
    'energy ionization third',
    'enthalpy atomization',
    'enthalpy vaporization',
    'latent heat of fusion',

    'energy cohesive brewer',
    'total energy',

    'electron number',
    'valence electron number',
    'charge nuclear effective(slater)',
    'periodic number',
    'electronegativity(martynov&batsanov)',
    "modulus compression",
    'volume atomic(villars,daams)',

]

select_element_table = element_table[feature_select]

"""transform composition to pymatgen Composition"""
x_name = all_import.index.values
y = all_import["site1"].values

x_name = pd.Series(map(Composition, x_name))

wei = WeightedAverage(elem_data=select_element_table, n_jobs=1, on_errors='raise', return_type='any')
x_DF = wei.fit_transform(x_name)
x = x_DF.values
# if_nan = np.where(np.isnan(x))

# 预处理
minmax = MinMaxScaler()
x = minmax.fit_transform(x)
# x = minmax.inverse_transform(x_new)

# corr = np.corrcoef(x)
# corr2 = DataFrame(x).corr()
m_corr = Corr(threshold=0.85, muti_grade=2, muti_index=None, must_index=None)
m_corr.fit(x)
corr = m_corr.cov
result_list = m_corr.count_cof(cof=None)

# 机器挑选
select = m_corr.remove_coef(result_list[1])
x = x[:, select]
# 人工挑选？
# select=[]

# #预筛选模型
# method_all=['KNR-set', 'SVR-set', "KRR-set", "GPR-set",
#                                   "RFR-em", "AdaBR-em", "DTR-em",
#                                   "LASSO-L1", "BRR-L1", "SGDR-L1", "PAR-L1"]
# methods = method_pack(method_all=method_all,
#                       me="reg", gd=False)
# # me="reg", gd=True)
# for name,methodi in zip(method_all, methods):
#     methodi.keywords["cv"]= 3
#     methodi.keywords["scoring"] = 'neg_mean_absolute_error'
#     score = methodi(X=x, y=y).mean()
#     print(name,score)

method_select = ["RFR-em", "AdaBR-em", "BRR-L1"]

me1 = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, bootstrap=True, oob_score=False,
                            random_state=None, verbose=0, warm_start=False)
param_grid1 = [{'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 3]}]

me2 = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                    copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                    n_iter=300, normalize=False, tol=0.01, verbose=False)
param_grid2 = [{'alpha_1': [1e-07, 1e-06, 1e-05], 'alpha_2': [1e-07, 1e-06, 1e-05]}]

dt = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=5, min_samples_split=4)
me3 = AdaBoostRegressor(dt, n_estimators=200, learning_rate=0.05, loss='linear', random_state=0)
param_grid3 = [{'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05]}]

# 2 model
ref = RFECV(me2, cv=3)
x_ = ref.fit_transform(x, y)
gd = GridSearchCV(me2, cv=3, param_grid=param_grid2, scoring="r2", n_jobs=1)
gd.fit(x_, y)
score = gd.best_score_

# 1,3 model
# gd = GridSearchCV(me1, cv=3, param_grid=param_grid1, scoring="r2", n_jobs=1)
# gd.fit(x,y)
# es = gd.best_estimator_
# sf = SelectFromModel(es, threshold=None, prefit=False,
#                  norm_order=1, max_features=None)
# sf.fit(x,y)
# feature = sf.get_support()
#
# gd.fit(x[:,feature],y)
# score = gd.best_score_

# 其他模型
# 穷举等...

# 导出
# pd.to_pickle(gd,r'C:\Users\Administrator\Desktop\skk\gd_model')
# pd.read_pickle(r'C:\Users\Administrator\Desktop\skk\gd_model')
store.to_pkl_sk(gd)
store.to_csv(x)
store.to_txt(score)
