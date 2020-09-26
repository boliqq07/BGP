import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR

x, y = load_boston(return_X_y=True)

method = SVR(gamma="scale")
gd2 = GridSearchCV(method, cv=5,
                   param_grid=[{'C': [10000, 100, 50, 10, 5, 2.5, 1, 0.5, 0.1, 0.01]}],
                   scoring="r2", n_jobs=1)
gd2.fit(x, y)
res = gd2.cv_results_["mean_test_score"]

best_index = gd2.best_index_
s1 = res[best_index]

res2 = cross_val_score(gd2.best_estimator_, x, y, cv=5, scoring="r2")
s2 = np.mean(res2)
