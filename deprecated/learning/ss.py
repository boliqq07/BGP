# 穷举
from itertools import combinations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from tqdm import tqdm


# 管道
# pca = PCA(n_components=10)
# SKB = SelectKBest(f_regression, k=5)
# clf = svm.SVC(kernel='linear')
#
# SKB_SVC = Pipeline([("pca",PCA),('SKB', SKB), ('svc', clf)])
#
# a = SKB_SVC[0]
# a = SKB_SVC["SKB"]
# a = SKB_SVC.named_steps.SKB


def exhausted(x, n_select=(2, 3, 4)):
    n_feature = x.shape[-1]
    n_feature_list = list(range(n_feature))
    slice_all = []

    for i in n_select:
        slice_all.extend(list(combinations(n_feature_list, i)))

    return slice_all


def score_exhausted(X, y, n_select=(2, 3, 4), store=True, model=None, gd=False, para_grid=None):
    a = exhausted(X, n_select=n_select)
    dict_re = {}

    if model is None:
        las = LogisticRegression(solver='lbfgs')
    else:
        las = model

    for j, i in enumerate(tqdm(a)):
        x = X[:, i]
        if gd is True:
            assert para_grid
            gd = GridSearchCV(las, param_grid=para_grid, cv=5)
            gd.fit(x, y)
            score = gd.best_score_
        else:
            score = cross_val_score(las, x, y, scoring='accuracy', cv=5).mean()

        dict_re["%s" % j] = [i, score]
    if store:
        import pandas as pd
        d = pd.DataFrame(dict_re).T
        d.to_csv("dict_re.csv")
    return dict_re


# import warnings
# warnings.filterwarnings("ignore")


X, y = make_classification(n_informative=5, n_redundant=0, random_state=42)
dict_re = score_exhausted(X, y, n_select=(2, 3), store=True, model=LinearSVC(), gd=True, para_grid={"C": [1, 10, 20]})
