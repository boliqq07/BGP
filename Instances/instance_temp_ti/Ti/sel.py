from functools import partial

import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, ShuffleSplit, train_test_split, cross_val_score
from sklearn.utils import shuffle

from ga import GA
from methods import method_pack


def fitness_func_cv(ind, model, x, y, return_model=False):
    index = np.where(np.array(ind) == 1)[0]
    x = x[:, index]
    if x.shape[1] > 1:
        pass
    else:
        x = x.reshape(-1, 1)

    svr = model
    svr.fit(x, y)
    if hasattr(svr, "best_score_"):
        sc = svr.best_score_
    else:
        sc = svr.score(x, y)
    if return_model:
        return sc, svr
    else:
        return sc,

def score_fitness(ind, model, x, y, scoring="r2"):
    index = np.where(np.array(ind) == 1)[0]
    x = x[:, index]
    X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.33)
    svr = model.best_estimator_
    svr.fit(X_train, y_train)
    pre_y_test = svr.predict(X_test)
    pre_y_train = svr.predict(X_train)
    if scoring =="r2":
        return r2_score(y_train,pre_y_train),r2_score(y_test,pre_y_test)
    elif scoring =='neg_root_mean_squared_error':
        return mean_squared_error(y_train,pre_y_train)**0.5,mean_squared_error(y_test,pre_y_test)**0.5
    else:
        raise NotImplementedError


if __name__ == "__main__":
    method = method_pack(['SVR-set', "KRR-set", "GBR-em",
                          "RFR-em", "AdaBR-em", "DTR-em",
                          "LASSO-L1", "BRR-L1", "SGDR-L1", "PAR-L1"], me="reg", scoreing=None, gd=True, cv=3)

    gd2 = method[0]

    dataTi = pd.read_csv("Ti.csv", index_col=0)

    datanp = dataTi.values

    x = datanp[:, 1:]
    y = datanp[:, 0]

    x, y = shuffle(x, y, random_state=0)

    nor = preprocessing.MinMaxScaler()
    x = nor.fit_transform(x)

    import numpy.random as nprd

    nprd.seed(1)

    fitness_func = fitness_func_cv

    ####GA
    size = 5
    def ga():
        fitn = partial(fitness_func, model=gd2, x=x, y=y)
        best = GA(x,y, fitn, pop_n=500, hof_n=1, cxpb=0.8, mutpb=0.4,
                  ngen=10, max_or_min="max",
                  mut_indpb=0.1, n_jobs=12, min_=size, max_=size+1)

        names = np.array((dataTi.columns))[1:]

        for i in best:
            indexes = np.array(i)
            indexes = indexes.astype(np.bool)
            print(names[indexes])
            print(i, i.fitness)

            gd2.scoring = 'neg_root_mean_squared_error'
            fitn = partial(fitness_func, model=gd2, x=x, y=y)
            score = fitn(ind=i)
            print("RMSE:",score)
            gd2.scoring = 'r2'
            fitn = partial(fitness_func, model=gd2, x=x, y=y)
            score = fitn(ind=i)
            print("R2:",score)

            gd2.scoring = 'neg_root_mean_squared_error'
            fitn = partial(score_fitness, model=gd2, x=x, y=y,scoring="neg_root_mean_squared_error")

            score = fitn(ind=i)
            print("RMSE:",score)

            gd2.scoring = 'r2'
            fitn = partial(score_fitness, model=gd2, x=x, y=y)
            score = fitn(ind=i)
            print("R2:",score)

    ga()
    # for i in range(100):
    #     ga()

    ######spilt
    # gd2.scoring = 'neg_root_mean_squared_error'
    # gd2.scoring = 'r2'
    # fitn = partial(fitness_func, model=gd2, x=x, y=y,return_model=True)
    # ind2 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # score, model2 = fitn(ind=ind2)  # 2
    # print(score)
    # ind3 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # score,model3 = fitn(ind=ind3)  # 3
    # print(score)
    # ind4=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # score,model4 = fitn(ind=ind4)  # 4
    # print(score)
    # ind5=[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # score,model5 = fitn(ind=ind5)  # 5
    # print(score)
    #
    # #####################
    # ind2 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    # ind3 = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #
    # ind4=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    #
    # ind5=[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    #
    # dataTi = pd.read_csv("Ti.csv", index_col=0)
    # samples = np.array((dataTi.index))
    # s0 = [sample for sample in samples if 'S0' in sample ]
    # s1 = [sample for sample in samples if 'S1' in sample ]
    # s2 = [sample for sample in samples if 'S2' in sample ]
    #
    # datas0 = dataTi.loc[s0].values
    # datas1 = dataTi.loc[s1].values
    # datas2 = dataTi.loc[s2].values
    # #
    # fitn = partial(fitness_func, model=gd2, x=x, y=y,return_model=True)
    # for mi,indi in enumerate([ind2,ind3,ind4,ind5]):
    #     score, modeli = fitn(ind=indi)
    #     print("CV", ["sub_f2", "sub_f3", "sub_f4", "sub_f5"][mi], score)
    #     for di, datai in enumerate([datas0, datas1, datas2]) :
    #
    #         x = datai[:, 1:]
    #         y = datai[:, 0]
    #         x = nor.transform(x)
    #         index = np.where(np.array(indi) == 1)[0]
    #         x = x[:, index]
    #
    #         model_ = modeli.best_estimator_
    #
    #         y_pred = model_.predict(x)
    #         scorermse = mean_squared_error(y, y_pred)**0.5
    #         scorer2 = r2_score(y, y_pred)
    #         # scorermse = cross_val_score(model_,x,y,scoring='neg_root_mean_squared_error').mean()
    #         # scorer2 = cross_val_score(model_,x,y).mean()
    #         print(["sub_f2","sub_f3","sub_f4","sub_f5"][mi],["dataS0","dataS1","dataS2"][di],scorer2,scorermse)
