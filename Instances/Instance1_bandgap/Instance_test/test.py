
import os
from collections import Counter
from itertools import count

import pandas as pd
import numpy as np
from mgetool.show import BasePlot
from sklearn.model_selection import cross_val_predict, cross_val_score

test_data = pd.read_csv("other_data.csv")

df2 = test_data[(test_data['electronegativity(martynov&batsanov)_2'
                           ].isna())]
df2 = df2[(df2['electronegativity(martynov&batsanov)_1'
                           ].notna())]

data2 = df2[['Band_gap_HSE','Space_group_rlx','electronegativity(martynov&batsanov)_0',
       'electronegativity(martynov&batsanov)_1','ele_den','ncom_0','ncom_1']]

ty = Counter(np.array(data2["Space_group_rlx"].tolist()).astype(int))


data3 = data2.where(data2["Space_group_rlx"] != 225)
data3 = data3.where(data3['Band_gap_HSE'] <=15)

# data3 = data3.where(data3["Space_group_rlx"] ==221)

for i in (14,62,1,12,166,221):
    data4 = data3.where(data3["Space_group_rlx"] ==i)
    data4 = data4.dropna()

    xy = data4.values
    Y= xy[:,0]
    X = xy[:,1:]
    XX = np.vstack((X[:, 4] ** (0.333), X[:, 2], X[:, 3])).T
    c=data3["Space_group_rlx"]

    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    lin.fit(XX, Y)
    coef = lin.coef_
    inter = lin.intercept_
    # y_pre = lin.predict(XX)
    # score = lin.score(XX, Y)
    y_pre = cross_val_predict(lin,XX,Y, cv=5)
    score = cross_val_score(lin,XX,Y,cv=5,scoring='neg_mean_absolute_error').mean()

    #
    label = "Space Group: %s\n"%i +"MAE(CV): %.2f"% abs(score)

    import matplotlib.pyplot as plt
    p = BasePlot(font=None)

    def scatter(y_true, y_predict, strx='y_true', stry='y_predicted',label = ""):
        x, y = y_true, y_predict
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, marker='o', s=70, alpha=0.8, c='orange', linewidths=None, edgecolors='blue')
        ax.plot([0, 10], [0, 10], '--', ms=7, lw=3, alpha=0.7, color='black')


        ax.tick_params(which='major', labelsize=16)

        plt.text(1,8,label,fontdict={"size":20})
        plt.xlabel(strx,fontdict={"size":20})
        plt.ylabel(stry,fontdict={"size":20})
        plt.xlim([0,10])
        plt.ylim([0,10])

    scatter(Y, y_pre, strx='Experimental $E_{gap}$', stry='Predicted $E_{gap}$',label=str(label))

    plt.show()