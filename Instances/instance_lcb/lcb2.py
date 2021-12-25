import numpy as np
import pandas as pd
from mgetool.show import BasePlot
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from bgp.skflow import SymbolLearning


def search_space(*arg:np.ndarray):
    meshes = np.meshgrid(*arg)
    meshes = [_.ravel() for _ in meshes]
    meshes = np.array(meshes).T
    return meshes

data = pd.read_csv("lcb2.csv",index_col=None,header=None).values

datav = data[1:,1:]
datam1 = data[1:,0]
datam2 = data[0,1:]
#
gd = search_space(datam1, datam2)
datav2 = datav.ravel(order="F")
#
data2 = np.concatenate((gd,datav2.reshape(-1,1)), axis=1)

# train_test_split()

est_gp = SymbolLearning(loop='MultiMutateLoop', pop=1000, gen=20, mutate_prob=0.5, mate_prob=0.8, hall=1, re_hall=1,
                 re_Tree=None, initial_min=None, initial_max=3, max_value=4,
                 scoring=(metrics.mean_absolute_error,), score_pen=(-1,), filter_warning=True, cv=1,
                 add_coef=True, inter_add=True, inner_add=True, vector_add=False, out_add=False, flat_add=False,
                 cal_dim=False, dim_type=None, fuzzy=False, n_jobs=8, batch_size=40,
                 random_state=1, stats=None, verbose=True, migrate_prob=0,
                 tq=True, store=True, personal_map=False, stop_condition=None, details=False, classification=False,
                 score_object="y",)

est_gp.fit(gd, datav2,categories=("Mul", "Div","Add", "exp"),power_categories=(0.5,2))
e = est_gp.loop.top_n(100,ascending=True)
#
x0 = data2[:,0]
x1 = data2[:,1]
y = data2[:,2]

pre_y = 16.18*np.exp(-0.46792396*x1 - 1.5177372*x1/(-0.1052*x0 - 9.002*x1)) - 2.694


from mgetool.show import BasePlot
bp = BasePlot()
plt = bp.scatter_45_line(y,pre_y)
plt.show()
