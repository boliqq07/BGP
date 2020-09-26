import warnings

import numpy as np
import pandas as pd
from bgp.selection.corr import Corr
from bgp.selection.ugs import cluster_printting
from mgetool.exports import Store
from mgetool.imports import Call
from scipy.spatial.distance import pdist, squareform


# from numbapro import jit, float32


def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def cal_dor(x):
    xT = x.T
    dis = [[distcorr(i, j) for j in xT] for i in xT]
    return np.array(dis)


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    def get_abbr(X_frame_name):
        element_table = pd.read_excel(r'F:\machine learning\feature_toolbox1.0\featurebox\data\element_table.xlsx',
                                      skiprows=0, index_col=0)
        name = list(element_table.loc["x_name"])
        abbr = list(element_table.loc["abbrTex"])
        name.extend(['face_dist1', 'vor_area1', 'face_dist2', 'vor_area2', "destiny", 'volume', "ele_ratio"])
        abbr.extend(['$d_{vf1}$', '$S_{vf1}$', '$d_{vf2}$', '$S_{vf2}$', r"$\rho_c$", "$V_c$", "$ele_ratio$"])
        index = [name.index(i) for i in X_frame_name]
        abbr = np.array(abbr)[index]
        return abbr


    store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS\3.2')
    data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp_last\1.generate_data',
                r'C:\Users\Administrator\Desktop\band_gap_exp_last\3.MMGS')

    all_import_structure = data.csv.all_import_structure
    data_import = all_import_structure

    select = ['destiny', 'distance core electron(schubert)', 'energy cohesive brewer', 'volume atomic(villars,daams)',
              'radii covalent', 'electronegativity(martynov&batsanov)', 'latent heat of fusion']
    select = ['destiny'] + [j + "_%i" % i for j in select[1:] for i in range(2)]

    data216_import = data_import.iloc[np.where(data_import['group_number'] == 216)[0]]
    data225_import = data_import.iloc[np.where(data_import['group_number'] == 225)[0]]
    data216_225_import = pd.concat((data216_import, data225_import))

    X_frame = data225_import[select]
    y_frame = data225_import['exp_gap']

    X = X_frame.values
    y = y_frame.values

    discorr = cal_dor(X)

    co = Corr(muti_index=[1, len(X)])
    co.fit(X, pre_cal=discorr, method="max")
    discorr_shirnk = 1 - co.cov_shrink
    X_frame_name = co.transform(X_frame.columns.values)
    X_frame_name = [i.replace("_0", "") for i in X_frame_name]
    X_frame_abbr = get_abbr(X_frame_name)

    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.15, min_samples=2, metric='precomputed', metric_params=None,
                algorithm='auto', leaf_size=30, p=2, n_jobs=None)
    db.fit(discorr_shirnk)

    label = db.labels_
    set_label = list(set(label))
    group = [[i for i in range(len(label)) if label[i] == aim] for aim in set_label]
    name = [[X_frame_name[j] for j in i] for i in group]

    see = [co.inverse_transform_index(i) for i in group]
    se = [co.feature_unfold(i) for i in see]
    a = np.zeros((13, 1))
    for i, j in enumerate(se[:]):
        for p in j:
            a[p] = i

    for t in se[-1]:
        a[t] = t + 20

    label[0] = 2
    label[2] = 3
    label[6] = 4
    cluster_printting(slices=X_frame_abbr, node_color=label, edge_color_pen=0.5, binary_distance=co.cov_shrink,
                      print_noise=0, node_name=X_frame_abbr,
                      )
