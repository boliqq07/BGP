#
# from sklearn.datasets import fetch_california_housing
# from sklearn.feature_selection import RFE
# from sklearn.preprocessing import normalize
# from sklearn.svm import SVR, LinearSVR
#
# data = fetch_california_housing()
# x = data.data
# y = data.target
# svr = SVR(gamma="scale")
# back = BackForward(svr, n_type_feature_to_select=4, primary_feature=None, multi_grade=2, multi_index=None,
#                    must_index=None, tolerant=0.01, verbose=0, random_state=None)
# # back.fit(x,y)
# x = normalize(x, norm='l2', axis=1, copy=True, return_norm=False)
# linearsvr = LinearSVR(max_iter=2000)
# rfe = RFE(linearsvr, n_features_to_select=None, step=1, verbose=0)
# rfe.fit(x, y)
