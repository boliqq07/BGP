import pandas as pd
from sklearn.linear_model import LinearRegression

re = pd.read_excel("reskk.xlsx")

lr = LinearRegression(fit_intercept=False, normalize=False)
y = re["y"].values
x = re[["x1", "x2"]].values
lr.fit(x, y)
score = lr.score(x, y)
predict_y = lr.predict(x)
coef = lr.coef_
if hasattr(lr, "intercept_"):
    intercept = lr.intercept_
