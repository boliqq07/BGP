import pandas as pd
from sklearn.metrics import r2_score

from bgp.skflow import SymbolLearning
from mgetool.exports import Store
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

x = pd.read_csv(r'data1.csv')
x = x.iloc[:, 0:28].values

y = pd.read_csv(r'kim_raw_data.csv')
y = y["Bandgap, HSE06 (eV)"].values

x,y = shuffle(x,y,random_state =1)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

st = Store("no_intercept")

stop = lambda ind: ind.fitness.values[0] >= 0.90
sl = SymbolLearning(loop=r'MultiMutateLoop', cal_dim=False, pop=2000,
                    gen=20, add_coef=True, re_hall=2,store=True,
                    inter_add=False,stop_condition=stop,
                    random_state=2, n_jobs=12,
                    initial_max=2, max_value=4,
                    stats={"fitness_dim_max": ("max",)}
                    )
sl.fit(X_train, y_train,
       power_categories=(2, 3, 0.5, 0.33),
       categories=("Add", "Mul", "Sub", "Div"), )

print("best expession:", sl.expr)
y_test_es = sl.predict(X_test)
r2 = r2_score(y_test, y_test_es)
print("test score:", r2)

# data = sl.loop.top_n(20, ascending=False)
# st.to_csv(data, file_new_name="top_n")