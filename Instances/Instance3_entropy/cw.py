from bgp.skflow import SymbolLearning
import pandas as pd
if __name__ == "__main__":
    ###########第一个###########
    com_data = pd.read_csv(r'reg1.csv')
    x=com_data.iloc[:,:-1]
    y=com_data.iloc[:,-1]
    sl = SymbolLearning(loop=r'MultiMutateLoop', gen=5, add_coef=True, re_hall=2,random_state=0,re_Tree=1)
    sl.fit(x,y)


