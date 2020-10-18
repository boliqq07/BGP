from bgp.skflow import SymbolLearning
import pandas as pd
if __name__ == "__main__":
    com_data = pd.read_csv(r'reg1.csv')
    x=com_data.iloc[:,:-1]
    y=com_data.iloc[:,1]
    # sl = SymbolLearning(loop="MutilMutateLoop", gen=30, pop=1000, hall=1, batch_size=40, re_hall=5,
    #                     n_jobs=12, mate_prob=0.9, max_value=2, initial_min=2, initial_max=3,
    #                     mutate_prob=0.8, tq=False, dim_type="coef",
    #                     re_Tree=0, store=False, random_state=1, verbose=True,
    #                     stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"], "h_bgp": ["mean"]},
    #                     add_coef=True, inter_add=True, out_add=True, cal_dim=True, vector_add=True,
    #                     personal_map=False)
    sl = SymbolLearning(loop=r'MultiMutateLoop')
    sl.fit()

