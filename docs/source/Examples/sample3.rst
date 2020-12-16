Min Problem
================

This is a Min Problem sample.
::

    if __name__ == "__main__":
        from sklearn.datasets import load_boston
        from bgp.skflow import SymbolLearning
        from sklearn import metrics
        data = load_boston()
        x = data["data"]
        y = data["target"]

        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1,
                          scoring=[metrics.mean_absolute_error,],
                          score_pen=[-1,],
                          stats = {"fitness_dim_min": ("min",), "dim_is_target": ("sum",)},
                          )
        sl.fit(x, y)
        print(sl.expr)
