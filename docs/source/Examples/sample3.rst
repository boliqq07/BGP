Min Problem
================

This is a Min Problem sample.
::

    if __name__ == "__main__":
        from sklearn.datasets import fetch_california_housing
        from bgp.skflow import SymbolLearning
        from sklearn import metrics
        data = fetch_california_housing()
        x = data["data"][:100]
        y = data["target"][:100]

        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1,
                          scoring=[metrics.mean_absolute_error,],
                          score_pen=[-1,],
                          stats = {"fitness_dim_min": ("min",), "dim_is_target": ("sum",)},
                          )
        sl.fit(x, y)
        print(sl.expr)
