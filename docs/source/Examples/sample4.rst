Dimension
================

This is a Dimension calculation sample.
::

      if __name__ == "__main__":
          from bgp.functions.dimfunc import dless
          from sklearn.datasets import fetch_california_housing
          from bgp.skflow import SymbolLearning

          data = fetch_california_housing()
          x = data["data"][:100]
          y = data["target"][:100]
          x_dim = [dless, dless, dless, dless, dless, dless, dless, dless,dless, dless, dless, dless, dless]
          y_dim = dless

          sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1,cal_dim=True, dim_type="coef")
          sl.fit(x, y,x_dim=x_dim, y_dim=y_dim)
          score = sl.score(x, y, "r2")
          print(sl.expr)

The details of `Dim` can be found in :doc:`../Guide/remarks`
