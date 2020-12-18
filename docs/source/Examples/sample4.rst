Dimension
================

This is a Dimension calculation sample.
::

      if __name__ == "__main__":
          from bgp.functions.dimfunc import dless
          from sklearn.datasets import load_boston
          from bgp.skflow import SymbolLearning

          data = load_boston()
          x = data["data"]
          y = data["target"]
          x_dim = [dless, dless, dless, dless, dless, dless, dless, dless,dless, dless, dless, dless, dless]
          y_dim = dless

          sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1,cal_dim=True, dim_type="coef")
          sl.fit(x, y,x_dim=x_dim, y_dim=y_dim)
          score = sl.score(x, y, "r2")
          print(sl.expr)

The details of `Dim` can be found in :doc:`../Guide/remarks`
