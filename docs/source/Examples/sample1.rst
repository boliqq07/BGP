Regression
===========

This is a regression, Max-problem sample.
::

    if __name__ == "__main__":
      from sklearn.datasets import fetch_california_housing
      from bgp.skflow import SymbolLearning

      data = fetch_california_housing()
      x = data["data"][:100]
      y = data["target"][:100]

      sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1)
      sl.fit(x, y)
      score = sl.score(x, y, "r2")
      print(sl.expr)
      
