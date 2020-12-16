Regression
===========

This is a regression, Max-problem sample.
::

    if __name__ == "__main__":
      from sklearn.datasets import load_boston
      from bgp.skflow import SymbolLearning

      data = load_boston()
      x = data["data"]
      y = data["target"]

      sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1)
      sl.fit(x, y)
      score = sl.score(x, y, "r2")
      print(sl.expr)
      
