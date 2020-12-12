Quick start
==================

The symbol are conformed with the ''sklearn-style'' type, which can be easily to modeling with
``fit``, ``predict``, ``score``.

::

    if __name__ == "__main__":
        # data
        from sklearn.datasets import load_boston
        from bgp.skflow import SymbolLearning

        data = load_boston()
        x = data["data"]
        y = data["target"]
        c = [6, 3, 4]

        # start->
        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1)
        sl.fit(x, y, c=c)
        score = sl.score(x, y, "r2")
        print(sl.expr)

And return the results::

    >>>62.33 - 2.156*x10

:Note:

**When the result of one problem is not stable, the final expression is changed with random_state (random seed).
The random seeds between window and linux are different.**
