Classification
================

This is a classification sample.
::

    if __name__ == "__main__":
        from sklearn import metrics
        from sklearn.utils import shuffle
        from sklearn.datasets import load_iris
        from bgp.skflow import SymbolLearning

        data = load_iris()
        x = data["data"][:98, :]
        x[40:60] = shuffle(x[40:60], random_state=2)
        y = data["target"][:98]
        c = None

        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=2, random_state=1,
        classification=True, scoring=[metrics.accuracy_score,], score_pen=[1,])
        sl.fit(x, y)

        print(sl.expr)
