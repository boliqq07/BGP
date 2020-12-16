base
==================

Base objects for symbolic regression.

Contains:
  - Class: :py:class:`bgp.base.SymbolSet`

  - Class: :py:class:`bgp.base.CalculatePrecisionSet`

  - Class: :py:class:`bgp.base.SymbolTree`

  - others


SymbolSet
>>>>>>>>>>>>

For example, the data can be imported from sklearn.
::

    if __name__ == "__main__":
        from sklearn.datasets import load_boston

        data = load_boston()
        x = data["data"]
        y = data["target"]
        c = [1, 2, 3]

The ``SymbolSet`` is a presentation set contains some 'blocks', which are including
features ( x\ :sub:`1`, x\ :sub:`2` .etc)
operators (+ - * / .etc) ,
and numerical term (2, 3, 0.5).
which can be added by ``add_features``, ``add_operations``, ``add_constants`` respectively.

The detail of ``add_features``,``add_operations`` can be found in :doc:`remarks`.

::

        from bgp.base import SymbolSet
        pset0 = SymbolSet()
        pset0.add_features(x, y)
        pset0.add_constants(c, )
        pset0.add_operations(power_categories=(2, 3, 0.5),
                 categories=("Add", "Mul", "exp"),
                 special_prob =  {"Mul": 0.5,"Add": 0.4,"exp": 0.1}
                 power_categories_prob = "balance")

Then the mode can be built with ``skflow.SymbolLearning``, just replace the fit parameters: 'pset'.
::

        from bgp.skflow import SymbolLearning
        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3,
                            cal_dim=True, re_hall=2, add_coef=True, cv=1,
                            random_state=1
                            )
        sl.fit(pset=pset0)
        score = sl.score(x, y, "r2")
        print(sl.expr)


SymbolTree
>>>>>>>>>>>

Individual Tree, each tree is one expression.

Generate expressions from pset.
::

    pset = SymbolSet()

    individual = SymbolTree.genGrow(pset, height , height+1,)

    population = [SymbolTree.genFull(pset, height , height+1,) for _ in range(5000)]


CalculatePrecisionSet
>>>>>>>>>>>>>>>>>>>>>>>

Definite the operations, features, and fixed constants.
One calculation ability extension for SymbolSet.
For example:
::

    cp = CalculatePrecisionSet(pset, scoring=[r2_score, ],score_pen=[1, ], filter_warning=True)

The cp could calculate the individual by:
::

    result = cp.calculate_detail(individual)

or calculate population::

    result = cp.parallelize_score(population)

