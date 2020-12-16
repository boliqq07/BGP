skflow
==================

Contains:
  - Class: :py:class:`bgp.skflow.SymbolLearning`

    One "sklearn-type" implement to run symbol learning.
    We recommend this approach when rapid modeling.
    The SymbolLearning could implement most of the
    functions and without other assistance functions.

For example, the data can be import from sklearn.
::

      if __name__ == "__main__":
          from sklearn.datasets import load_boston
          from bgp.skflow import SymbolLearning

          data = load_boston()
          x = data["data"]
          y = data["target"]
          c = [1, 2, 3]

Import ``SymbolLearning`` and add the parameter (such as, with 500 population each generation,
with 3 generations, calculate the dimensions(units) of expressions, with 2 elites feedback,
add coefficient in expression,
with random state = 1).
::

          from bgp.skflow import SymbolLearning
          sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3, cal_dim=True,
                                re_hall=2, add_coef=True, random_state=1
                                )

Fitting data and add the binding with ``x_group``.
::

          sl.fit(x, y, c=c,x_group=[[1, 3], [0, 2], [4, 7]]))
          score = sl.score(x, y, "r2")
          print(sl.expr)

The detail of ``x_group`` can be found in :doc:`remarks`.

The ``SymbolLearning`` could implement most of the functions and without other assistance functions.

:Except:

* user-defined new operations
* user-defined probability of operation occurrence
* user-defined probability of features mutual influence

For these realizations, we could customer the pset (base.SymbolSet) in advance and pass to "pset" parameters.
For in-depth customization, please refer to ``base`` part and ``flow`` part.

More Examples:

:doc:`../Examples/index`


**Parameters** and **Methods** can be found in :py:mod:`bgp.skflow.SymbolLearning`.

**Attributes**

loop: str
    the running loop in flow part.
best_one:  SymbolTree
    the best one of expressions.
expr:  sympy.Expr
    the best one of expressions.
y_dim:  Dim
    dim of calculate y.
fitness: float
    score

The call relationship(correspondence) is as follows:

``flow.loop`` --> ``skflow.SymbolLearning``

``base.pset.add_features_and_constants`` --> ``skflow.SymbolLearning.fit``

``base.pset.add_operations`` --> ``skflow.SymbolLearning.fit``

