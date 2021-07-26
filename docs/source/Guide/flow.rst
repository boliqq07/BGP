flow
==================

Some definitions loop for genetic algorithm.

Contains:
  - Class: :py:class:`bgp.flow.BaseLoop`

    one node mate and one tree mutate.

  - Class: :py:class:`bgp.flow.MultiMutateLoop`

    one node mate and (one tree mutate, one node Replacement mutate, shrink mutate, difference mutate).

  - Class: :py:class:`bgp.flow.OnePointMutateLoop`

    one node Replacement mutate: (keep height of tree)

  - Class: :py:class:`bgp.flow.DimForceLoop`

    Select with dimension : (keep dimension of tree)

::

    if __name__ == "__main__":
        pset = SymbolSet()
        stop = lambda ind: ind.fitness.values[0] >= 0.880963
        bl = OnePointMutateLoop(pset=pset, gen=10, pop=1000, hall=1, batch_size=40, re_hall=3, \n
                        n_jobs=12, mate_prob=0.9, max_value=5, initial_min=1, initial_max=2, \n
                        mutate_prob=0.8, tq=True, dim_type="coef", stop_condition=stop,\n
                        re_Tree=0, store=False, random_state=1, verbose=True,\n
                        stats={"fitness_dim_max": ["max"], "dim_is_target": ["sum"]},\n
                        add_coef=True, inter_add=True, inner_add=False, cal_dim=True, vector_add=False,\n
                        personal_map=False)
        bl.run()

The **Parameters**, **Methods**, and **Attributes** for all loops are same.

* Parameters

    The Parameters is the same with ``skflow.SymbolLearning``, except the 'loop' parameter in ``skflow.SymbolLearning``.

* Methods

run:
    run the loop.

    The ``flow.BaseLoop.run`` is the base of  ``skflow.SymbolicLearning.fit``



