flow
==================

.. _flow:

Some definition loop for genetic algorithm.

contains:
  - name: BaseLoop

    one node mate and one tree mutate.

  - name: MultiMutateLoop

    one node mate and (one tree mutate, one node Replacement mutate, shrink mutate, difference mutate).
  - name: OnePointMutateLoop

    one node Replacement mutate: (keep height of tree)
  - name: DimForceLoop

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

The **Parameters**, **Methods**, and **Attributes** for all loop is same.

* Parameters

    The Parameters is the same with ``skflow``, except the 'loop' parameter in ``skflow``.

* Methods

    run

        run the loop.
        The ``flow.BaseLoop.run`` is the base of  ``skflow.SymbolicLearning.fit``

* Attributes

    None


