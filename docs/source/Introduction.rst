Introduction
==================

.. image:: img.jpg

BGP (Binding Genetic Programming)

Binding Genetic Programming is developed on Genetic Programming.
This tool is a symbol regression tool with dimension calculation,
which is aimed to establish expressions with physical limitation.
This tool built formulas, or called expression, using genetic algorithm,
by combined the ``blocks``, which is including
``operators`` (+ - * / .etc) ,
``features`` ( x\ :sub:`1`, x\ :sub:`2` .etc)
and ``numerical term`` (1,0.5,0.16).

**Dimension calculation** and **Artificial binding** are embedded in this tool.
These added modules are aimed to reduce the invalid search space, especially in
specific physical domain knowledge.

The numerical terms (coefficient,and intercept) in expressions are added
by **coefficient fitting**.
which is different to most of Genetic Programming packages.
We offered different methods to site the numerical terms,
to control the expressions to generate the best suitable one near the expected outcome.

Some helpful code can be copied from others package and adapt to new environment.
Infringement contents would be removed.
