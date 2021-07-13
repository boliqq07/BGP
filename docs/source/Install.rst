Install
==================

Requirements
::::::::::::

Packages:

============= ============  ============
 Dependence   Name          Version
------------- ------------  ------------
 necessary    sympy         >=1.6
 necessary    deap          >=1.3.1
 necessary    scikit-learn  >=0.22.1
 recommend    torch         >=1.5.0
 recommend    pymatgen      \
 recommend    scikit-image  \
 recommend    minepy        \
============= ============  ============

Method 1
::::::::::::

Install with pip ::

    pip install BindingGP

:Note:

    If VC++ needed for windows wheel, such as `spblib <https://github.com/spglib/spglib>`_ ,
    Please download the dependent packages named (spglib‑1.*‑cp3*‑*.whl) from
    `Python Extension Packages <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_ and install offline.
    please reference to Method 2.

Method 2
::::::::::::

Install by step:

1. sympy ::

    pip install sympy>=1.6

Reference: https://www.sympy.org/en/index.html

2. deap ::

    pip install deap

Reference: https://github.com/DEAP/deap

3. pymatgen ::

    conda install /your/local/path/spglib‑1.*‑cp3*‑*.whl
    pip install pymatgen

Reference: https://github.com/materialsproject/pymatgen,
https://github.com/spglib/spglib/tree/develop/python

3. pymatgen(options) ::

    conda install --channel conda-forge pymatgen

Reference: https://github.com/materialsproject/pymatgen

4. scikit-learn ::

    conda install sklearn

Reference: https://github.com/materialsproject/pymatgen


5. mgetool::

    pip install mgetool

Reference: https://github.com/Mgedata/mgetool

6. BGP::

    pip install BindingGP
