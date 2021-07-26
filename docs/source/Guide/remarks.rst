Remarks
==================

This part is not a module but some notes about key parameters and core problems.
The examples in :doc:`../Examples/index`, :doc:`../Examples/sample7`

Contains:
  - Binding: ``x_group`` in :py:func:`bgp.skflow.SymbolLearning.fit` or :py:func:`bgp.base.SymbolSet.add_features`.

  - Dim: :py:class:`bgp.functions.dimfunc.Dim`

  - Dimension: ``cal_dim``, ``dim_type`` in :py:class:`bgp.skflow.SymbolLearning` or or :py:class:`BaseLoop`, ``x_dim, y_dim, c_dim`` in :py:func:`bgp.skflow.SymbolLearning.fit` or :py:class:`BaseLoop`.

  - Coefficients: ``add_coef, inner_add, inter_add, out_add, flat_add, vector_add`` in :py:class:`bgp.skflow.SymbolLearning` or :py:class:`BaseLoop`.

Binding
>>>>>>>>>>>>
Assume there is (x\ :sub:`1`,x\ :sub:`2`,x\ :sub:`3`,...,x\ :sub:`n`) features in data.
if you want to make x\ :sub:`1`,x\ :sub:`2` banded:
::

    x_group = [[1,2],]

if you want make each banded with size, such as (x\ :sub:`1`,x\ :sub:`2`),(x\ :sub:`3`,x\ :sub:`4`),...,(x\ :sub:`n-1`,x\ :sub:`n`):

::

    x_group = 2

The group size should be more than 2.


Dim
>>>>>

The ``Dim`` is :py:class:`bgp.functions.dimfunc.Dim`.

The default dimension SI system with 7 number.

The basic unit are:

{'meter': "m", 'kilogram': "kg", 'second': "s",
'ampere': "A", 'mole': "mol", 'candela': "cd", 'kelvin': "K"}


The basic unit can be represented by:
['length', 'mass', 'time', 'current', 'amount_of_substance',
'luminous_intensity', 'temperature']

1.can be constructed by list of number.

2.can be translated from a sympy.physics.unit.

Examples::

    from sympy.physics.units import Kg
    scale,dim = Dim.convert_to_Dim(Kg)

Examples::

    dim=[1,0,1,0,1,0,0]
    dim = Dim(dim)

Supplementary Note 1:
    Dimensional calculation, when the data should be units, but you do not use dimensional calculation subjectively, in this case, the model can be performed, that is, the data is assumed to be completely dimensional-uniformed data.

Supplementary Note 2:
    Dimensional calculation, from different units to standard units for calculation, which must have shrinkage coefficient.

    1. We suggest that the data be processed into the data value corresponding to the standard unit (dimensional reference), that is, the scaling coefficient is multiplied into the data.

    2. We have also integrated unit to dimensional transformation tools from sympy units.

    (Dim.convert_x, dim.convert_XI, dim.convert_x are used to convert x, y, and C respectively and the converted data and dimensions are obtained)

    In either case, the formula for the final result is correct, but the coefficient value is not your initial data (but the data corresponding to the standard units).You may need to manually re-fit the coefficient values.

Supplementary Note 3:
    Dimension calculation, some units are too small or too large, the data will be very small or very large when multiplied by the scaling factor, such as 10^16.

    If data is pre-processed using the ``MagnitudeTransformer`` provided by us, in this case you also need to manually re-fit the coefficient value.

Supplementary Note 4:
    The unit of coefficient, we do not provide the unit of coefficient value, the default unit of coefficient is calculated by means of completion.

Supplementary Note 5:
    The calculation roles for dimension can be seen in `Developer Manual.pdf <https://boliqq07.github.io/BGPdocument/doc.pdf)>`_,

Dimension:
>>>>>>>>>>>>
For :py:class:`bgp.skflow.SymbolLearning`.

The ``cal_dim`` is only valid when ``x_dim, y_dim`` are given.
When it is True, the dimension of result expression would be checked with ``dim_type``.

In default, the ``dim_type`` is "coef",

Without coefficient: that is ``dim_type``=``y_dim`` ,

With coefficient: assume expression is y=af(x)+b, a,b have dimension, f(x)'s dimension is not nan .

Of course, we can tighten the restriction,
such as make the ``dim_type`` = ``y_dim`` make the expression must have dimension.

The more strict from top to bottom:

Parameters:

    :"coef": af(x)+b. a,b have dimension, f(x)'s dimension is not dnan.

    :"integer": af(x)+b. f(x) is with integer dimension.

    :[Dim1,Dim2]: f(x)'s dimension in list.

    :Dim: f(x) ~= Dim. (see fuzzy)

    :Dim: f(x) == Dim.

    :None: f(x) == pset.y_dim

Coefficients:
>>>>>>>>>>>>>>

``add_coef, inner_add, out_add, flat_add, vector_add, inter_add`` in :py:class:`bgp.skflow.SymbolLearning` or :py:class:`BaseLoop`.

``add_coef`` is 'main switch' or others.

Assume the initial expression is y=f(x)

add_coef:
The main switch of coefficients. default:
Add the coefficients of expression. such as y=cf(x).

inter_add:
Add the intercept of expression. such as y=f(x)+b.

out_add:
Add the coefficients of expression. such as y=a(x),
but for polynomial join by ``+`` and ``-``,the coefficient would add before each term.
such as y=af1(x)+bf2(x).

flat_add:
flatten the expression and add the coefficients out of expression. such as y=af`1(x)+bf`2(x)+ef`3(x),
(the old expression: y = x*(f1(x)+f2(x)+f3(x))).

inner_add:
Add the coefficients inner of expression. such as y=cf(ax).

vector_add:
only valid when x_group is True, add different coefficients on group x pair.

For the ``inner_add, inter_add, out_add, flat_add``, just only one can be selected.
