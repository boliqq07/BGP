Complexity Control
===================

This is a part to equation complexity control from 3 aspect.

1.limitation of length of equation.

    initial_max:int
        max initial size of expression when first producing.
    initial_min : None,int
        min initial size of expression when first producing.
    max_value:int
        max size of expression.
    limit_type: "height" or "length",","h_bgp"
        limitation type for max_value, but just affect ``max_value`` rather than ``initial_max``, ``initial_min``.


2. Sites of fit coefficients.

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


3. Dimension limitation.
    (To some extent, the Dimension limitation could affects the complexity of the formula indirectly.)

    ``cal_dim``:
    The main switch of calculate dimension or not.

    ``dim_type``:
    What kind of dimension of equation fit the bill.

    "coef": af(x)+b. a,b have dimension, f(x)'s dimension is not dnan.

    "integer": af(x)+b. f(x) is with integer dimension.

    [Dim1,Dim2]: f(x)'s dimension in list.

    Dim: f(x) ~= Dim. (see fuzzy)

    Dim: f(x) == Dim.

    None: f(x) == pset.y_dim

Note:
    From sample 4, The formula to be more and more complicated.

1. Ordinary SL.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=False, n_jobs = 10,add_coef=False)
    sl.fit(x,y)


2. SL with add coefficients: af(x).::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=False, n_jobs = 10,add_coef=True,inter_add=False)
    sl.fit(x,y)


3. SL with add coefficients (default,if do not change the default parameters): af(x)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=False, n_jobs = 10,add_coef=True,inter_add=True)
    sl.fit(x,y)

4. SL with add coefficients, with dimension calculation (default): af(x)+b.::

    from sympy.physics.units import kg, m, pa, J, mol, K
    from bgp.functions.dimfunc import Dim, dless

    # Transform to SI unit, and get Dims
    gpa_dim= Dim.convert_to_Dim(1e9*pa, unit_system="SI")
    j_d_mol_dim = Dim.convert_to_Dim(1000*J/mol, unit_system="SI")
    kg_d_m3_dim = Dim.convert_to_Dim(kg/m**3, unit_system="SI")

    # or just write Dim by yourself
    K_dim= Dim([0,1,0,0,0,0,0])

    y_dim = dless
    x_dim = [dless,gpa_dim[1],j_d_mol_dim[1],K_dim[1],dless,kg_d_m3_dim]

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,
                        dim_type=None
                        # dim_type=y_dim
                        n_jobs = 10,add_coef=True,inter_add=True,)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)


5. SL with add coefficients, with dimension calculation, but relax the requirement:
just require that the dimension f(x) is not NaN for af(x)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,
                        dim_type="coef"
                        n_jobs = 10,add_coef=True,inter_add=True,)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)


6. SL with add coefficients, with dimension calculation,
but relax the requirement: just require that the dimension f(x) is not NaN for af(x)+b or af(x)+cf(x)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,dim_type="coef"
                        n_jobs = 10,add_coef=True,inter_add=True, inner_add=False, out_add=True, flat_add=False)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)


7. SL with add coefficients, with dimension calculation,
but relax the requirement: just require that the dimension f(x) is not NaN for flattened af(x)+cf(x)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,dim_type="coef"
                        n_jobs = 10,add_coef=True, inter_add=True, inner_add=False, out_add=False, flat_add=True)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)

8. SL with add coefficients, with dimension calculation,
but relax the requirement: just require that the dimension f(x) is not NaN for af(cx)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,dim_type="coef"
                        n_jobs = 10,add_coef=True,inter_add=True, inner_add=True, out_add=False, flat_add=False)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)

9. SL with add coefficients, with dimension calculation,
but relax the requirement: just require that the dimension f(x) is not NaN for af(cx)+b.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,dim_type="coef"
                        n_jobs = 10,add_coef=True,inter_add=True, inner_add=True, out_add=False, flat_add=False)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)

10. SL with add coefficients, with dimension calculation, change max_value.::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=True,dim_type="coef",
                        initial_max=7, initial_min=3,max_value=7,limit_type="h_bgp",
                        n_jobs = 10,add_coef=True,inter_add=True, inner_add=True, out_add=False, flat_add=False)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)

11. Complex equation(most complicated, slowest, unaccountably).::

    sl = SymbolLearning(loop="MultiMutateLoop", pop=100, gen=10, random_state=1,
                        cal_dim=False,
                        max_value=7,
                        n_jobs = 10,
                        add_coef=True,
                        inner_add=True)
    sl.fit(x,y,x_dim=x_dim,y_dim=y_dim)

