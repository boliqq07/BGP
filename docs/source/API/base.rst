base
==================

.. _base:

Base objects for to symbolic regression.

contains:
  - name: SymbolSet

  - name: CalculatePrecisionSet

  - name: SymbolTree


SymbolSet
>>>>>>>>>>>>

For example, the data can be import from sklearn.
::

    if __name__ == "__main__":
        from sklearn.datasets import load_boston

        data = load_boston()
        x = data["data"]
        y = data["target"]
        c = [1, 2, 3]

The ``SymbolSet`` is a presentation set contain the blocks, which is including
features ( x\ :sub:`1`, x\ :sub:`2` .etc)
operators (+ - * / .etc) ,
and numerical term (2, 3, 0.5).
which is added by ``add_features``,``add_operations``,
``add_constants``  "respectively"

::

        from bgp.base import SymbolSet
        pset0 = SymbolSet()
        pset0.add_features(x, y)
        pset0.add_constants(c, )
        pset0.add_operations(power_categories=(2, 3, 0.5),
                 categories=("Add", "Mul", "exp"),
                 special_prob =  {"Mul": 0.5,"Add": 0.4,"exp": 0.1}
                 power_categories_prob = "balance")

Then the mode can be built the same with ``skflow``, just replace the fit parameters: 'pset'.
::

        from bgp.skflow import SymbolLearning
        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3,
                            cal_dim=True, re_hall=2, add_coef=True, cv=1,
                            random_state=1
                            )
        sl.fit(pset=pset0)
        score = sl.score(x, y, "r2")
        print(sl.expr)


Parameters
:::::::::::

parameters for SymbolSet()

name: str

Methods
:::::::::::

* add_features

Add features with dimension and probability.

Parameters

X: np.ndarray
    2D data
y: np.ndarray
    1D data
feature_name: None, list of str
    the same size wih x.shape[1]
x_dim: 1 or list of Dim
    the same size wih x.shape[1], default 1 is dless for all x
y_dim: 1,Dim
    dim of y
x_prob: None,list of float
    the same size wih x.shape[1]
x_group: None or list of list, int
    features group

* add_operations

Add operations with probability.

Parameters

power_categories: Sized,tuple, None
    Examples:(0.5,2,3)
categories: tuple of str
    map table:
            {"Add": sympy.Add, 'Sub': Sub, 'Mul': sympy.Mul, 'Div': Div}

            {"sin": sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'ln': sympy.ln,

            {'Abs': sympy.Abs, "Neg": functools.partial(sympy.Mul, -1.0),

            "Rec": functools.partial(sympy.Pow, e=-1.0)}

            Others:  \n
            "Rem":  f(x)=1-x,if x true \n
            "Self":  f(x)=x,if x true \n

power_categories_prob:"balance", float
    float in (0,1]
    probability of power categories, "balance" is 1/n_power_cat
categories_prob: "balance", float
    float in (0,1]
    probabilityty of categories, except (+,-*,/), "balance" is 1/n_categories.\n
    Notes: the  (+,-*,/) are set as 1 to be a standard.
special_prob: None or dict
    Examples: {"Mul":0.6,"Add":0.4,"exp":0.1}
self_categories:list of dict,None
    the dict can be generate from newfuncV or defination self.
    the function at least containing:
    {"func": func, "name": name, "arity":2,"np_func": npf, "dim_func": dimf, "sym_func": gsymf}
    func:sympy.Function(name) object
    name:name
    arity:int,the number of parameter
    np_func:numpy function
    dim_func:dimension function
    sym_func:NewArray function. (unpack the group,used just for shown)
    See Also bgp.function.newfunc.newfuncV

* add_constants

Add features with dimension and probability.

Parameters

c_dim: 1, list of Dim
    the same size wih c
c: float,list
    list of float
c_prob: None, float, list of float
    the same size wih c



* add_features_and_constants

A unified version for add features and constant

* add_accumulative_operation

add accumulative operation.

Parameters

categories: tuple of str
    categories=("Self","MAdd","MSub", "MMul","MDiv")
categories_prob: None, "balance" or float.
    probility of categories  (0,1], except ("Self","MAdd", "MSub", "MMul", "MDiv"),
    "balance" is 1/n_categories.
    "MSub", "MMul", "MDiv" only work on the size of group is 2, else work like "Self".
    Notes: the  ("Self","MAdd","MSub", "MMul", "MDiv") are set as 1 and 0.1 to be a standard.
self_categories:list of dict,None
    the dict can be generate from newfuncD or defination self.
    the function at least containing:
    {"func": func, "name": name, "np_func": npf, "dim_func": dimf, "sym_func": gsymf}
    func:sympy.Function(name) object,which need add attributes: is_jump,keep.
    name:name
    np_func:numpy function
    dim_func:dimension function
    sym_func:NewArray function. (unpack the group,used just for shown)
    See Also bgp.function.newfunc.newfuncV
special_prob: None or dict
    Examples: {"MAdd":0.5,"Self":0.5}



* add_tree_to_features

Add the individual as a new feature to initial features.
not sure add seccess,because the value and name should be check and
different to exist.

Parameters

Tree: SymbolTree
    individual or expression
prob: int
    probability of this individual

* set_personal_maps
    To be developed
* bonding_personal_maps
    to be developed

Attributes
:::::::::::

Too much to show.

Some Example:

self.arguments = []  # for translate
self.name = name
self.y = None  # data y
self.y_dim = dless  # dim y

self.data_x_dict = {}  # data x

self.new_num = 0

self.terms_count = 0
self.prims_count = 0
self.constant_count = 0
self.dispose_count = 0

self.context = {"__builtins__": None}  # all elements map

self.dim_map = dim_map()
self.np_map = np_map()
self.gsym_map = gsym_map()

self.primitives_dict = {}
self.prob_pri = {}  # probability of operation default is 1

self.dispose_dict = {}
self.prob_dispose = {}  # probability of  structure operation, default is 1/n

self.ter_con_dict = {}  # term and const
self.dim_ter_con = {}  # Dim of and features and constants
self.prob_ter_con = {}  # probability of and features and constants

self.gro_ter_con = {}  # for group size calculation and simple

self.terminals_init_map = {}  # for Tree show
# terminals representing name "gx0" to represented name "[x1,x2]",
# or "newx0" to represented name "Add(Mul(x2,x4)+[x1,x2])".

self.terminals_symbol_map = {}  # for Tree show
# terminals representing name "gx0" to represented name "[x1,x2]",
# or "newx0" to represented name "Add(Mul(x2,x4)+[x1,x2])".

self.expr_init_map = {}  # for expr show
# terminals representing name "newx0" to represented name "(x2*x4+gx0)"
self.terminals_fea_map = {}  # for terminals Latex feature name show.

self.premap = PreMap.from_shape(3)
self.x_group = [[]]

SymbolTree
>>>>>>>>>>>

Individual Tree, each tree is one expression.

Generate expressions from pset.
::

    pset = SymbolSet()

    individual = SymbolTree.genGrow(pset, height , height+1,)

    population = [SymbolTree.genFull(pset, height , height+1,) for _ in range(5000)]


Parameters
:::::::::::::

arg: list
    list of operation or terminals.

Methods
::::::::::

cls.genGrow:
    Generate SymbolTree by genGrow method.
cls.genFull
    generate SymbolTree by genFull method.
self.to_expr:
    Transform to sympy object.
self.ppprint
    deprecated
self.depart
    simplified individual for calculation.
    Del the Attached properties, just name and expression for calculation.

Attributes
:::::::::::::::

self.p_name:
    present name
self.y_dim:
    dim of y
self.pre_y: np.ndarray
    predict y
self.expr: sympy.Expr
    expression
self.dim_score: 1 or 0
    correspond to dim of y or not



CalculatePrecisionSet
>>>>>>>>>>>>>>>>>>>>>>>

Definite the operations, features, and fixed constants.
One calculation ability extension for SymbolSet.
For example:
::

    cp = CalculatePrecisionSet(pset, scoring=[r2_score, ],score_pen=[1, ], filter_warning=True)

The cp could could calculate the individual by:
::

    result = cp.calculate_detail(individual)

or calculate population::

    result = cp.parallelize_score(population)

Parameters
:::::::::::::::::

name: str
    name
fuzzy: bool
    fuzzy or not
dim_type: object
    if None, use the y_dim
pset: SymbolSet
    SymbolSet
scoring: Callbale, default is sklearn.metrics.r2_score
    See Also sklearn.metrics
score_pen: tuple, default is sklearn.metrics.r2_score
    See Also sklearn.metrics
filter_warning:bool
    bool
score_pen: tuple of 1 or -1
    1 : best is positive, worse -np.inf \n
    -1 : best is negative, worse np.inf \n
    0 : best is positive , worse 0 \n
cal_dim: bool
    calculate dim or not, if not return dimless
add_coef: bool
    bool
inter_add: bool
    bool
inner_add: bool
    bool
n_jobs:int
    running core
batch_size:int
    batch size, advice batch_size*n_jobs = inds/n
tq:bool
    bool
cv:sklearn.model_selection._split._BaseKFold,int
    the shuffler must be False
    use cv spilt for score,return the mean_test_score.
    use cv spilt for predict,return the cv_predict_y.(not be used)
    Notes:
    if cv and refit, all the data is refit to determination the coefficients.
    Thus the expression is not compact with the this scores, when re-calculated by this expression


Methods
::::::::

All the methods of SymbolSet as the flowing.
The details can be found in self.__doc__.

* parallelize_score
    return list of (fitness_value, dim, dim_fitness)

* calculate_detail
    calculate the expression with fitting for one tree.
* calculate_simple
    calculate the expression without fitting for one tree.

* calculate_cv_score
    calculate the score.
* calculate_score
    calculate the score.

