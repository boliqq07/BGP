skflow
==================

One sklearn-type implement to run symbol learning. We recommend this approach when rapid modeling.
The SymbolLearning could implement most of the functions and without other assistance functions.

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
with random state = 1) .
::

          from bgp.skflow import SymbolLearning
          sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3, cal_dim=True,
                                re_hall=2, add_coef=True, random_state=1
                                )

Fitting data and  add the binding with ``x_group``.
::

          sl.fit(x, y, c=c,x_group=[[1, 3], [0, 2], [4, 7]]))
          score = sl.score(x, y, "r2")
          print(sl.expr)

The ``skflow.SymbolLearning`` could implement most of the functions and without other assistance functions.

Except:

* user-defined new operations
* user-defined probability of operation occurrence
* user-defined probability of features mutual influence

For these realization, we could customer the pset (base.SymbolSet) in advance and pass to "pset" parameters.
For in-depth customization, please refer to ``base`` part and ``flow`` part.

The call relationship(correspondence) is as follows:

``flow.loop`` --> ``skflow.SymbolLearning``

``base.pset.add_features_and_constants`` --> ``skflow.SymbolLearning.fit``

``base.pset.add_operations`` --> ``skflow.SymbolLearning.fit``


Parameters
>>>>>>>>>>>

loop:  str,None
    bgp.flow.BaseLoop
    ['BaseLoop','MultiMutateLoop','OnePointMutateLoop', 'DimForceLoop'...]
pop: int
    number of population
gen: int
    number of generation
mutate_prob: float
    probability of mutate
mate_prob: float
    probability of mate(crossover)
initial_max: int
    max initial size of expression when first producing.
initial_min :  None,int
    min initial size of expression when first producing.
max_value: int
    max size of expression
hall: int,>=1
    number of HallOfFame (elite) to maintain
re_hall: None or int>=2
    Notes:  only valid when hall
    number of HallOfFame to add to next generation.
re_Tree:  int
    number of new features to add to next generation.
    0 is false to add.
personal_map: bool or "auto"
    "auto" is using 'premap' and with auto refresh the 'premap' with individual.
    True is just using constant 'premap'.
    False is just use the prob of terminals.
scoring:  list of Callable, default is [sklearn.metrics.r2_score,]
    See Also sklearn.metrics
score_pen:  tuple of  1, -1 or float but 0.
    >0 :  max problem, best is positive, worse -np.inf
    <0 :  min problem, best is negative, worse np.inf
    Notes: 
    if multiply score method, the scores must be turn to same dimension in prepossessing
    or weight by score_pen. Because the all the selection are stand on the mean(w_i*score_i)
    Examples:  [r2_score] is [1],
cv: sklearn.model_selection._split._BaseKFold,int
    the shuffler must be False,
    default=1 means no cv
filter_warning: bool
    filter warning or not
add_coef: bool
    add coef in expression or not.
inter_add：bool
    add intercept constant or not
inner_add: bool
    add inner coefficients or not
out_add: bool
    add out coefficients or not
flat_add: bool
    add flat coefficients or not
vector_add: bool
    add vector coefficients or not
n_jobs: int
    default 1, advise 6
batch_size: int
    default 40, depend of machine
random_state: int
    None,int
cal_dim: bool
    calculate the dim or not.
dim_type: Dim or None or list of Dim
    "coef":  af(x)+b. a,b have dimension,f(x) is not dnan. 
    "integer":  af(x)+b. f(x) is integer dimension. 
    [Dim1,Dim2]:  f(x) in list. 
    Dim:  f(x) ~= Dim. (see fuzzy) 
    Dim:  f(x) == Dim. 
    None:  f(x) == pset.y_dim
fuzzy: bool
    choose the dim with same base with dim_type,such as m,m^2,m^3.
stats: dict
    details of logbook to show. 
    Map: 
    values= {"max":  np.max, "mean":  np.mean, "min":  np.mean, "std":  np.std, "sum":  np.sum}
    keys= {"fitness": just see fitness[0],"fitness_dim_max":  max problem, see fitness with demand dim,
    "fitness_dim_min":  min problem, see fitness with demand dim,
    "dim_is_target":  demand dim,
    "coef":   dim is True, coef have dim, 
    "integer":   dim is integer}
    if stats is None, default is : 
    stats = {"fitness_dim_max":  ("max",), "dim_is_target":  ("sum",)}   for cal_dim=True
    stats = {"fitness":  ("max",)}                                      for cal_dim=False
    if self-definition, the key is func to get attribute of each ind./n
    Examples: 
    def func(ind): 
    return ind.fitness[0]
    stats = {func:  ("mean",), "dim_is_target":  ("sum",)}
verbose: bool
    print verbose logbook or not
tq: bool
    print progress bar or not
store: bool or path
    bool or path
stop_condition: callable
    stop condition on the best ind of hall, which return bool,the true means stop loop.
    Examples: 
    def func(ind):
    c = ind.fitness.values[0]>=0.90
    return c
pset: SymbolSet
    the feature x and target y and others should have been added.
details: bool
    return expr and predict_y or not.
classification:  bool
    classification or not.

  
Methods
>>>>>>>>>>>

* fit

If pset is given, the other parameters for fit method is invalid. Due to the other parameters
have defined in pset in advance.

Parameters

X: np.ndarray

y: np.ndarray

c: list of float, None

x_dim:  1 or list of Dim
    the same size wih x.shape[1], default 1 is dless for all x
y_dim:  1,Dim
    dim of y
c_dim:  1,list of Dim
    the same size wih c.shape, default 1 is dless for all c

x_prob:  None,list of float
    the same size wih x.shape[1]
c_prob:  None,list of float
    the same size wih c
x_group: list of list
    Group of x.
    See Also pset.add_features_and_constants
power_categories:  Sized,tuple, None
    Examples: (0.5,2,3)
categories:  tuple of str
    map table: 
            {"Add":  sympy.Add, 'Sub':  Sub, 'Mul':  sympy.Mul, 'Div':  Div}

            {"sin":  sympy.sin, 'cos':  sympy.cos, 'exp':  sympy.exp, 'ln':  sympy.ln,

            {'Abs':  sympy.Abs, "Neg":  functools.partial(sympy.Mul, -1.0),

            "Rec":  functools.partial(sympy.Pow, e=-1.0)}

            Others:   
            "Rem":   f(x)=1-x,if x true 
            "Self":   f(x)=x,if x true 

pset: SymbolSet
    See Also SymbolSet
warm_start:  bool
    warm start or not.
    Note: 
    If you offer pset in advance by user, please check carefully the feature numbers,
    especially when use "re_Tree.
    because the new features are add.
    Reference: 
    CalculatePrecisionSet.update_with_X_y
new_gen:  None,int
    warm_start generation.

* preidct
    return the predicted y.
* score
    return the score.

Attributes
>>>>>>>>>>>

.loop: str
    the running loop in flow part.
.best_one:  SymbolTree
    the best one of expressions
.expr:  sympy.Expr
    the best one of expressions
.y_dim:  Dim
    dim of calculate y
.fitness: float
    score


