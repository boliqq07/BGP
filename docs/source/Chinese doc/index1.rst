中文文档
=========
:py:class:`bgp.skflow.SymbolLearning` 使用"sklearn-type" 的方式实现符号回归，在快速建模时，我们推荐使用此方法。
  
默认参数针对最大值回归问题，您可以自定义参数以应用到其他方面。

由于参数过多，我们建议关注方向如下：

:关注点: 参数名称

:基础: loop, pop, gen, random_state, mutate_prob, mate_prob

:效率控制: random_state, batch_size, random_state

:展示控制: stats, verbose, store

:公式长度控制: max_value, initial_max

:系数添加方式: add_coef, inner_add, inter_add, out_add, flat_add

:精英算法: re-hall

:量纲计算: dim_type, cal_dim, fuzzy, fit(x_dim,y_dim,c_dim)

:分类问题: classification, scoring, score_pen

:最小值问题: stats, scoring, score_pen

:特征绑定: fit(x_group)

:自定义概率: fit(x_prob, c_prob)

:热启动: fit.(warm_start), fit(new_gen)
---
1.最简单的使用可以参考
此 ``SymbolLearning`` 封装可以在不借助其他辅助功能的情况下实现大部分功能。
::

    if __name__ == "__main__":
        from sklearn.datasets import load_boston
        from bgp.skflow import SymbolLearning

        data = load_boston()
        x = data["data"]
        y = data["target"]
        c = [1, 2, 3]

        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3, cal_dim=True, re_hall=2, add_coef=True, cv=1, random_state=1
                    )
        sl.fit(x, y, c=c,x_group=[[1, 3], [0, 2], [4, 7]])
        score = sl.score(x, y, "r2")
        print(sl.expr)

2.如果想更加需要定义功能细节（自定义特殊运算符，自定义运算符出现概率， 定义特征互影响概率）,
我们可以预先建立 ``SymbolSet`` 对象:py:doc:`bap.base.SymbolSet` 并传递给 ``SymbolLearning`` 的fit函数的pset参数。

此深度定制细节，请参考**base**部分和**flow**部分。
::

    if __name__ == "__main__":
        from sklearn.datasets import load_boston
        from bgp.skflow import SymbolLearning
        from bgp.base import SymbolSet

        data = load_boston()
        x = data["data"]
        y = data["target"]
        c = [1, 2, 3]

        pset0 = SymbolSet()
        pset0.add_features(x, y, x_dim=x_dim, y_dim=y_dim, x_group=[[1, 2], [3, 4], [5, 6]])
        pset0.add_constants(c, c_dim=c_dim, c_prob=None)
        pset0.add_operations(power_categories=(2, 3, 0.5),
                 categories=("Add", "Mul", "exp"),
                 special_prob =  {"Mul":0.5,"Add":0.4,"exp":0.1},
                 power_categories_prob = "balance"
                 )

        sl = SymbolLearning(loop="MultiMutateLoop", pop=500, gen=3, cal_dim=True, re_hall=2, add_coef=True, cv=1, random_state=1
                    )
        sl.fit(pset=pset0)
        score = sl.score(x, y, "r2")
        print(sl.expr)


更多样例参考：:doc:`../Examples/index`

SL参数
:::::::::::::::::

loop: str
    不同的符号回归循环方式（默认为'MultiMutateLoop'）.

    'BaseLoop'：基本循环，

    'MultiMutateLoop'：多变异方式循环，

    'OnePointMutateLoop'：单点变异循环，适合某固定长度公式的问题，

    'DimForceLoop'：量纲锁定循环，所有个体必须都有量纲。（请保证输入量纲有足够的能力计算出目标量纲）。

pop:int
    每一次迭代表达式数量。

gen:int
    迭代次数。

mutate_prob:float
    变异概率。

mate_prob:float
    交叉概率。

initial_max:int
    初始表达式大小上限。

initial_min : None,int
    初始表达式大小下限。

max_value:int
    最终表达式上限。

hall:int,>=1
    精英表达式个数（展示作用）。

re_hall:None or int>=2
    加入到下一代的精英表达式个数。当 “hall” 使用时可用。

re_Tree: int
    本次迭代的最好表达式当作新的特征添加到下一代的推荐个数。

personal_map:bool or "auto"
    互作用系数
    "auto" 根据表达式出现的特征自动调节特征的概率。

    True 使用等同的互作用概率。

    False 不使用互作用系数，使用独立的概率。

scoring: list of Callbale, default is [sklearn.metrics.r2_score,]
    sklearn.metrics评价函数，可以多个评价。

score_pen: tuple of  1, - 1 or float but 0.
    >0 : 求最大值问题, 下限为 - np.inf，（适合r2_score，accuracy等）

    <0 : 求最小值问题, 上限为 np.inf，（适合MAE,MSE等）

    Notes:
    如果采用多重评价，则必须预先将分值转换为相同的量级及正负，或者直接用score_pen表示权重。

    因为最终分值为均值 mean(w_i * score_i)

    Examples: [r2_score] is [1]。

cv:sklearn.model_selection._split._BaseKFold,int
    交叉验证(默认不使用 cv)。

    这里不建议打乱数据，建议预处理提前打乱数据。
filter_warning:bool
    是否过滤warning。

add_coef:bool
    是否添加系数。

inter_add：bool
    是否添加截距。

inner_add:bool
    是否添加公式内层系数。

out_add:bool
    是否添加公式外层系数。

flat_add:bool
    是否将公式全部展开添加系数。

vector_add:bool
    是否在绑定特征前添加系数。

n_jobs:int
    并行数。

batch_size:int
    并行分批数。

    数值根据机器性能调节。

random_state:int
    随机数。

cal_dim:bool
    是否计算量纲。

dim_type:Dim or None or list of Dim
    目标量纲过滤条件，由上到下逐渐严格。

    "coef": af(x)+b. a,b have dimension,f(x) is not dnan.
    默认系数自动补全量纲，只要f(x)能够被计算，均成立。

    "integer": af(x)+b. f(x) is interger dimension.
    f(x)量纲为整数。

    [Dim1,Dim2]: f(x) in list.
    f(x)量纲在目标列表内。

    Dim: f(x) ~= Dim. (see fuzzy)
    f(x)量纲为目标量纲的同底量纲。由fuzzy参数控制。

    Dim: f(x) == Dim.
    f(x)量纲为自定义的目标量纲。

    None: f(x) == pset.y_dim
    f(x)量纲为目标量纲。

fuzzy:bool
    f(x)量纲为目标量纲的同底量纲。例如 m,m^2,m^3。

stats:dict
    显示信息。

    values= {"max": np.max, "mean": np.mean, "min": np.mean, "std": np.std, "sum": np.sum}
    keys= {
    "fitness": just see fitness[0],
    "fitness_dim_max": max problem, see fitness with demand dim,
    "fitness_dim_min": min problem, see fitness with demand dim,
    "dim_is_target": demand dim,
    "coef":  dim is True, coef have dim,
    "integer":  dim is integer,}

     当 cal_dim=True,stats = {"fitness_dim_max": ("max",), "dim_is_target": ("sum",)}

     当 cal_dim=False,stats = {"fitness": ("max",)}

    keys可以被自定义，创建处理单个个体的函数。
    例如::

        def func(ind):
            return ind.fitness[0]
        stats = {func: ("mean",), "dim_is_target": ("sum",)}

verbose:bool
    是否打印显示信息。

tq:bool
    打印进度条。

store:bool or path
    是否存储（可输入存储位置的绝对路径）。

stop_condition:callable
    终止条件，可以被自定义，创建处理单个个体的函数。
    例如::

        def func(ind):
            c = ind.fitness.values[0]>=0.90
            return c

details:bool
    是否返回全部个体的预测值及表达式（打开会降低速度）。

classification: bool
    是否是分类问题。

pset:SymbolSet
    （默认为None）
    准备序列，用来预先自定义，设置特征X，X量纲，目标y，y量纲，运算符等。主要用于复杂功能设置，
    若为None, 默认使用fit方法自动建立简单的pset。


  
SL方法
:::::::::::

**fit**

X:np.ndarray
    输入数据。

y:np.ndarray
    目标值。

c:list of float
    常数项。

x_dim: 1 or list of Dim
    输入数据量纲。

y_dim: 1,Dim
    目标值量纲。

c_dim: 1,Dim
    常数量纲。

x_prob: None,list of float
    每个特征概率。

c_prob: None,list of float
    每个常数概率。

x_group:int, list of list
    绑定条件，默认不绑定，退化为普通GP问题。

    绑定方式可以直接定义分组大小：

    如：x_group=2

    绑定方式可以自定义分组：

    如：x_group=[[1,2][3,4]],为x1，x2绑定，为x3，x4绑定。

    See Also pset.add_features_and_constants

pset:SymbolSet
   （默认为None）

    准备序列，同初始化参数中的pset，并将其覆盖, 这里的再次输入可以用来做自定义的功能调整工作。

    若两处的pset均为None, 默认使用fit方法中的其他参数自动建立简单的pset。

    Note:
        如果给定pset，fit方法的其他参数无效，因为这些参数已经预先在pset中定义。
warm_start: bool
    是否热启动

    Note:
        如果用户预先提供pset，请仔细检查特性数目，特别是在使用“re_Tree”=True时。因为新的特征出现。

    参考:
        CalculatePrecisionSet.update_with_X_y
new_gen: None,int
    热启动迭代数.

SL属性
::::::::::::

loop
    所有循环细节内容，用来提取细节信息。

best_one: SymbolTree
    最好的表达式（SymbolTree对象）。

expr: sympy.Expr
    最好的表达式（sympy.Expr对象）。

y_dim: Dim
    最好的表达式的量纲。

fitness
    评分。


