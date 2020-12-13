备注
=============

**补充说明1**：

量纲计算，当数据应有单位，但是您主观不使用量纲计算，这种情况下，模型完全可以进行，即假定数据为是完全的统一好量纲的数据。

**补充说明2**：

量纲计算，都是从不同的单位转换到标准单位进行计算，这其中一定会有放缩系数出现。

1. 我们建议在处理数据时，把数据处理成标准单位（量纲基准）所对应的数据值，即放缩系数乘到数据中去。

2. 我们也集成了单位到量纲的转换工具，从sympy中的单位转换而来的。

（使用 Dim.convert_x, Dim.convert_xi, Dim.convert_x 分别转换 x，y，c并得到转换后的数据以及量纲)

无论哪种情况，最终结果的公式形式正确，但是系数值是并非对应您的初始数据（而是标准单位对应的数据）。可能需要您手动重新拟合系数值。

**补充说明3**：

量纲计算，部分单位过小，或者过大，数据在乘上放缩系数之后会特别小或者特别大，如10^16。

如果使用我们提供的 MagnitudeTransformer 进行对数据预处理，这种情况下也需要您手动重新拟合系数值。

**补充说明4**：

系数单位，我们没有提供系数值的单位，默认系数单位为以补全方式计算。

例如单位量纲转换：

::

    from sympy.physics.units import kg, m
    from bgp.functions.dimfunc import Dim, dless

    x_u = [kg] * 12 + m
    y_u = kg
    c_u = [dless, dless, dless]

    # Dim    the dim also could get by Dim(numpy.array([****])) directly.
    x, x_dim = Dim.convert_x(x, x_u, target_units=None, unit_system="SI")
    y, y_dim = Dim.convert_xi(y, y_u)
    c, c_dim = Dim.convert_x(c, c_u)
