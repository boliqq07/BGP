import numpy as np


def np_map():
    """user's sympy.expr to np.ndarray function"""

    def Flat(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.sum(x, axis=0)
            else:
                return x
        else:
            return x

    def Comp(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                return np.prod(x, axis=0)
            else:
                return x
        else:
            return x

    def Diff(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return x[0] - x[1]
                else:
                    return x
            else:
                return x
        else:
            return x

    def Quot(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return x[0] / x[1]
                else:
                    return x
            else:
                return x
        else:
            return x

    def Conv(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                if x.shape[0] == 2:
                    return np.array((x[1], x[0]))
                else:
                    return x
            else:
                return x
        else:
            return x

    return {"MAdd": Flat, "MMul": Comp, "MSub": Diff, "MDiv": Quot, "Conv": Conv,
            "Self": lambda x_: x_}
