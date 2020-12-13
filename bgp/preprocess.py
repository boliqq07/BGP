import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array


def _ger_magnitude(a):
    c = 0
    if abs(a) > 1:
        while a >= 1:
            a /= 10
            c += 1
    elif 1 >= abs(a) > 0:
        while a <= 1:
            a *= 10
            c += 1
        c = -c

    return c


def _scale(a):
    return 10 ** _ger_magnitude(a)


class MagnitudeTransformer(TransformerMixin, BaseEstimator):
    """
    Transform x, y or c to near to 1, and store the transform Magnitude.
    """

    def __init__(self, standard=1, tolerate=0):
        self.standard = standard
        self.scale = None
        self.scale_y = None
        self.scale_c = None
        self.tolerate = tolerate

    def fit(self, X, y=None, group=2, apply=None, keep=None):
        """

        Parameters
        ----------
        X: np.ndarray

        y: np.ndarray

        group: group index of x

        apply: specific which index of x to transform

        keep: specific which index of x to not transform

        """

        X = X.astype(np.float32)
        X = check_array(X, ensure_2d=True)
        n = X.shape[1]
        assert isinstance(X, np.ndarray)
        means = np.mean(X, axis=0)

        if not group:
            pass
        else:
            if isinstance(group, int):
                assert n > group > 1, "the len of group should in (2,x.shape[1]]"
                indexes = [_ for _ in range(n)]
                group = [indexes[i:i + group] for i in range(0, len(indexes), group)]
            for g in group:
                if len(g) > 0:
                    means[g] = np.mean(means[g])

        scale = np.array([_scale(i) for i in means])

        if self.tolerate:
            scale = np.array([i if not 10 ** (-self.tolerate) <= i <= 10 ** self.tolerate else 1 for i in scale])
        scale = scale.astype(np.float32)
        scale /= self.standard

        if apply is not None:
            if isinstance(apply, int):
                apply = [apply, ]
            li = list(range(n))
            keep = list(set(li) - set(apply))
            keep.sort()

        if keep is not None:
            if isinstance(keep, int):
                keep = [keep, ]
            for i in keep:
                scale[i] = 1

        self.scale = scale

        if y is not None:
            y = y.astype(np.float32)
            y = check_array(y, ensure_2d=False)
            assert isinstance(y, np.ndarray)
            means = np.mean(y)
            scale = _scale(means)
            if self.tolerate:
                scale = scale if not 10 ** (-self.tolerate) <= scale <= 10 ** self.tolerate else 1
            scale /= self.standard
            self.scale_y = scale
        return self

    def transform(self, X):
        if self.scale is None:
            raise NotImplementedError("the method should be fitted first.")
        else:
            return X / self.scale

    def inverse_transform(self, X):
        if self.scale_y is None:
            raise NotImplementedError("the method should be fitted first.")
        else:
            return self.scale_y * X

    def transform_y(self, y):
        if self.scale_y is None:
            raise NotImplementedError("the method should be fitted y first.")
        else:
            return y / self.scale_y

    def inverse_transform_y(self, y):
        if self.scale_y is None:
            raise NotImplementedError("the method should be fitted y first.")
        else:
            return self.scale_y * y

    def fit_transform_all(self, X, y, **fit_params):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            self.fit(X, y, **fit_params)
            return self.transform(X), self.transform_y(y)

    def fit_constant(self, c):
        if isinstance(c, float):
            c = [c, ]
        scale_c = []
        for ci in c:
            scale = _scale(ci)
            if self.tolerate:
                scale = scale if not 10 ** (-self.tolerate) <= scale <= 10 ** self.tolerate else 1
            scale /= self.standard
            scale_c.append(float(scale))
        scale_c = np.array(scale_c)
        scale_c.astype(np.float32)
        self.scale_c = scale_c
        return self

    def transform_constant(self, c):
        if self.scale_y is None:
            raise NotImplementedError("the method should be fitted c first.")
        if isinstance(c, float):
            c = [c, ]
        c = np.array([float(i) for i in c])
        c.astype(np.float32)
        return c / self.scale_c

    def inverse_transform_constant(self, c):
        if self.scale_y is None:
            raise NotImplementedError("the method should be fitted c first.")
        if isinstance(c, float):
            c = [c, ]
        c = np.array(c)
        c.astype(np.float32)
        return c * self.scale_c

    def fit_transform_constant(self, c):
        return self.fit_constant(c).transform_constant(c)
