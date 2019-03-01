from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin

from nautilus.utils import np_utils


class TransformerFromFeature(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self, func: Callable[[np_utils.ndarray], np_utils.ndarray]):
        """Constructor for TransformerFromFunc"""
        self.func = func

    def fit(self, X, y=None):
        pass

    def transform(self, X, y='deprecated'):
        return np_utils.map_arr(X, self.func)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)