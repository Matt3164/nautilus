from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator

from nautilus.model.sklearn.sklearn_estimator import SkEstimator


class PartialSkEstimator(SkEstimator):
    """"""

    def __init__(self, sk_estimator: BaseEstimator):
        super().__init__(sk_estimator)

    def fit_on_batch(self, x: ndarray, y: ndarray, classes=list()):
        self.sk_extimator.partial_fit(x, y.ravel(), classes=classes)

