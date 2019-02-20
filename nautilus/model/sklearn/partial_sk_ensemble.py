from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator
from nautilus.model.partial_estimator import PartialSkEstimator

class PartialSkEnsemble(PartialSkEstimator):
    """"""

    def __init__(self, sk_estimator: BaseEstimator):
        super().__init__(sk_estimator)

    def fit_on_batch(self, x: ndarray, y: ndarray, classes=list()):
        self.sk_extimator.set_params(n_estimators=self.sk_extimator.n_estimators+1)
        self.sk_extimator.fit(x, y.ravel())

