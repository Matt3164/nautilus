from nautilus.model.partial_sk_ensemble import PartialSkEnsemble
from numpy.core.multiarray import ndarray

from nautilus.model.model import Model
from nautilus.model.partial_estimator import PartialSkEstimator
from nautilus.model.sklearn.sklearn_estimator import SkEstimator


class SkModel(Model):
    """"""

    def __init__(self,
                 estimator: SkEstimator):
        """Constructor for SkModel"""
        self.estimator = estimator

    def __call__(self, tensor: ndarray) -> ndarray:
        return self.estimator.predict_proba(tensor)

    @staticmethod
    def from_sklearn(model):
        return SkModel(
            SkEstimator(model)
        )

    @staticmethod
    def from_partial_sklearn(model):
        return SkModel(
            PartialSkEstimator(model)
        )

    @staticmethod
    def from_partial_sklearn_ensemble(model):
        model.set_params(warm_start=True)
        model.set_params(n_estimators=1)
        return SkModel(
            PartialSkEnsemble(model)
        )


