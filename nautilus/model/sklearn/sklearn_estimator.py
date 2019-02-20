from numpy.core._multiarray_umath import ndarray
from sklearn.base import BaseEstimator


class SkEstimator(BaseEstimator):
    """"""

    def __init__(self,
                 sk_estimator: BaseEstimator):
        """Constructor for SkEstimator"""
        self.sk_extimator = sk_estimator

    def fit(self, x: ndarray, y: ndarray):
        self.sk_extimator.fit(x, y)

    def predict_proba(self, x: ndarray)->ndarray:

        if hasattr(self.sk_extimator, "predict_proba"):
            return self.sk_extimator.predict_proba(x)
        elif hasattr(self.sk_extimator, "predict"):
            return self.sk_extimator.predict(x).reshape(-1, 1)
        else:
            raise NotImplementedError

    def transform(self, x: ndarray):
        return self.predict_proba(x)