from sklearn.base import BaseEstimator, TransformerMixin


class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        self.model.fit(X, y)

    def transform(self, X, y=None):
        return self.model.predict_proba(X)