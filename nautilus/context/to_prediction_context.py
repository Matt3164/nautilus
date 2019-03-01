from sklearn.base import BaseEstimator

from nautilus.context.prediction_context import BufferPredictionCtx
from nautilus.dataset.dataset import Dataset
from nautilus.transform.transform import Transform


class ToPredictionContext(Transform):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def __call__(self, dataset: Dataset)-> BufferPredictionCtx:

        Ytrue = dataset.Y.reshape(-1, 1)

        Ypred = self.model.predict_proba(dataset.X)

        return BufferPredictionCtx(
            y_true=Ytrue,
            y_pred=Ypred.argmax(axis=1),
            y_pred_prob=Ypred
        )


