import mlflow
from numpy.core._multiarray_umath import ndarray
from sklearn.metrics import accuracy_score

from nautilus.metrics.buffermetric import BufferMetric


class BufferAccuracy(BufferMetric):
    def _metric(self, y_true: ndarray, y_pred: ndarray):
        acc = accuracy_score(y_true, y_pred)

        mlflow.log_metric(type(self).__name__, acc)

        return acc

