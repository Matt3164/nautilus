import logging

from numpy.core.multiarray import ndarray

from nautilus.context.prediction_context import BufferPredictionCtx
from nautilus.transform.transform import Transform
logger=logging.getLogger(__name__)

class BufferMetric(Transform):
    """"""

    def _metric(self, y_true: ndarray, y_pred:ndarray):
        raise NotImplementedError

    def __call__(self, data: BufferPredictionCtx):

        metric = self._metric(data.y_true, data.y_pred)

        return metric




