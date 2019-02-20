import logging

from numpy.core.multiarray import ndarray

from nautilus.context.prediction_context import BufferPredictionContext
from nautilus.transform.transform import Transform
logger=logging.getLogger(__name__)

class BufferMetric(Transform):
    """"""

    def _metric(self, y_true: ndarray, y_pred:ndarray):
        raise NotImplementedError

    def __call__(self, data: BufferPredictionContext)->BufferPredictionContext:

        logging.info(">>>> TRAIN")

        self._metric(data.train.y_true, data.train.y_pred)

        logging.info(">>>> TEST")

        self._metric(data.test.y_true, data.test.y_pred)

        return data




