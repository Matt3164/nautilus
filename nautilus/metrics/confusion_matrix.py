import tempfile

import mlflow
import yaml
from numpy.core.multiarray import ndarray
from sklearn.metrics import confusion_matrix
import logging

from nautilus.metrics.buffermetric import BufferMetric

logger=logging.getLogger(__name__)

class BufferedConfusionMatrix(BufferMetric):
    """"""

    def _metric(self, y_true: ndarray, y_pred: ndarray):

        logger.info("Confusion matrix: ")
        cm = confusion_matrix(y_true, y_pred)
        logger.info(
            cm
        )

        # Mlflow logging

        fn = tempfile.mktemp(suffix=".txt")
        with open(fn, "w") as f:
            yaml.dump(
                cm.tolist(),
                f
            )
        mlflow.log_artifact(fn)

        return cm


