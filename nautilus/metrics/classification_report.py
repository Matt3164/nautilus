import tempfile

import mlflow
import yaml
from numpy.core.multiarray import ndarray
from sklearn.metrics import classification_report
import logging

from nautilus.metrics.buffermetric import BufferMetric

logger=logging.getLogger(__name__)

class BufferedClassificationReport(BufferMetric):
    """"""

    def _metric(self, y_true: ndarray, y_pred: ndarray):

        logger.info("Confusion matrix: ")
        logger.info(
            classification_report(y_true, y_pred)
        )

        # Mlflow logging

        fn = tempfile.mktemp(suffix=".txt")
        with open(fn, "w") as f:
            yaml.dump(
                classification_report(y_true, y_pred, output_dict=True),
                f
            )
        mlflow.log_artifact(fn)

        return None


