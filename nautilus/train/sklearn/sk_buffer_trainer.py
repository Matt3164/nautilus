import logging

from numpy.core.multiarray import ndarray

from nautilus.config.buffer_config import BufferConfig
from nautilus.model.model import Model
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.train.sklearn.buffer_trainer import BufferTrainer

logger = logging.getLogger(__name__)


class SkBufferTrainer(BufferTrainer):
    """"""

    def fit_on_buffer(self, x_buffer: ndarray, y_buffer: ndarray, model: Model, config: BufferConfig) -> Model:
        assert isinstance(model, SkModel)

        model.estimator.fit(x_buffer, y_buffer.ravel())

        return model
