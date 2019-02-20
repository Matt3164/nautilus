from numpy.core.multiarray import ndarray

from nautilus.config.buffer_config import BufferConfig
from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.sample.array_sample_dataset import XYSampleDataset
from nautilus.model.model import Model
from nautilus.model.pipeline_model import PipelineModel
from nautilus.train.trainer.trainer import Trainer


class BufferTrainer(Trainer):
    """"""
    
    def fit_on_buffer(self, x_buffer: ndarray, y_buffer: ndarray, model: Model, config: BufferConfig)->Model:
        raise NotImplementedError

    def __call__(self, train_ctx: TrainContext)->TrainContext:

        model = train_ctx.model

        assert isinstance(model, PipelineModel)

        assert isinstance(train_ctx.dataset.train, XYSampleDataset)

        self.fit_on_buffer(train_ctx.dataset.train.x_data.array, train_ctx.dataset.train.y_data.array, train_ctx.model.model, train_ctx.config)

        return TrainContext(
            train_ctx.dataset,
            train_ctx.model,
            train_ctx.config

        )
        
        



