from nautilus.config.partial_trainer_config import PartialTrainerConfig
from nautilus.context.train_context import TrainContext
from nautilus.model.pipeline_model import PipelineModel
from nautilus.train.loader.loader import Loader
from nautilus.train.trainer.trainer import Trainer
from nautilus.transform.sample_transform_utils import SampleTrsUtils


class PartialFitTrainer(Trainer):
    """"""

    def __call__(self, train_ctx: TrainContext) -> TrainContext:
        assert isinstance(train_ctx.config, PartialTrainerConfig)
        model = train_ctx.model
        assert isinstance(model, PipelineModel)


        loader = Loader(train_ctx.dataset.train, SampleTrsUtils.on_x(model.features_trs),
                        batch_size=train_ctx.config.batch_size,
                        n_epoch=train_ctx.config.n_epoch
                        )



        for x_batch, y_batch in loader:
            train_ctx.model.model.estimator.fit_on_batch(x_batch, y_batch, train_ctx.config.classes)

        return train_ctx





