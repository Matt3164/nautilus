from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.model.pipeline_model import PipelineModel
from nautilus.transform.dataset.to_array_dataset import DatasetToArrayDataset
from nautilus.transform.sample_transform_utils import SampleTrsUtils
from nautilus.transform.transform import Transform


class ToArrayContext(Transform):
    def __call__(self, ctx: TrainContext) -> TrainContext:
        assert isinstance(ctx.model, PipelineModel)

        array_extractor = DatasetToArrayDataset(transform=SampleTrsUtils.on_x(ctx.model.features_trs),
                                                subsample=ctx.config.subsample, )

        return TrainContext(
            dataset=TrainTestDataset(
                train_dataset=array_extractor(ctx.dataset.train),
                test_dataset = array_extractor(ctx.dataset.test)
            ),
            model=ctx.model,
            config=ctx.config

        )
