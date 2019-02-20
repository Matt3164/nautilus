
import logging
from functools import partial

from nautilus.train.trainer.sklearn import SkBufferTrainer
from sklearn.ensemble import RandomForestClassifier

from examples.mnist.data import MnistLoader
from nautilus.config.train_config import TrainingConfig
from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.dataset.utils import DatasetUtils
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.sample_transform_utils import SampleTrsUtils
from nautilus.transform.tensor.flatten import Flatten

logging.basicConfig(level=logging.INFO)

class MnistConfig:

    model = SkModel.from_sklearn(
        RandomForestClassifier(max_depth=1, n_estimators=25)
    )

    config = TrainingConfig()

def to_flatten_dataset(dataset: SampleDataset):
    return DatasetUtils.from_dataset_and_trs(
        dataset,
        SampleTrsUtils.on_x(Flatten()),
    )



datasets = map(lambda x: x.dataset(), [MnistLoader()])
datasets = map(to_flatten_dataset, datasets)
train_contexts = map(partial(TrainContext, model=MnistConfig.model, config=MnistConfig.config), datasets)
train_contexts = map(SkBufferTrainer(), train_contexts)
train_contexts = list(train_contexts)