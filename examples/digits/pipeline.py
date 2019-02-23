
import logging
from functools import partial

from nautilus.config.sk_config import BufferConfig
from nautilus.data.dataset.utils import DatasetUtils
from nautilus.experiment.trainer.sklearn import SkBufferTrainer
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.sample_transform_utils import SampleTrsUtils
from nautilus.transform.tensor.flatten import Flatten
from sklearn.ensemble import RandomForestClassifier

from examples.digits.digits_loader import DigitsLoader
from nautilus.context.to_prediction_context import ToPredictionContext
from nautilus.context.train_context import TrainContext
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix

logging.basicConfig(level=logging.INFO)

class ShapesConfig:

    model = SkModel.from_sklearn(
        RandomForestClassifier(max_depth=1, n_estimators=25)
    )

    config = BufferConfig(train_size=0.8, test_size=0.2)

def to_flatten_dataset(dataset):
    return DatasetUtils.traintest_apply(
        dataset,
        SampleTrsUtils.on_x(Flatten()),
    )



datasets = map(lambda x: x.dataset(), [DigitsLoader()])
datasets = map(to_flatten_dataset, datasets)
train_contexts = map(partial(TrainContext, model=ShapesConfig.model, config=ShapesConfig.config), datasets)
train_contexts = map(SkBufferTrainer(), train_contexts)
prediction_contexts = map(ToPredictionContext(), train_contexts)
prediction_contexts = map(BufferedConfusionMatrix(), prediction_contexts)
prediction_contexts = map(BufferedClassificationReport(), prediction_contexts)
prediction_contexts = list(prediction_contexts)