
import logging

from nautilus.train.trainer.sklearn import SkBufferTrainer
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

from nautilus.config.train_config import TrainingConfig
from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.array_dataset import ArrayDataset
from nautilus.data.dataset.utils import DatasetUtils
from nautilus.model.sklearn.sklearn_model import SkModel
from nautilus.transform.sample_transform_utils import SampleTrsUtils
from nautilus.transform.tensor.flatten import Flatten

logging.basicConfig(level=logging.INFO)

X, Y = load_digits()["data"], load_digits()["target"]

dataset = ArrayDataset(X.reshape(-1, 8, 8))

print(dataset[0].shape)

dataset = DatasetUtils.from_xy_array(
    X.reshape(-1, 8, 8),
    Y.reshape(-1, 1)
)

dataset = DatasetUtils.from_dataset_and_trs(
    dataset,
    SampleTrsUtils.on_x(Flatten()),
)

print(dataset[0].x.shape)

# loader = Loader(
#     dataset,
#     SampleTrsUtils.on_x(Flatten()),
#     batch_size=8
# )
#
# for a in loader:
#     print(a[0].shape)
#     print(a[1].shape)
#
#

rf = RandomForestClassifier(max_depth=1, n_estimators=25)

model = SkModel.from_sklearn(rf)

config = TrainingConfig()

context = TrainContext(
    dataset,
    model,
    config
)

SkBufferTrainer()(context)