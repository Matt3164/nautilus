
import logging

from examples.fashion_mnist.config.fashion_config import MnistConfig
from examples.fashion_mnist.data import FashionMnistLoader
from examples.fashion_mnist.utils import cached_array_context
from nautilus.context.train_context import TrainContext
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.train.sklearn.sk_buffer_trainer import SkBufferTrainer
from nautilus.transform.context.to_prediction_context import ToPredictionContext

logging.basicConfig(level=logging.INFO)

dataset = FashionMnistLoader().dataset()

# PCA and Nearest Neighbours

train_context = TrainContext(dataset=dataset, model=MnistConfig.PCA_NN, config=MnistConfig.config)
train_context = cached_array_context(train_context)
train_context = SkBufferTrainer()(train_context)
prediction_context = ToPredictionContext()(train_context)
prediction_context = BufferedConfusionMatrix()(prediction_context)
prediction_context = BufferedClassificationReport()(prediction_context)

# Random forest and LBP

train_context = TrainContext(dataset=dataset, model=MnistConfig.RF_LBP, config=MnistConfig.config)
train_context = cached_array_context(train_context)
train_context = SkBufferTrainer()(train_context)
prediction_context = ToPredictionContext()(train_context)
prediction_context = BufferedConfusionMatrix()(prediction_context)
prediction_context = BufferedClassificationReport()(prediction_context)

