import logging

from examples.fashion_mnist.config.fashion_config import MnistConfig
from examples.fashion_mnist.data import FashionMnistLoader
from nautilus.context.train_context import TrainContext
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.train.trainer import PartialFitTrainer
from nautilus.transform.context.to_prediction_context import ToPredictionContext

logging.basicConfig(level=logging.INFO)

dataset = FashionMnistLoader().dataset()
train_context = TrainContext(dataset=dataset, model=MnistConfig.PARTIAL_RF, config=MnistConfig.partial_config)
train_context = PartialFitTrainer()(train_context)
prediction_context = ToPredictionContext()(train_context)
prediction_context = BufferedConfusionMatrix()(prediction_context)
prediction_context = BufferedClassificationReport()(prediction_context)
