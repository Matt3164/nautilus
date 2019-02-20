"""

Not currently working because of low contrast

"""


import logging
import os
from examples.fashion_mnist.config.fashion_config import MnistConfig
from examples.fashion_mnist.data import FashionMnistLoader
from examples.fashion_mnist.features.orb_flat_map import OrbFlatMap
from examples.fashion_mnist.model.bag_of_words import BagOfWords
from examples.fashion_mnist.utils import cached_array_context
from nautilus.config.buffer_config import BufferConfig
from nautilus.context.train_context import TrainContext
from nautilus.data.dataset.dataset import Dataset
from nautilus.data.dataset.flat_map_dataset import FlatMapDataset
from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.model.pipeline_model import PipelineModel
from nautilus.train.sklearn.sk_buffer_trainer import SkBufferTrainer
from nautilus.transform.context.to_prediction_context import ToPredictionContext
from nautilus.transform.sample.classiftiler import ClassifTiler, TileParameter
from nautilus.transform.sklearn.model.sk_model_to_path import SkModelToPath
from nautilus.transform.sklearn.path.path_to_sk_model import PathToSkModel

logging.basicConfig(level=logging.INFO)


def create_fm_dataset(dataset: Dataset):
    return FlatMapDataset( dataset, trs=OrbFlatMap(n_keypoints=100))

def apply_orb_flatmap(traintest_dataset: TrainTestDataset)->TrainTestDataset:
    return TrainTestDataset(
        train_dataset=create_fm_dataset(traintest_dataset.train),
        test_dataset=create_fm_dataset(traintest_dataset.test),
    )

# Bow model on ORB descriptors --> Not working on Fashion MNIST because of low contrast
# Tiler / Tile parameter
# Model to learn on it / codebook_model

if not os.path.exists(MnistConfig.orb_km_path):
    dataset = FashionMnistLoader().dataset()
    dataset = apply_orb_flatmap(dataset)
    train_context = TrainContext(dataset=dataset, model=MnistConfig.BOW_ORB_KM, config=BufferConfig(subsample=0.1))
    train_context = cached_array_context(train_context)
    train_context = SkBufferTrainer()(train_context)
    SkModelToPath(MnistConfig.orb_km_path)(train_context.model)

# Model to learn after all

# if not os.path.exists(MnistConfig.bow_model_path):
#     model = PathToSkModel()(MnistConfig.model_path)
#
#     dataset = FashionMnistLoader().dataset()
#
#     bow = BagOfWords(model=model, tiler=ClassifTiler(tile_parameter=TileParameter(tile_size=14, stride=3)), n_max_features=100)
#
#     bow_model = PipelineModel(
#         MnistConfig.BOW_MODEL,
#         in_trs=bow,
#     )
#
#     train_context = TrainContext(dataset=dataset, model=bow_model, config=BufferConfig(subsample=None))
#     train_context = cached_array_context(train_context)
#     train_context = SkBufferTrainer()(train_context)
#
#     SkModelToPath(MnistConfig.bow_model_path)(train_context.model)
#
# # Show metric
#
# bow_model = PathToSkModel()(MnistConfig.bow_model_path)
# dataset = FashionMnistLoader().dataset()
# train_context = TrainContext(dataset=dataset, model=bow_model, config=BufferConfig(subsample=None))
# train_context = cached_array_context(train_context)
# prediction_contexts = ToPredictionContext()(train_context)
# prediction_contexts = BufferedConfusionMatrix()(prediction_contexts)
# prediction_contexts = BufferedClassificationReport()(prediction_contexts)