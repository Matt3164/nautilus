from itertools import chain
from typing import List, Tuple, Iterator, Callable

from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import logging

from sklearn.feature_extraction.image import PatchExtractor, extract_patches
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from examples.fashion_mnist.data import FashionMnistLoader
from nautilus.dataset import dataset_utils
from nautilus.dataset.dataset import Dataset
from nautilus.experiment.experiment import Experiment
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.transform.sequential import Sequential
from nautilus.utils import image_utils, np_utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("shape_resized", 8, "Image shape after resize")
flags.DEFINE_integer("n_jobs", 1, "Number of parallel jobs")
flags.DEFINE_integer("n_estimators", 25, "Number of decision trees to learn")

logging.basicConfig(level=logging.INFO)


def compute_bow(arr: ndarray, model: BaseEstimator, extractor: Callable[[
                                                                            ndarray], ndarray])->ndarray:
    extracted_arr = extractor(arr)

    n_patch_i, n_patch_j, _, _, = extracted_arr.shape

    reshaped_arr = extracted_arr.reshape(n_patch_i*n_patch_j, -1)

    features = model.transform(reshaped_arr)

    return features.flatten()


def main(_):
    dataset = FashionMnistLoader().dataset()

    extractor = PatchExtractor(patch_size=(14,14), max_patches=10).transform

    kmeans = KMeans(n_clusters=100, n_init=1, max_iter=50)
    pca = PCA(n_components=8)
    km_pipeline = Pipeline(steps=[('pca', pca), ('kmeans', kmeans)])

    km_exp = Experiment(
        train_dataset_fn=lambda : dataset_utils.map_x(dataset_utils.on_x(
            dataset.train.X, extractor),
            np_utils.flatten),
        test_dataset_fn=lambda : dataset_utils.map_x(dataset_utils.on_x(
            dataset.test.X, extractor),
            np_utils.flatten),
        model=km_pipeline,
        exp_tag="pca_km",
        metrics=[]
    )

    km_exp.run()

    extractor = lambda arr: extract_patches(arr, patch_shape=(14,14),
                                            extraction_step=(7,7))

    feature_extractor = lambda arr: compute_bow(arr, km_exp.model, extractor)

    rf = RandomForestClassifier(max_depth=5, n_estimators=25)
    rf_pipeline = Pipeline(steps=[('normalizer', Normalizer()), ('rf', rf)])

    rf_exp = Experiment(
        train_dataset_fn=lambda : dataset_utils.map_x(
            dataset.train, feature_extractor),
        test_dataset_fn=lambda : dataset_utils.map_x(
            dataset.test, feature_extractor),
        model=rf_pipeline,
        exp_tag="bow_compute",
        metrics=[
            BufferedConfusionMatrix(),
            BufferedClassificationReport()
        ]
    )

    rf_exp.run()



if __name__ == '__main__':
    app.run(main)