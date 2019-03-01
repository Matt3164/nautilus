from itertools import chain
from typing import List, Tuple, Iterator, Callable

from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
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
from nautilus.transformer.from_feature import TransformerFromFeature
from nautilus.utils import image_utils, np_utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("shape_resized", 8, "Image shape after resize")
flags.DEFINE_integer("n_jobs", 1, "Number of parallel jobs")
flags.DEFINE_integer("n_estimators", 25, "Number of decision trees to learn")

logging.basicConfig(level=logging.INFO)

class BagOfWordExtractor(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self,
                 model,
                 patch_size: int,
                 patch_stride: int,
                 max_patches: int
                 ):
        """Constructor for BagOfWordExtractor"""
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.max_patches = max_patches

        self.model = model

    def fit(self, X, y=None):
        extractor = PatchExtractor(patch_size=(self.patch_size, self.patch_size),
                                   max_patches=self.max_patches)

        nX = extractor.transform(X)

        self.model.fit(nX)

    def transform(self, X, y=None):
        return np_utils.map_arr(X, self.compute_bow)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


    def compute_bow(self, arr: ndarray)->ndarray:
        extracted_arr = extract_patches(arr, (self.patch_size,
                                              self.patch_size),
                                        (self.patch_stride, self.patch_stride))

        n_patch_i, n_patch_j, _, _, = extracted_arr.shape

        reshaped_arr = extracted_arr.reshape(n_patch_i*n_patch_j, -1)

        features = self.model.transform(reshaped_arr)

        return features.flatten()


def main(_):
    dataset = FashionMnistLoader().dataset()

    kmeans = KMeans(n_clusters=100, n_init=1, max_iter=50)
    pca = PCA(n_components=8)
    km_pipeline = Pipeline(steps=[("flattener", TransformerFromFeature(np_utils.flatten)), ('pca', pca), ('kmeans',
                                                              kmeans)])

    rf = RandomForestClassifier(max_depth=5, n_estimators=25)
    rf_pipeline = Pipeline(steps=[
        ('bow', BagOfWordExtractor(model=km_pipeline, patch_size=14,
                                   patch_stride=7, max_patches=10)),
        ('normalizer', Normalizer()),
        ('rf', rf)])

    rf_exp = Experiment(
        train_dataset_fn=lambda : dataset.train,
        test_dataset_fn=lambda : dataset.test,
        model=rf_pipeline,
        exp_tag="bow_compute",
        metrics=[
            BufferedConfusionMatrix(),
            BufferedClassificationReport()
        ],
        use_cache=False
    )

    rf_exp.run()



if __name__ == '__main__':
    app.run(main)