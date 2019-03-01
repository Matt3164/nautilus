import logging
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from examples.fashion_mnist.data import FashionMnistLoader
from nautilus.dataset import dataset_utils
from nautilus.experiment.experiment import Experiment
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.transformer.from_feature import TransformerFromFeature
from nautilus.utils import image_utils, np_utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("shape_resized", 8, "Image shape after resize")
flags.DEFINE_integer("pca_components", 8, "PCA Components")
flags.DEFINE_integer("k", 5, "k of nearest neighbours")
flags.DEFINE_integer("n_jobs", 1, "Number of parallel jobs")

logging.basicConfig(level=logging.INFO)

def main(_):
    dataset = FashionMnistLoader().dataset()

    pca = PCA(n_components=FLAGS.pca_components)
    nn = KNeighborsClassifier(n_neighbors=FLAGS.k, n_jobs=FLAGS.n_jobs)

    pipeline = Pipeline(steps=[
        ("resizer", TransformerFromFeature(image_utils.cvresize_fn((
        FLAGS.shape_resized,FLAGS.shape_resized)))),
        ("flattener", TransformerFromFeature(np_utils.flatten)),
        ('pca', pca), ('nn', nn)])

    experiment = Experiment(
        train_dataset_fn=lambda : dataset.train,
        test_dataset_fn=lambda : dataset.test,
        model=pipeline,
        exp_tag="pca_nn_exp",
        metrics=[BufferedConfusionMatrix(), BufferedClassificationReport()],
        use_cache=False
    )

    experiment.run()

if __name__ == '__main__':
    app.run(main)