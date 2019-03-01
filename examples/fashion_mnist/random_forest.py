from sklearn.ensemble import RandomForestClassifier
import logging
from examples.fashion_mnist.data import FashionMnistLoader
from nautilus.dataset import dataset_utils
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

def main(_):
    dataset = FashionMnistLoader().dataset()

    feature_computer = Sequential.from_transforms(
        image_utils.cvresize_fn((FLAGS.shape_resized, FLAGS.shape_resized)),
        np_utils.flatten
    )

    # Random forest

    rf = RandomForestClassifier(max_depth=None, n_estimators=FLAGS.n_estimators,
                                bootstrap=True, n_jobs=FLAGS.n_jobs)

    experiment = Experiment(
        train_dataset_fn=lambda : dataset_utils.map_x(dataset.train,
                                                      feature_computer),
        test_dataset_fn=lambda : dataset_utils.map_x(dataset.test,
                                                      feature_computer),
        model=rf,
        exp_tag="rf_exp",
        metrics=[BufferedConfusionMatrix(), BufferedClassificationReport()]
    )

    experiment.run()

if __name__ == '__main__':
    app.run(main)