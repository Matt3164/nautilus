from sklearn.ensemble import RandomForestClassifier
import logging

from sklearn.pipeline import make_pipeline

from examples.fashion_mnist.data import FashionMnistLoader
from nautilus.experiment.experiment import Experiment
from nautilus.metrics.classification_report import BufferedClassificationReport
from nautilus.metrics.confusion_matrix import BufferedConfusionMatrix
from nautilus.transformer.from_feature import TransformerFromFeature
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

    # Random forest

    rf = RandomForestClassifier(max_depth=None, n_estimators=FLAGS.n_estimators,
                                bootstrap=True, n_jobs=FLAGS.n_jobs)

    pipeline = make_pipeline(
        TransformerFromFeature(image_utils.cvresize_fn((FLAGS.shape_resized,
                                                        FLAGS.shape_resized))),
        TransformerFromFeature(np_utils.flatten),
        rf
    )

    experiment = Experiment(
        train_dataset_fn=lambda : dataset.train,
        test_dataset_fn=lambda : dataset.test,
        model=pipeline,
        exp_tag="rf_exp",
        metrics=[BufferedConfusionMatrix(), BufferedClassificationReport()],
        use_cache=False
    )

    experiment.run()

if __name__ == '__main__':
    app.run(main)