from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from nautilus.data.dataset.sample_dataset import SampleDataset
from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.data.dataset.utils import DatasetUtils


class DigitsLoader(object):
    """"""

    def dataset(self)->TrainTestDataset:
        X, Y = load_digits()["data"], load_digits()["target"]

        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

        return TrainTestDataset(
            train_dataset=DatasetUtils.from_xy_array(
            xtrain.reshape(-1, 8, 8),
            ytrain.reshape(-1, 1)
            ),
            test_dataset=DatasetUtils.from_xy_array(
            xtest.reshape(-1, 8, 8),
            ytest.reshape(-1, 1)
            ),
        )
