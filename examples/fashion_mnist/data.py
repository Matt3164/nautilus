from keras.datasets import fashion_mnist

from nautilus.data.dataset.train_test_dataset import TrainTestDataset
from nautilus.data.dataset.utils import DatasetUtils


class FashionMnistLoader(object):
    """"""

    @staticmethod
    def dataset():
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return TrainTestDataset(
            train_dataset=DatasetUtils.from_xy_array(
                x_train,
                y_train.reshape(-1, 1)
            ),
            test_dataset=DatasetUtils.from_xy_array(
                x_test,
                y_test.reshape(-1, 1)
            )
        )

if __name__ == '__main__':
    print(fashion_mnist.load_data()[0][0].shape)