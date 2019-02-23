from keras.datasets import fashion_mnist

from nautilus.dataset.dataset import Dataset
from nautilus.dataset.train_test_dataset import TrainTestDataset


class FashionMnistLoader(object):
    """"""

    @staticmethod
    def dataset():
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return TrainTestDataset(
            train_dataset=Dataset.from_arrays(
                x_train,
                y_train.reshape(-1, 1)
            ),
            test_dataset=Dataset.from_arrays(
                x_test,
                y_test.reshape(-1, 1)
            )
        )

if __name__ == '__main__':
    print(fashion_mnist.load_data()[0][0].shape)