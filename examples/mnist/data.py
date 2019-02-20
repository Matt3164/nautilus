from keras.datasets import mnist

from nautilus.data.dataset.utils import DatasetUtils


class MnistLoader(object):
    """"""

    @staticmethod
    def dataset():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return DatasetUtils.from_xy_array(
            x_train,
            y_train.reshape(-1, 1)
        )

if __name__ == '__main__':
    print(mnist.load_data())