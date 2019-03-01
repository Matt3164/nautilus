import cv2
from numpy.core.multiarray import ndarray
import numpy as np
from nautilus.transform.transform import Transform


class ImStats(Transform):
    """"""

    def __call__(self, data: ndarray):
        (means, stds) = cv2.meanStdDev(data)
        return np.concatenate([means, stds]).flatten()


if __name__ == '__main__':
    print(ImStats()(np.random.randint(0, 255, size=(256, 256, 3))).shape)