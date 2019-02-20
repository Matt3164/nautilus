from numpy.core.multiarray import ndarray

from nautilus.transform.transform import Transform


class Flatten(Transform):
    """"""

    def __call__(self, tensor: ndarray)->ndarray:
        return tensor.flatten()

