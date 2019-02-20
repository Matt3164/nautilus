from numpy.core.multiarray import ndarray

from nautilus.transform.transform import Transform


class Model(Transform):
    """"""

    def __call__(self, tensor: ndarray)->ndarray:
        raise NotImplementedError


